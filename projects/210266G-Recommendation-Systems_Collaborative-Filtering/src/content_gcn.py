import torch
import torch.nn as nn

class ContentGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, n_layers, graph, content_features):
        super(ContentGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.graph = graph
        
        device = self.graph.device
        self.register_buffer('content_features', content_features.to(device))

        self.content_projection = nn.Linear(
            in_features=self.content_features.shape[1], 
            out_features=self.embed_dim
        ).to(device)
        
        # Gating mechanism
        self.gate_layer = nn.Sequential(
            nn.Linear(in_features=self.content_features.shape[1], out_features=1),
            nn.Sigmoid()
        ).to(device)

        self.item_embedding_collaborative = nn.Embedding(n_items, embed_dim)
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)  # Tune between 0.1-0.3 if needed
        
        # Content loss weight
        self.content_loss_weight = 0.1  # Tune between 0.01-0.5
        
        nn.init.xavier_uniform_(self.item_embedding_collaborative.weight)
        nn.init.xavier_uniform_(self.user_embedding.weight)

    def computer(self):
        users_emb = self.dropout(self.user_embedding.weight)  # Apply dropout
        
        items_emb_collab = self.dropout(self.item_embedding_collaborative.weight)
        items_emb_content = self.content_projection(self.content_features)
        
        gate_values = self.gate_layer(self.content_features)
        items_emb = ((1 - gate_values) * items_emb_collab) + (gate_values * items_emb_content)
        
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        for _ in range(self.n_layers):
            layer_emb = torch.sparse.mm(self.graph, embs[-1])
            layer_emb = nn.LayerNorm(self.embed_dim).to(self.graph.device)(layer_emb)  # Normalize
            all_emb = layer_emb + embs[-1]  # Residual connection
            embs.append(all_emb)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.n_users, self.n_items])
        return users, items

    def forward(self, users, pos_items, neg_items):
        final_users, final_items = self.computer()
        
        users_emb = final_users[users]
        pos_items_emb = final_items[pos_items]
        neg_items_emb = final_items[neg_items]
        
        # Auxiliary loss: Align final item embeddings with content features
        projected_content = self.content_projection(self.content_features)
        content_loss = nn.MSELoss()(final_items, projected_content) * self.content_loss_weight
        
        return users_emb, pos_items_emb, neg_items_emb, content_loss