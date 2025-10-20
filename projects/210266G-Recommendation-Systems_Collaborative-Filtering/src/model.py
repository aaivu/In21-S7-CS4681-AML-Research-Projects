import torch
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, n_layers, graph):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.graph = graph

        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def computer(self):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(self.graph, all_emb)
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
        
        return users_emb, pos_items_emb, neg_items_emb

    def get_user_ratings(self, users):
        final_users, final_items = self.computer()
        user_embeddings = final_users[users]
        item_embeddings = final_items
        ratings = torch.matmul(user_embeddings, item_embeddings.T)
        return ratings