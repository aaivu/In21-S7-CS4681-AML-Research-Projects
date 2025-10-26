import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import Dataset

class BPRDataset(Dataset):
    def __init__(self, df, n_items):
        self.df = df
        self.n_items = n_items
        
        self.users = torch.LongTensor(df['user_id'].values)
        self.items = torch.LongTensor(df['track_id'].values)
        
        # Pre-compute positive sets
        self.user_pos_items = df.groupby('user_id')['track_id'].apply(set).to_dict()
        
        # Pre-sample negatives for efficiency (optional: sample 100 negatives per user)
        self.user_neg_items = {}
        for user in self.user_pos_items:
            pos_set = self.user_pos_items[user]
            negatives = set()
            while len(negatives) < 100:  # Pre-sample 100 negatives per user
                neg = random.randint(0, n_items - 1)
                if neg not in pos_set:
                    negatives.add(neg)
            self.user_neg_items[user] = list(negatives)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        
        # Use pre-sampled negatives if available, else fallback
        if user.item() in self.user_neg_items and self.user_neg_items[user.item()]:
            neg_item = random.choice(self.user_neg_items[user.item()])
        else:
            # Fallback with limited attempts
            max_attempts = 50
            for _ in range(max_attempts):
                neg_item = random.randint(0, self.n_items - 1)
                if neg_item not in self.user_pos_items.get(user.item(), set()):
                    break
            else:
                neg_item = random.randint(0, self.n_items - 1)  # Fallback
        
        return user, pos_item, torch.LongTensor([neg_item]).squeeze()

def get_metrics(model, test_data, k, device):
    model.eval()
    
    recalls = []
    ndcgs = []

    with torch.no_grad():
        final_users, final_items = model.computer()
        
        for user_id, true_items in tqdm(test_data.items(), desc="Evaluating"):
            # Ensure user_id is an integer for indexing
            if not isinstance(user_id, int): continue

            user_emb = final_users[user_id].unsqueeze(0)
            scores = torch.matmul(user_emb, final_items.T).squeeze()
            
            # Exclude items the user has already interacted with in training
            # This is a common practice to not recommend already-known items
            # For simplicity, we are not implementing it here, but it's a good future step.
            
            _, top_k_items = torch.topk(scores, k=k)
            top_k_items = top_k_items.cpu().numpy()
            
            # Recall
            hits = sum([1 for item in top_k_items if item in true_items])
            recalls.append(hits / len(true_items))
            
            # NDCG
            dcg_ranks = [i + 1 for i, item in enumerate(top_k_items) if item in true_items]
            if not dcg_ranks: continue
            
            dcg = sum([1 / np.log2(rank + 1) for rank in dcg_ranks])
            idcg = sum([1 / np.log2(i + 2) for i in range(len(true_items))])
            ndcgs.append(dcg / (idcg + 1e-8))

    recall_mean = np.mean(recalls) if recalls else 0.0
    ndcg_mean = np.mean(ndcgs) if ndcgs else 0.0
    
    return recall_mean, ndcg_mean

    return np.mean(recalls), np.mean(ndcgs)