import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix
import torch

from content_utils import process_content_features

class Loader:
    def __init__(self, path):
        self.path = path
        
        print("Loading data...")
        # Use the new, preprocessed files
        self.train_df = pd.read_csv(f"{self.path}/train.csv")
        self.val_df = pd.read_csv(f"{self.path}/val.csv")
        self.test_df = pd.read_csv(f"{self.path}/test.csv")
        
        # Build the user and item maps ONLY from the training data
        self.users = self.train_df['user_id'].unique()
        self.items = self.train_df['track_id'].unique()
        
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        self.user_map = {id: i for i, id in enumerate(self.users)}
        self.item_map = {id: i for i, id in enumerate(self.items)}

        self.train_df['user_id'] = self.train_df['user_id'].map(self.user_map)
        self.train_df['track_id'] = self.train_df['track_id'].map(self.item_map)
        
        self.val_data = self._get_positive_lists(self.val_df)
        self.test_data = self._get_positive_lists(self.test_df)
        
        print(f"Data loaded: {self.n_users} users, {self.n_items} items.")
        self.graph = self._create_sparse_graph()
        print("Graph created.")
        self.content_features = process_content_features(path, self.item_map)
        print("Content features loaded and processed.")

    def _get_positive_lists(self, df):
        # Map string IDs to integer indices
        users = df['user_id'].map(self.user_map)
        items = df['track_id'].map(self.item_map)
        
        # More robustly create the dictionary, dropping any potential NaNs
        df_mapped = pd.DataFrame({'user_id': users, 'track_id': items}).dropna()
        
        positive_lists = {}
        for _, row in df_mapped.iterrows():
            u = int(row['user_id'])
            i = int(row['track_id'])
            if u not in positive_lists:
                positive_lists[u] = []
            positive_lists[u].append(i)
        return positive_lists

    def _create_sparse_graph(self):
        R = dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        for _, row in self.train_df.iterrows():
            R[row['user_id'], row['track_id']] = 1.0
        R = R.tocsr()
        
        # Debugging: Check for isolated nodes
        user_degrees = R.sum(axis=1)
        item_degrees = R.sum(axis=0)
        print(f"Users with no interactions: {(user_degrees == 0).sum()}")
        print(f"Items with no interactions: {(item_degrees == 0).sum()}")
        
        adj_mat = dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        
        row_sum = np.array(adj_mat.sum(axis=1)).flatten()
        # Handle zero degrees
        d_inv = np.zeros_like(row_sum, dtype=np.float32)
        non_zero = row_sum > 0
        d_inv[non_zero] = np.power(row_sum[non_zero], -0.5)
        
        d_mat = csr_matrix((d_inv, (np.arange(len(d_inv)), np.arange(len(d_inv)))), shape=(len(d_inv), len(d_inv)))
        
        norm_adj = adj_mat.dot(d_mat)
        norm_adj = norm_adj.tocoo()
        
        values = torch.FloatTensor(norm_adj.data)
        indices = torch.LongTensor(np.vstack((norm_adj.row, norm_adj.col)))
        
        graph = torch.sparse.FloatTensor(indices, values, torch.Size(norm_adj.shape))
        return graph

