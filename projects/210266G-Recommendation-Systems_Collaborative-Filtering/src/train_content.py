import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from dataset import Loader
from content_gcn import ContentGCN
from utils import BPRDataset, get_metrics

def bpr_loss(users_emb, pos_items_emb, neg_items_emb, reg_weight):
    pos_scores = torch.sum(users_emb * pos_items_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_items_emb, dim=1)
    
    bpr = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-9))
    
    reg_loss = (torch.norm(users_emb, p=2).pow(2) +
                torch.norm(pos_items_emb, p=2).pow(2) +
                torch.norm(neg_items_emb, p=2).pow(2)) / len(users_emb)
    
    return bpr + reg_weight * reg_loss

def main(args):
    # --- Experiment Setup ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"ContentGCN_dim{args.embed_dim}_layers{args.n_layers}_lr{args.lr}_reg{args.reg}_{timestamp}"
    results_path = os.path.join('results/contentgcn_logs', experiment_name)
    os.makedirs(results_path, exist_ok=True)
    
    log_file_path = os.path.join(results_path, 'log.txt')
    log_file = open(log_file_path, 'w')

    def log_and_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_and_print(f"Using device: {device}")
    log_and_print(f"Starting experiment: {experiment_name}")
    log_and_print("Hyperparameters:\n" + str(args))

    # --- Data and Model ---
    data_loader = Loader(path=args.data_path)
    graph = data_loader.graph.to(device)
    
    model = ContentGCN(
        n_users=data_loader.n_users,
        n_items=data_loader.n_items,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        graph=graph,
        content_features=data_loader.content_features.to(device)
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_recall = -1.0
    best_model_path = os.path.join(results_path, 'best_model.pth')

    # --- Memory-Efficient Dataloader ---
    train_dataset = BPRDataset(df=data_loader.train_df, n_items=data_loader.n_items)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # --- Training Loop ---
    for epoch in range(args.epochs):
        model.train()
        
        epoch_loss = 0.0
        
        for users, pos_items, neg_items in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
            optimizer.zero_grad()
            users_emb, pos_items_emb, neg_items_emb, content_loss = model(users, pos_items, neg_items)
            loss = bpr_loss(users_emb, pos_items_emb, neg_items_emb, args.reg) + content_loss
            # loss = bpr_loss(users_emb, pos_items_emb, neg_items_emb, args.reg) + content_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        log_and_print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(dataloader):.4f}")

        # --- Validation ---
        is_last_epoch = (epoch + 1) == args.epochs
        if ((epoch + 1) % args.eval_every == 0) or is_last_epoch:
            recall, ndcg = get_metrics(model, data_loader.val_data, args.k, device)
            log_and_print(f"Validation Recall@{args.k}: {recall:.4f}, NDCG@{args.k}: {ndcg:.4f}")
            
            if recall > best_recall:
                best_recall = recall
                torch.save(model.state_dict(), best_model_path)
                log_and_print(f"Saved new best model to {best_model_path}")

    # --- Final Test Evaluation ---
    log_and_print("\n--- Final Test Evaluation ---")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        log_and_print(f"Loaded best model from {best_model_path} for final test.")
        test_recall, test_ndcg = get_metrics(model, data_loader.test_data, args.k, device)
        log_and_print(f"Test Recall@{args.k}: {test_recall:.4f}, NDCG@{args.k}: {test_ndcg:.4f}")
    else:
        log_and_print("No best model was saved. Skipping final test evaluation.")

    log_file.close()
    print(f"Training complete. Results saved in {results_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContentGCN Training')
    parser.add_argument('--data_path', type=str, default='data', help='Dataset directory relative to project root')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--reg', type=float, default=1e-4, help='L2 Regularization')
    parser.add_argument('--embed_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--k', type=int, default=20, help='K for evaluation metrics')
    parser.add_argument('--eval_every', type=int, default=5, help='Evaluate every N epochs')
    args = parser.parse_args()
    main(args)