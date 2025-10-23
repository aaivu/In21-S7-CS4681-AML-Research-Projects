import numpy as np
import torch

def evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads=1, device='cpu'):
    hits, ndcgs = [], []
    model.eval()
    with torch.no_grad():
        for idx in range(len(testRatings)):
            rating = testRatings[idx]
            items = testNegatives[idx]
            u = torch.LongTensor([rating[0]] * (len(items) + 1)).to(device)
            i = torch.LongTensor([rating[1]] + items).to(device)
            # Only use the prediction output from the model (first element of tuple)
            scores = model(u, i)[0].cpu().numpy()
            rank = scores.argsort()[::-1]
            rank_index = np.where(rank == 0)[0][0]
            if rank_index < topK:
                hits.append(1)
                ndcgs.append(1 / np.log2(rank_index + 2))
            else:
                hits.append(0)
                ndcgs.append(0)
    return hits, ndcgs