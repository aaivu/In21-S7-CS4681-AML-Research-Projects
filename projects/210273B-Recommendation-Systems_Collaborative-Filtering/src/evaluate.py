'''
Created on Apr 15, 2016
Evaluate the performance of Top-K recommendation:
    Protocol: leave-1-out evaluation
    Measures: Hit Ratio and NDCG
    (more details are in: Xiangnan He, et al. Fast Matrix Factorization for Online Recommendation with Implicit Feedback. SIGIR'16)

@author: hexiangnan
'''
import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
import torch

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None
_device = 'cpu'

def evaluate_model(model, testRatings, testNegatives, K, num_thread, device='cpu'):
    global _model, _testRatings, _testNegatives, _K, _device
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
    _device = device

    hits, ndcgs = [], []
    # Batch evaluation for speed
    for idx in range(len(_testRatings)):
        rating = _testRatings[idx]
        items = list(_testNegatives[idx])
        u = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        users = np.full(len(items), u, dtype='int64')
        items_arr = np.array(items, dtype='int64')
        _model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor(users).to(_device)
            item_tensor = torch.LongTensor(items_arr).to(_device)
            scores = _model(user_tensor, item_tensor).cpu().numpy()
        map_item_score = {item: score for item, score in zip(items, scores)}
        items.pop()
        ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
        hr = getHitRatio(ranklist, gtItem)
        ndcg = getNDCG(ranklist, gtItem)
        hits.append(hr)
        ndcgs.append(ndcg)
    return (hits, ndcgs)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = list(_testNegatives[idx])
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype='int64')
    items_arr = np.array(items, dtype='int64')
    # PyTorch forward pass
    _model.eval()
    with torch.no_grad():
        user_tensor = torch.LongTensor(users).to(_device)
        item_tensor = torch.LongTensor(items_arr).to(_device)
        scores = _model(user_tensor, item_tensor).cpu().numpy()
    for i, item in enumerate(items):
        map_item_score[item] = scores[i]
    items.pop()

    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i, item in enumerate(ranklist):
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0
