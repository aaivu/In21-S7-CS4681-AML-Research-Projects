import numpy as np
import random
from collections import defaultdict
import torch


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    # assume user/item index starting from 1
    with open('data/%s.txt' % fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    return [user_train, user_valid, user_test, usernum, itemnum]


def evaluate(model, dataset, args, device, sample_users=10000):
    [train, valid, test, usernum, itemnum] = dataset
    model.eval()

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    users = list(range(1, usernum + 1))
    if usernum > sample_users:
        users = random.sample(users, sample_users)

    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(test[u]) < 1:
                continue

            # Prepare sequence
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(train[u])
            rated.add(0)
            item_idx = [test[u][0]]  # positive item
            while len(item_idx) < 101:  # add 100 negatives
                t = np.random.randint(1, itemnum + 1)
                if t not in rated:
                    item_idx.append(t)
                    rated.add(t)

            # Tensors
            u_tensor = torch.tensor([u], dtype=torch.long, device=device)
            seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            item_tensor = torch.tensor(item_idx, dtype=torch.long, device=device)

            # Predict scores (NumPy array)
            scores = model.predict(u_tensor, seq_tensor, item_tensor)

            # Get rank of the positive item (index 0 in item_idx)
            sorted_indices = np.argsort(scores)[::-1]
            rank = int(np.where(sorted_indices == 0)[0][0])

            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args, device, sample_users=10000):
    [train, valid, test, usernum, itemnum] = dataset
    model.eval()

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    users = list(range(1, usernum + 1))
    if usernum > sample_users:
        users = random.sample(users, sample_users)

    with torch.no_grad():
        for u in users:
            if len(train[u]) < 1 or len(valid[u]) < 1:
                continue

            # Prepare sequence
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break

            rated = set(train[u])
            rated.add(0)
            item_idx = [valid[u][0]]  # positive item
            while len(item_idx) < 101:
                t = np.random.randint(1, itemnum + 1)
                if t not in rated:
                    item_idx.append(t)
                    rated.add(t)

            u_tensor = torch.tensor([u], dtype=torch.long, device=device)
            seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)
            item_tensor = torch.tensor(item_idx, dtype=torch.long, device=device)

            scores = model.predict(u_tensor, seq_tensor, item_tensor)

            sorted_indices = np.argsort(scores)[::-1]
            rank = int(np.where(sorted_indices == 0)[0][0])

            valid_user += 1
            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    return NDCG / valid_user, HT / valid_user
