import os
import time
import argparse
from tqdm import tqdm

from sampler import WarpSampler
from modules import SASRec  # Assuming converted PyTorch model is in modules.py
from util import *

import torch
import torch.nn as nn
import torch.optim as optim


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)

args = parser.parse_args()


if __name__ == '__main__':
    # Create directory for logs
    if not os.path.isdir(args.dataset + '_' + args.train_dir):
        os.makedirs(args.dataset + '_' + args.train_dir)
    with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
        f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
    f.close()

    # Load dataset
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print('average sequence length: %.2f' % (cc / len(user_train)))

    f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Sampler
    sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    # Model
    model = SASRec(usernum, itemnum, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    T = 0.0
    t0 = time.time()

    #try:
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        print('trained')
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            print("num_batch:", num_batch)
            print("step:", step)
            u, seq, pos, neg = sampler.next_batch()

            u = torch.tensor(np.array(u), dtype=torch.long, device=device)
            seq = torch.tensor(np.array(seq), dtype=torch.long, device=device)
            pos = torch.tensor(np.array(seq), dtype=torch.long, device=device)
            neg = torch.tensor(np.array(seq), dtype=torch.long, device=device)

            loss = model.calculate_loss(u, seq, pos, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time() - t0
            T += t1
            print('Evaluating', end=' ')
            t_test = evaluate(model, dataset, args, device)
            t_valid = evaluate_valid(model, dataset, args, device)
            print('')
            print('epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
                epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1]))

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()
    # except:
    #     sampler.close()
    #     f.close()
    #     exit(1)

    f.close()
    sampler.close()
    print("Done")
