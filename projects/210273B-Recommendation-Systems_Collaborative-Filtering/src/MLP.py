'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def init_normal(tensor):
    if tensor is not None:
        nn.init.normal_(tensor, mean=0.0, std=0.01)

class MLP(nn.Module):
    def __init__(self, num_users, num_items, layers=[20,10], reg_layers=[0,0]):
        super(MLP, self).__init__()
        assert len(layers) == len(reg_layers)
        embedding_dim = int(layers[0] // 2)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        init_normal(self.user_embedding.weight)
        init_normal(self.item_embedding.weight)
        mlp_layers = []
        input_dim = layers[0]
        for idx in range(1, len(layers)):
            mlp_layers.append(nn.Linear(input_dim, layers[idx]))
            mlp_layers.append(nn.ReLU())
            input_dim = layers[idx]
        self.mlp_layers = nn.Sequential(*mlp_layers)
        self.predict_layer = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.predict_layer.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_latent = self.user_embedding(user_indices)
        item_latent = self.item_embedding(item_indices)
        vector = torch.cat([user_latent, item_latent], dim=-1)
        vector = self.mlp_layers(vector)
        prediction = self.sigmoid(self.predict_layer(vector))
        return prediction.squeeze()

def get_train_instances(train, num_negatives, num_items):
    user_input, item_input, labels = [], [], []
    for (u, i) in train.keys():
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset_name = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("MLP arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.pt' % (args.dataset, args.layers, int(time()))

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = MLP(num_users, num_items, layers, reg_layers).to(device)
    if learner.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif learner.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif learner.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time()-t1))

    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, num_negatives, num_items)
        user_input = torch.LongTensor(user_input).to(device)
        item_input = torch.LongTensor(item_input).to(device)
        labels = torch.FloatTensor(labels).to(device)
        dataset_torch = TensorDataset(user_input, item_input, labels)
        loader = DataLoader(dataset_torch, batch_size=batch_size, shuffle=True)

        model.train()
        epoch_loss = 0.0
        for u, i, l in loader:
            optimizer.zero_grad()
            output = model(u, i)
            loss = criterion(output, l)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * u.size(0)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            model.eval()
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2-t1, hr, ndcg, epoch_loss/len(loader), time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    torch.save(model.state_dict(), model_out_file)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))
