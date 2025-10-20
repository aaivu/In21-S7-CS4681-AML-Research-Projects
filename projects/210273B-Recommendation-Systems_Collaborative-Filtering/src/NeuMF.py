import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
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

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
        super(NeuMF, self).__init__()
        assert len(layers) == len(reg_layers)
        embedding_dim = int(layers[0] // 2)
        self.mf_user_embedding = nn.Embedding(num_users, mf_dim)
        self.mf_item_embedding = nn.Embedding(num_items, mf_dim)
        init_normal(self.mf_user_embedding.weight)
        init_normal(self.mf_item_embedding.weight)
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        init_normal(self.mlp_user_embedding.weight)
        init_normal(self.mlp_item_embedding.weight)
        mlp_layers = []
        input_dim = layers[0]
        for idx in range(1, len(layers)):
            mlp_layers.append(nn.Linear(input_dim, layers[idx]))
            mlp_layers.append(nn.ReLU())
            input_dim = layers[idx]
        self.mlp_layers = nn.Sequential(*mlp_layers)
        self.predict_layer = nn.Linear(mf_dim + input_dim, 1)
        nn.init.xavier_uniform_(self.predict_layer.weight)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        mf_user_latent = self.mf_user_embedding(user_indices)
        mf_item_latent = self.mf_item_embedding(item_indices)
        mf_vector = mf_user_latent * mf_item_latent
        mlp_user_latent = self.mlp_user_embedding(user_indices)
        mlp_item_latent = self.mlp_item_embedding(item_indices)
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.sigmoid(self.predict_layer(predict_vector))
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
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    print("NeuMF arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.pt' % (args.dataset, mf_dim, args.layers, int(time()))

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
    model = NeuMF(num_users, num_items, mf_dim, layers, reg_layers, reg_mf).to(device)
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
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads, device=device)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f' % (hr, ndcg))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        torch.save(model.state_dict(), model_out_file)

    # Training model
    for epoch in range(num_epochs):
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
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads, device=device)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2-t1, hr, ndcg, epoch_loss/len(loader), time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    torch.save(model.state_dict(), model_out_file)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best NeuMF model is saved to %s" % (model_out_file))
