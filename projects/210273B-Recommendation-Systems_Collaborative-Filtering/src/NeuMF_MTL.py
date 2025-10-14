import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from evaluate_MTL import evaluate_model
from Dataset import Dataset
from time import time
import argparse

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF with Data Augmentation and Multi-Task Learning.")
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
                        help="MLP layers.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0]',
                        help="Regularization for each MLP layer.")
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
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Embedding dropout rate for stochastic data augmentation.')
    parser.add_argument('--ssl_weight', type=float, default=0.1,
                        help='Weight for self-supervised loss in multi-task objective.')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for contrastive loss.')
    parser.add_argument('--pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for NeuMF_MTL. If empty, no pretrain will be used')
    # Learning rate scheduler arguments
    parser.add_argument('--scheduler', nargs='?', default='',
                        help='Learning rate scheduler: step, cosine, plateau, exponential')
    parser.add_argument('--step_size', type=int, default=20,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='Decay factor for learning rate schedulers')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau scheduler')
    return parser.parse_args()

def init_normal(tensor):
    if tensor is not None:
        nn.init.normal_(tensor, mean=0.0, std=0.01)

class NeuMF_MTL(nn.Module):
    def __init__(self, num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0, dropout=0.2):
        super(NeuMF_MTL, self).__init__()
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
        self.embedding_dropout = nn.Dropout(dropout)
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
        # Self-supervised projection head
        self.ssl_head = nn.Sequential(
            nn.Linear(mf_dim + input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, user_indices, item_indices, augment=False):
        mf_user_latent = self.mf_user_embedding(user_indices)
        mf_item_latent = self.mf_item_embedding(item_indices)
        mlp_user_latent = self.mlp_user_embedding(user_indices)
        mlp_item_latent = self.mlp_item_embedding(item_indices)
        if augment:
            mf_user_latent = self.embedding_dropout(mf_user_latent)
            mf_item_latent = self.embedding_dropout(mf_item_latent)
            mlp_user_latent = self.embedding_dropout(mlp_user_latent)
            mlp_item_latent = self.embedding_dropout(mlp_item_latent)
        mf_vector = mf_user_latent * mf_item_latent
        mlp_vector = torch.cat([mlp_user_latent, mlp_item_latent], dim=-1)
        mlp_vector = self.mlp_layers(mlp_vector)
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=-1)
        prediction = self.sigmoid(self.predict_layer(predict_vector)).squeeze()
        return prediction, torch.concat([mf_user_latent, mlp_user_latent]), torch.concat([mf_item_latent, mlp_item_latent])

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


def contrastive_loss(z1, z2, temperature=0.5, device='cpu'):
    # Simple contrastive loss for self-supervised learning
    z1 = nn.functional.normalize(z1, dim=-1)
    z2 = nn.functional.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    labels = torch.cat([torch.arange(batch_size, device=device) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    logits = torch.cat([positives, negatives], dim=1) / temperature
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
    loss = nn.CrossEntropyLoss()(logits, targets)
    return loss


def load_pretrain_model(model, pretrain_model_path):
    if pretrain_model_path == '':
        print("No pretrain model path provided.")
        return
    print('Loading pretrained model from %s' % pretrain_model_path)
    pretrain_dict = torch.load(pretrain_model_path)
    model_dict = model.state_dict()
    # Filter out unnecessary keys
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and 'ssl_head' not in k}
    # Overwrite entries in the existing state dict
    model_dict.update(pretrain_dict) 
    # Load the new state dict
    model.load_state_dict(model_dict)
    print('Pretrained model loaded.')


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
    dropout = args.dropout
    ssl_weight = args.ssl_weight
    temperature = args.temperature
    pretrain_model = args.pretrain
    # Scheduler arguments
    scheduler_type = args.scheduler
    step_size = args.step_size
    gamma = args.gamma
    patience = args.patience

    # Set seed for reproducibility
    torch.manual_seed(2)
    np.random.seed(23)

    topK = 10
    evaluation_threads = 1
    print("NeuMF-MTL arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_NeuMF_MTL_%d_%s_%d.pt' % (args.dataset, mf_dim, args.layers, int(time()))

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" %
          (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    # Build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model
    model = NeuMF_MTL(num_users, num_items, mf_dim, layers, reg_layers, reg_mf, dropout).to(device)
    if pretrain_model != '':
        print("Using pre-trained model: ", pretrain_model)
        load_pretrain_model(model, pretrain_model)

    # Define optimizer
    if learner.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif learner.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif learner.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Initialize learning rate scheduler
    scheduler = None
    if scheduler_type.lower() == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"Using StepLR scheduler: step_size={step_size}, gamma={gamma}")
    elif scheduler_type.lower() == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01)
        print(f"Using CosineAnnealingLR scheduler: T_max={num_epochs}, eta_min={learning_rate * 0.01}")
    elif scheduler_type.lower() == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=gamma, 
                                                       patience=patience, verbose=True)
        print(f"Using ReduceLROnPlateau scheduler: factor={gamma}, patience={patience}")
    elif scheduler_type.lower() == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        print(f"Using ExponentialLR scheduler: gamma={gamma}")
    elif scheduler_type == '':
        print("No learning rate scheduler used")
    else:
        print(f"Unknown scheduler type: {scheduler_type}. No scheduler will be used.")
    
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
        print("Training epoch %d with %d instances..." % (epoch, len(labels)))
        loader = DataLoader(dataset_torch, batch_size=batch_size, shuffle=True)

        model.train()
        epoch_loss = 0.0
        for u, i, l in loader:
            optimizer.zero_grad()
            # Create two augmented views and compute predictions
            pred_aug_1, user_embedding_1, item_embedding_1  = model(u, i, augment=True)
            pred_aug_2, user_embedding_2, item_embedding_2  = model(u, i, augment=True)

            # Compute losses
            rec_loss_1 = criterion(pred_aug_1, l)
            rec_loss_2 = criterion(pred_aug_2, l)
            rec_loss = (rec_loss_1 + rec_loss_2) / 2
            user_ssl_loss = contrastive_loss(user_embedding_1, user_embedding_2, temperature=temperature, device=device) / u.size(0)
            item_ssl_loss = contrastive_loss(item_embedding_1, item_embedding_2, temperature=temperature, device=device) / u.size(0)
            ssl_loss = user_ssl_loss + item_ssl_loss
            total_loss = rec_loss + ssl_weight * ssl_loss
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item() * u.size(0)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            model.eval()
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads, device=device)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            current_lr = optimizer.param_groups[0]['lr']
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f, lr = %.6f [%.1f s]'
                  % (epoch, t2-t1, hr, ndcg, epoch_loss/len(loader), current_lr, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    torch.save(model.state_dict(), model_out_file)
        
        # Step the scheduler
        if scheduler is not None:
            if scheduler_type.lower() == 'plateau':
                # For ReduceLROnPlateau, we need to pass the metric to monitor
                model.eval()
                if epoch % verbose != 0:  # If we haven't evaluated this epoch yet
                    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads, device=device)
                    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                scheduler.step(hr)  # Use HR as the metric to monitor
            else:
                scheduler.step()

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best NeuMF-MTL model is saved to %s" % (model_out_file))