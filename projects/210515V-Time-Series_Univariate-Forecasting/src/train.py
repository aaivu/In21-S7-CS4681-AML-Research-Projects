import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from data_loader import data_provider
from models import DLinear
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

class Trainer:
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = DLinear.Model(self.args).float()
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.MSELoss()

    def _process_one_batch(self, batch_x, batch_y):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        outputs = self.model(batch_x)
        outputs = outputs[:, -self.args.pred_len:, :]
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

        return outputs, batch_y

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(vali_loader):
                pred, true = self._process_one_batch(batch_x, batch_y)
                loss = criterion(pred, true).item()
                total_loss.append(loss)
        self.model.train()
        return np.average(total_loss)

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        
        path = os.path.join(self.args.checkpoints, self.setting)
        if not os.path.exists(path):
            os.makedirs(path)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            train_loss = []
            self.model.train()

            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                optimizer.zero_grad()
                pred, true = self._process_one_batch(batch_x, batch_y)
                loss = criterion(pred, true)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch + 1}, Cost time: {time.time() - epoch_time:.4f}s")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            
            print(f"Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

    def predict(self, load_model=False):
        if load_model:
            path = os.path.join(self.args.checkpoints, self.setting)
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        _, test_loader = self._get_data(flag='test')
        preds = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                pred, _ = self._process_one_batch(batch_x, batch_y)
                preds.append(pred.detach().cpu().numpy())
        
        return np.concatenate(preds, axis=0)

