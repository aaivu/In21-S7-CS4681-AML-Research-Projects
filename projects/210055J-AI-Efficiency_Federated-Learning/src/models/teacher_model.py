import copy
import torch
import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(self, global_model):
        super(TeacherModel, self).__init__()
        # Clone the global model and freeze its parameters
        self.model = copy.deepcopy(global_model)
        self.freeze()

    def freeze(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return self

    def forward(self, x):
        return self.model(x)
