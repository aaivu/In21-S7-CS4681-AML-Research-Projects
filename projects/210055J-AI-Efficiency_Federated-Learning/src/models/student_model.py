import copy
import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, global_model):
        super(StudentModel, self).__init__()
        # Clone the architecture from the global model to create a new student
        self.model = copy.deepcopy(global_model)
        
    def forward(self, x):
        return self.model(x)
