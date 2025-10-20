import torch
def multi_loss(y_true,y_pred,alpha=0.6,beta=0.4):
    Lmag=((torch.abs(y_true)-torch.abs(y_pred))**2).mean()
    Lri=((y_true-y_pred)**2).mean()
    return alpha*Lmag+beta*Lri
