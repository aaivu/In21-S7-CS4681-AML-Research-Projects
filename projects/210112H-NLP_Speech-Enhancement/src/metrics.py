from pesq import pesq
from pystoi import stoi
import numpy as np
def compute_metrics(clean,enh,sr=16000):
    c=clean.cpu().numpy(); e=enh.cpu().numpy()
    pesq_val=pesq(sr,c,e,'wb')
    stoi_val=stoi(c,e,sr,extended=False)
    sdr=10*np.log10(np.sum(c**2)/np.sum((c-e)**2))
    return pesq_val,stoi_val,sdr
