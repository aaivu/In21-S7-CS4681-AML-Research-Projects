import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import DemandVCTKDataset
from src.model.dpcrn_two_stage import DPCRN_TwoStage
from src.loss import multi_loss
from src.utils import stft
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clean, noise = '/content/MiniVCTK/', '/content/DEMAND_noise/'

ds = DemandVCTKDataset(clean, noise)
dl = DataLoader(ds, batch_size=2, shuffle=True)
model = DPCRN_TwoStage().to(device)
opt = optim.Adam(model.parameters(), lr=1e-4)

for ep in range(3):
    model.train()
    tot = 0
    for n, c in tqdm(dl):
        n, c = n.to(device), c.to(device)
        Ns, Cs = stft(n), stft(c)
        mag, ph = torch.abs(Ns), torch.angle(Ns)

        # 🔍 Step 1: Diagnostic shape prints
        print(f"mag shape: {mag.shape}, phase shape: {ph.shape}")

        r, i = model(mag, ph)
        enh = torch.complex(r, i)
        real_tgt, imag_tgt = Cs.real, Cs.imag
        loss = torch.nn.functional.mse_loss(enh.real, real_tgt) + \
        torch.nn.functional.mse_loss(enh.imag, imag_tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += loss.item()

    print(f"Epoch {ep+1}: {tot/len(dl):.4f}")

import os
ckpt_dir = "/content/dpcrn_project/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(ckpt_dir, "dpcrn_two_stage_ep3.pth")
torch.save(model.state_dict(), ckpt_path)
print(f"💾 Saved checkpoint to {ckpt_path}")

print('✅ Training complete')
