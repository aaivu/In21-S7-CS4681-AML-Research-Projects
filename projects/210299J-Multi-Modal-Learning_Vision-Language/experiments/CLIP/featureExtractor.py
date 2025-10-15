import torch, torchvision.transforms as T
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load CLIP (ViT-B/32 is lighter)
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
model = model.to(device).eval()

# 2) Your toy dataset
samples = [
    ("images/golden.jpeg", 0),
    ("images/green car in street.jpeg", 1),
    ("images/old black car in town.jpeg", 1),

]
num_classes = 2

class ImgDS(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        x = preprocess(Image.open(p).convert("RGB"))
        return x, y

train_loader = DataLoader(ImgDS(samples), batch_size=2, shuffle=True)

# 3) Freeze CLIP params
for p in model.parameters():
    p.requires_grad_(False)

# 4) Trainable linear head
clf = nn.Linear(model.visual.output_dim, num_classes).to(device)

opt = optim.AdamW(clf.parameters(), lr=1e-3)
ce  = nn.CrossEntropyLoss()

# 5) Training loop
for epoch in range(5):  # fewer epochs, tiny dataset
    tot = 0.0
    for x, y in train_loader:
        x, y = x.to(device), torch.tensor(y).to(device)

        with torch.no_grad():
            f = model.encode_image(x)           # extract CLIP features
            f = f / f.norm(dim=-1, keepdim=True)

        logits = clf(f)                         # linear head
        loss = ce(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        tot += loss.item() * x.size(0)

    print(f"epoch {epoch+1}: loss={tot/len(train_loader.dataset):.4f}")
