# Text → Image retrieval (search a folder of images with a query)

import os, glob
from PIL import Image
import torch, torchvision.transforms as T
import open_clip
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

# 1) Index all images in a folder
img_paths = []
img_embeds = []
with torch.no_grad():
    for p in tqdm(glob.glob("images/*.*")):  # put your folder
        try:
            im = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            f = model.encode_image(im)
            f = f / f.norm(dim=-1, keepdim=True)
            img_paths.append(p)
            img_embeds.append(f.cpu())
        except Exception as e:
            print("skip", p, e)

img_embeds = torch.cat(img_embeds, dim=0)  # (N, D)

# 2) Query with text
query = "a red sports car on a street"
with torch.no_grad():
    q = tok([query]).to(device)
    qf = model.encode_text(q)
    qf = qf / qf.norm(dim=-1, keepdim=True)
    sims = (qf.cpu() @ img_embeds.T).squeeze(0)  # cosine similarity

topk = sims.topk(k=min(10, len(img_paths)))
for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
    print(f"{score:.3f}  {img_paths[idx]}")
