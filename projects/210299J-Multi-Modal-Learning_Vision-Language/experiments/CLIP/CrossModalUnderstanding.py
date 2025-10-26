# Cross-modal Understanding (Matching).py
import open_clip
from PIL import Image

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tok = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

# Example: does this caption describe the image?
image = preprocess(Image.open("images/golden.jpeg").convert("RGB")).unsqueeze(0).to(device)
texts = tok(["a dog running on grass", "a cat sleeping on a sofa", "a golden retriever sitting on a beach in a sunset whixh has a collar"]).to(device)

with torch.no_grad():
    img_feat = model.encode_image(image); img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat = model.encode_text(texts);  txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

    sims = (img_feat @ txt_feat.T).squeeze(0)
    for txt, score in zip(["dog running", "cat sleeping", "dog sitting"], sims.tolist()):
        print(f"{txt:25s}  similarity={score:.3f}")
