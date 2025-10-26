import torch
from PIL import Image
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load CLIP
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

# 2. Candidate text descriptions (your "caption database")
captions = [
    "a photo of a golden retriever",
    "a photo of a husky in the snow",
    "a black sports car parked in the city",
    "a person riding a bicycle",
    "a big cargo ship in the ocean",
    "a cat sleeping on a couch",
]

# 3. Input image (replace with your image path)
image_path = "images/golden.jpeg"
image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

# 4. Encode features
with torch.no_grad():
    img_feat = model.encode_image(image)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

    txt_tokens = tokenizer(captions).to(device)
    txt_feats  = model.encode_text(txt_tokens)
    txt_feats  = txt_feats / txt_feats.norm(dim=-1, keepdim=True)

    # 5. Compute similarity (image vs all captions)
    sims = (img_feat @ txt_feats.T).squeeze(0)

# 6. Rank and show top-k captions
topk = sims.topk(k=6)  # top 3 captions
print(f"\nImage: {image_path}")
for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
    print(f"{score:.3f}  {captions[idx]}")
