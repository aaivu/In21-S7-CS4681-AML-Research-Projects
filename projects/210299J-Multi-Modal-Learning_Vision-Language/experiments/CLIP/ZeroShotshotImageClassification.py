import torch, torchvision.transforms as T
from PIL import Image
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Load CLIP
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device).eval()

# 2) Load an image
img = Image.open("download.jpeg").convert("RGB")  # put a real path here
image = preprocess(img).unsqueeze(0).to(device)

# 3) Write your label prompts (templates help)
labels = [
    "dog runing in the park",
    "a photo of a tabby cat",
    "a photo of a car",
    "a photo of a person",
]
text = tokenizer(labels).to(device)

# 4) Get logits and probabilities
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features  = model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
    # cosine similarity scaled by CLIP logit_scale
    logit_scale = model.logit_scale.exp()
    logits = logit_scale * image_features @ text_features.T
    probs = logits.softmax(dim=-1).squeeze(0).tolist()

for lbl, p in sorted(zip(labels, probs), key=lambda x: -x[1]):
    print(f"{lbl:30s}  {p:.3f}")
