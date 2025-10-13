# pip install transformers pillow torch

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load BLIP captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# 2. Load an image
img_path = "images/golden.jpeg"   # <-- change this
image = Image.open(img_path).convert("RGB")

# 3. Generate caption
inputs = processor(image, return_tensors="pt").to(device)
out = model.generate(**inputs, max_length=30, num_beams=3)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated caption:", caption)
