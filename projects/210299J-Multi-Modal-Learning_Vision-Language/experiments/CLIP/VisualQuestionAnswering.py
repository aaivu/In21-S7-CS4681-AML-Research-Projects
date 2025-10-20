import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor

# Pick device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load BLIP VQA model
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

# 2. Load image + question
image = Image.open("images/golden.jpeg").convert("RGB")
question = "what kind of animal is this?"

# 3. Preprocess inputs
inputs = processor(image, question, return_tensors="pt").to(device)

# 4. Generate answer
out = model.generate(**inputs, max_length=20, num_beams=3)
answer = processor.decode(out[0], skip_special_tokens=True)

print("Q:", question)
print("A:", answer)
