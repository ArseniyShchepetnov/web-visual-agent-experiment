"""Small CLIP image feature smoke test."""

import requests
from PIL import Image
from transformers import AutoProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True, timeout=30).raw)

inputs = processor(images=image, return_tensors="pt")
image_features = model.get_image_features(**inputs)
