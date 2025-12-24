from PIL import Image
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True).eval().to(device)

image_size = (1024, 1024)
transform_image = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_path = sys.argv[1] if len(sys.argv) > 1 else 'input.png'
output_path = sys.argv[2] if len(sys.argv) > 2 else 'output.png'

image = Image.open(input_path).convert('RGB')
input_tensor = transform_image(image).unsqueeze(0).to(device)

with torch.no_grad():
    preds = model(input_tensor)[-1].sigmoid().cpu()

mask = transforms.functional.resize(preds[0], image.size[::-1])[0]
mask = (mask * 255).byte()
mask_pil = Image.fromarray(mask.numpy(), mode='L')

image.putalpha(mask_pil)
image.save(output_path)
print(f'Saved: {output_path}')
