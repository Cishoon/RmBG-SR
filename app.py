import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from spandrel import ModelLoader
from aura_sr import AuraSR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
rmbg_model = AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-2.0', trust_remote_code=True).eval().to(device)
animesharp_model = ModelLoader().load_from_file("weights/4x-AnimeSharp.pth").eval().to(device)
aura_model = AuraSR.from_pretrained("fal/AuraSR-v2")

transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def upscale_animesharp(img):
    tensor = transforms.ToTensor()(img.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        out = animesharp_model(tensor)
    return transforms.ToPILImage()(out.squeeze(0).clamp(0, 1).cpu())

def upscale_aura(img):
    return aura_model.upscale_4x_overlapped(img.convert('RGB'))

def do_rmbg(img, threshold, feather, invert, bg_color):
    rgb = img.convert('RGB')
    input_tensor = transform_image(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = rmbg_model(input_tensor)[-1].sigmoid().cpu()
    mask = transforms.functional.resize(preds[0], rgb.size[::-1])[0].numpy()
    if threshold < 1.0:
        mask = np.where(mask > threshold, 1.0, 0.0)
    if feather > 0:
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=feather))
        mask = np.array(mask_pil) / 255.0
    if invert:
        mask = 1.0 - mask
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
    if bg_color == "透明":
        rgb.putalpha(mask_pil)
        return rgb
    colors = {"白色": (255,255,255), "黑色": (0,0,0), "红色": (255,0,0), "绿色": (0,255,0), "蓝色": (0,0,255)}
    bg = Image.new('RGB', rgb.size, colors[bg_color])
    bg.paste(rgb, mask=mask_pil)
    return bg

def do_sr(img, method):
    if method == "无":
        return img
    upscale_fn = upscale_animesharp if method == "AnimeSharp 4x" else upscale_aura
    if img.mode == 'RGBA':
        rgb, alpha = img.convert('RGB'), img.split()[3]
        rgb = upscale_fn(rgb)
        result = rgb.copy()
        result.putalpha(alpha.resize(rgb.size, Image.LANCZOS))
        return result
    return upscale_fn(img)

def process(image, threshold, feather, invert, bg_color, sr_method, order, enable_rmbg):
    if order == "先去背景":
        result = do_rmbg(image, threshold, feather, invert, bg_color) if enable_rmbg else image
        result = do_sr(result, sr_method)
    else:
        result = do_sr(image, sr_method)
        result = do_rmbg(result, threshold, feather, invert, bg_color) if enable_rmbg else result
    return result

gr.Interface(
    fn=process,
    inputs=[
        gr.Image(type="pil", label="上传图片"),
        gr.Slider(0, 1, value=1.0, step=0.05, label="阈值 (1.0=平滑, <1=硬边缘)"),
        gr.Slider(0, 20, value=0, step=1, label="羽化半径"),
        gr.Checkbox(label="反转蒙版 (保留背景)"),
        gr.Radio(["透明", "白色", "黑色", "红色", "绿色", "蓝色"], value="透明", label="背景颜色"),
        gr.Radio(["无", "AnimeSharp 4x", "AuraSR-v2 4x"], value="无", label="超分方法"),
        gr.Radio(["先去背景", "先超分"], value="先去背景", label="处理顺序"),
        gr.Checkbox(value=True, label="启用去背景"),
    ],
    outputs=gr.Image(type="pil", label="结果", format="png"),
    title="RMBG-2.0 去背景 + 超分",
).launch(server_name="0.0.0.0", server_port=7860)
