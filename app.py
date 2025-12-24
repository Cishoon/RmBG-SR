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

def generate_mask(image, threshold, feather, invert):
    if image is None:
        return None
    rgb = image.convert('RGB')
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
    # 返回带背景的编辑器格式
    return {"background": rgb, "layers": [], "composite": mask_pil.convert("RGBA")}

def apply_mask(image, mask_editor, bg_color, sr_method, order, enable_rmbg):
    if image is None:
        return None
    rgb = image.convert('RGB')
    
    if enable_rmbg and mask_editor is not None:
        if isinstance(mask_editor, dict) and "composite" in mask_editor:
            composite = mask_editor["composite"]
            if composite is not None:
                mask_pil = composite.convert("L")
            else:
                mask_pil = Image.new("L", rgb.size, 255)
        else:
            mask_pil = mask_editor.convert("L") if mask_editor else Image.new("L", rgb.size, 255)
        
        if bg_color == "透明":
            result = rgb.copy()
            result.putalpha(mask_pil)
        else:
            colors = {"白色": (255,255,255), "黑色": (0,0,0), "红色": (255,0,0), "绿色": (0,255,0), "蓝色": (0,0,255)}
            bg = Image.new('RGB', rgb.size, colors[bg_color])
            bg.paste(rgb, mask=mask_pil)
            result = bg
    else:
        result = rgb
    
    # 超分
    if sr_method != "无":
        upscale_fn = upscale_animesharp if sr_method == "AnimeSharp 4x" else upscale_aura
        if result.mode == 'RGBA':
            rgb_part, alpha = result.convert('RGB'), result.split()[3]
            rgb_part = upscale_fn(rgb_part)
            result = rgb_part.copy()
            result.putalpha(alpha.resize(rgb_part.size, Image.LANCZOS))
        else:
            result = upscale_fn(result)
    
    return result

with gr.Blocks(title="RMBG-2.0 去背景 + 超分") as demo:
    gr.Markdown("# RMBG-2.0 去背景 + 超分")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传图片")
            with gr.Row():
                threshold = gr.Slider(0, 1, value=1.0, step=0.05, label="阈值")
                feather = gr.Slider(0, 20, value=0, step=1, label="羽化")
            invert = gr.Checkbox(label="反转蒙版")
            gen_mask_btn = gr.Button("生成蒙版", variant="primary")
        
        with gr.Column():
            mask_editor = gr.ImageEditor(type="pil", label="编辑蒙版 (白色=保留, 黑色=删除)", 
                                         brush=gr.Brush(colors=["#FFFFFF", "#000000"], default_size=20))
    
    with gr.Row():
        bg_color = gr.Radio(["透明", "白色", "黑色", "红色", "绿色", "蓝色"], value="透明", label="背景颜色")
        sr_method = gr.Radio(["无", "AnimeSharp 4x", "AuraSR-v2 4x"], value="无", label="超分方法")
    
    with gr.Row():
        enable_rmbg = gr.Checkbox(value=True, label="启用去背景")
        order = gr.Radio(["先去背景", "先超分"], value="先去背景", label="处理顺序")
    
    apply_btn = gr.Button("应用并导出", variant="primary")
    output = gr.Image(type="pil", label="结果", format="png")
    
    gen_mask_btn.click(generate_mask, [image_input, threshold, feather, invert], mask_editor)
    apply_btn.click(apply_mask, [image_input, mask_editor, bg_color, sr_method, order, enable_rmbg], output)

demo.launch(server_name="0.0.0.0", server_port=7860)
