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

def flood_fill(mask, x, y, fill_value):
    arr = np.array(mask)
    h, w = arr.shape
    x, y = int(x), int(y)
    if x < 0 or x >= w or y < 0 or y >= h:
        return mask
    target = arr[y, x]
    if target == fill_value:
        return mask
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
        if arr[cy, cx] != target:
            continue
        arr[cy, cx] = fill_value
        stack.extend([(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)])
    return Image.fromarray(arr)

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
    return Image.fromarray((mask * 255).astype(np.uint8), mode='L')

def do_rmbg(img, mask, bg_color):
    rgb = img.convert('RGB')
    mask_pil = mask if isinstance(mask, Image.Image) else Image.fromarray(mask)
    if bg_color == "透明":
        result = rgb.copy()
        result.putalpha(mask_pil)
        return result
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

def apply_result(image, mask, bg_color, sr_method, enable_rmbg, order):
    if image is None:
        return None
    if order == "先去背景":
        result = do_rmbg(image, mask, bg_color) if enable_rmbg and mask else image
        result = do_sr(result, sr_method)
    else:
        result = do_sr(image, sr_method)
        if enable_rmbg and mask:
            if sr_method != "无":
                mask_resized = mask.resize(result.size, Image.LANCZOS) if mask else None
                result = do_rmbg(result, mask_resized, bg_color)
            else:
                result = do_rmbg(result, mask, bg_color)
    return result

with gr.Blocks(title="RMBG-2.0 去背景 + 超分") as demo:
    gr.Markdown("# RMBG-2.0 去背景 + 超分")
    
    # 上传图片和结果并排
    with gr.Row():
        image_input = gr.Image(type="pil", label="上传图片")
        output = gr.Image(type="pil", label="结果", format="png")
    
    # 基本设置
    with gr.Row():
        with gr.Column():
            with gr.Row():
                threshold = gr.Slider(0, 1, value=1.0, step=0.05, label="阈值")
                feather = gr.Slider(0, 20, value=0, step=1, label="羽化")
            with gr.Row():
                invert = gr.Checkbox(label="反转蒙版")
                enable_rmbg = gr.Checkbox(value=True, label="启用去背景")
        with gr.Column():
            bg_color = gr.Radio(["透明", "白色", "黑色", "红色", "绿色", "蓝色"], value="透明", label="背景颜色")
            sr_method = gr.Radio(["无", "AnimeSharp 4x", "AuraSR-v2 4x"], value="无", label="超分方法")
            order = gr.Radio(["先去背景", "先超分"], value="先去背景", label="处理顺序")
    
    with gr.Row():
        gen_mask_btn = gr.Button("生成蒙版", variant="secondary")
        apply_btn = gr.Button("应用并导出", variant="primary")
    
    # 高级选项 - 编辑蒙版
    with gr.Accordion("高级选项 - 编辑蒙版", open=False):
        with gr.Row():
            with gr.Column():
                fill_color = gr.Radio(["白色(保留)", "黑色(删除)"], value="白色(保留)", label="点击填充颜色")
                mask_click = gr.Image(type="pil", label="点击填充 (点击区域自动填充)", image_mode="L", interactive=False)
            with gr.Column():
                mask_editor = gr.ImageEditor(type="pil", label="画笔编辑 (白=保留, 黑=删除)",
                                             brush=gr.Brush(colors=["#FFFFFF", "#000000"], default_size=20))
    
    # 生成蒙版 -> 同时更新两个窗口
    def on_gen_mask(image, th, fe, inv):
        mask = generate_mask(image, th, fe, inv)
        if mask is None or image is None:
            return None, None
        editor_val = {"background": image.convert("RGB"), "layers": [], "composite": mask.convert("RGBA")}
        return mask, editor_val
    
    # 点击填充 -> 同时更新两个窗口
    def on_click_fill(image, mask, evt: gr.SelectData, fill_color):
        if mask is None:
            return None, None
        x, y = evt.index
        fill_val = 255 if fill_color == "白色(保留)" else 0
        arr = np.array(mask)
        arr = np.where(arr > 127, 255, 0).astype(np.uint8)
        mask_bin = Image.fromarray(arr)
        new_mask = flood_fill(mask_bin, x, y, fill_val)
        editor_val = {"background": image.convert("RGB"), "layers": [], "composite": new_mask.convert("RGBA")} if image else None
        return new_mask, editor_val
    
    # 画笔编辑 -> 同步到点击窗口
    def on_editor_change(editor_data):
        if editor_data is None:
            return None
        composite = editor_data.get("composite")
        if composite is None:
            return None
        return composite.convert("L")
    
    gen_mask_btn.click(on_gen_mask, [image_input, threshold, feather, invert], [mask_click, mask_editor])
    mask_click.select(on_click_fill, [image_input, mask_click, fill_color], [mask_click, mask_editor])
    mask_editor.change(on_editor_change, [mask_editor], mask_click)
    apply_btn.click(apply_result, [image_input, mask_click, bg_color, sr_method, enable_rmbg, order], output)

demo.launch(server_name="0.0.0.0", server_port=7860)
