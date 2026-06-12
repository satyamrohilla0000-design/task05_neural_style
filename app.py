"""
Task-05: Neural Style Transfer
Applies the artistic style of one image to the content of another
using VGG19 feature extraction and gradient descent optimization.
"""

import os
import copy
import time
from io import BytesIO

import gradio as gr
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models

# ── Device setup ───────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[NST] Using device: {DEVICE}")

# ── Image utilities ────────────────────────────────────────────────────────
def load_image(img: Image.Image, size: int = 512) -> torch.Tensor:
    """Convert PIL image to normalised tensor."""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(img.convert("RGB")).unsqueeze(0).to(DEVICE)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert output tensor back to PIL image."""
    img = tensor.cpu().clone().detach().squeeze(0)
    # Denormalise
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img)


# ── Loss modules ───────────────────────────────────────────────────────────
class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.loss = F.mse_loss(x, self.target)
        return x


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.size()
    features = x.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class StyleLoss(nn.Module):
    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x


class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE).view(-1, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


# ── Build NST model ────────────────────────────────────────────────────────
CONTENT_LAYERS = ["conv_4"]
STYLE_LAYERS   = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]


def build_model_and_losses(
    cnn: nn.Module,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
):
    cnn = copy.deepcopy(cnn)
    norm = Normalization().to(DEVICE)

    content_losses, style_losses = [], []
    model = nn.Sequential(norm)

    conv_idx = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            name = f"conv_{conv_idx}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{conv_idx}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{conv_idx}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_idx}"
        else:
            name = f"unknown_{conv_idx}"

        model.add_module(name, layer)

        if name in CONTENT_LAYERS:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{conv_idx}", cl)
            content_losses.append(cl)

        if name in STYLE_LAYERS:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f"style_loss_{conv_idx}", sl)
            style_losses.append(sl)

        # Stop after last needed layer
        if conv_idx >= 5:
            break

    # Trim trailing layers after last loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i + 1]

    return model, content_losses, style_losses


# ── NST optimisation loop ──────────────────────────────────────────────────
def run_style_transfer(
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    input_img: torch.Tensor,
    num_steps: int = 300,
    style_weight: float = 1e6,
    content_weight: float = 1.0,
    progress=None,
) -> torch.Tensor:

    cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(DEVICE).eval()
    model, content_losses, style_losses = build_model_and_losses(cnn, content_img, style_img)

    # Freeze model; optimise input image pixels
    for param in model.parameters():
        param.requires_grad_(False)
    input_img.requires_grad_(True)

    optimizer = optim.LBFGS([input_img])
    step = [0]

    while step[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()
            step[0] += 1
            if step[0] % 50 == 0 and progress:
                progress(step[0] / num_steps, desc=f"Step {step[0]}/{num_steps}")
            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


def style_transfer_pipeline(
    content_pil: Image.Image,
    style_pil: Image.Image,
    image_size: int,
    steps: int,
    style_weight: float,
    content_weight: float,
    init_from: str,
    progress=gr.Progress(),
) -> tuple:
    if content_pil is None or style_pil is None:
        return None, "⚠️ Please upload both a content and a style image."

    try:
        progress(0, desc="Loading images…")
        content_tensor = load_image(content_pil, size=image_size)
        style_tensor   = load_image(style_pil,   size=image_size)

        if init_from == "Content image":
            input_tensor = content_tensor.clone()
        elif init_from == "Style image":
            input_tensor = style_tensor.clone()
        else:  # random noise
            input_tensor = torch.randn_like(content_tensor).clamp(0, 1)

        progress(0.05, desc="Building model…")
        result_tensor = run_style_transfer(
            content_tensor,
            style_tensor,
            input_tensor,
            num_steps=steps,
            style_weight=style_weight,
            content_weight=content_weight,
            progress=progress,
        )

        result_pil = tensor_to_pil(result_tensor)
        progress(1.0, desc="Done!")
        return result_pil, f"✅ Style transfer complete! ({steps} optimisation steps)"

    except Exception as e:
        import traceback
        return None, f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"


# ── Gradio UI ──────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(
        title="Task-05 · Neural Style Transfer",
        theme=gr.themes.Base(
            primary_hue="orange",
            secondary_hue="amber",
            font=gr.themes.GoogleFont("Playfair Display"),
        ),
        css="""
        .gradio-container { max-width: 1100px; margin: auto; }
        #title { text-align: center; padding: 20px 0 10px; }
        """,
    ) as demo:

        gr.Markdown(
            """
# 🎭 Task-05 — Neural Style Transfer
**Paint any photo in the style of famous artworks using VGG19 feature extraction.**
            """,
            elem_id="title",
        )

        with gr.Tabs():
            with gr.TabItem("🖼️ Stylize"):
                with gr.Row():
                    content_img = gr.Image(label="📷 Content Image", type="pil", height=300)
                    style_img   = gr.Image(label="🎨 Style Image",   type="pil", height=300)
                    result_img  = gr.Image(label="✨ Stylized Result", type="pil", height=300)

                with gr.Row():
                    with gr.Column():
                        size_dd = gr.Dropdown(
                            choices=[256, 384, 512],
                            value=256,
                            label="Image Size (larger = slower, higher quality)",
                        )
                        steps_sl = gr.Slider(50, 500, value=200, step=50, label="Optimisation Steps")
                        init_dd = gr.Dropdown(
                            choices=["Content image", "Style image", "Random noise"],
                            value="Content image",
                            label="Initialise from",
                        )
                    with gr.Column():
                        style_w = gr.Slider(1e3, 1e7, value=1e6, step=1e5, label="Style Weight")
                        content_w = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Content Weight")
                        run_btn = gr.Button("🚀 Run Style Transfer", variant="primary", size="lg")

                status_out = gr.Textbox(label="Status", interactive=False)

                run_btn.click(
                    fn=style_transfer_pipeline,
                    inputs=[content_img, style_img, size_dd, steps_sl, style_w, content_w, init_dd],
                    outputs=[result_img, status_out],
                )

            with gr.TabItem("💡 Tips"):
                gr.Markdown(
                    """
## Getting Great Results

### Content Images (best choices)
- Clear subject matter (portrait, landscape, building)
- Good lighting and contrast
- Avoid very busy compositions

### Style Images (best choices)
- Famous paintings: Van Gogh, Picasso, Monet, Kandinsky
- Textured artwork with strong brushstroke patterns
- High contrast, distinct color palette

### Parameter Guide
| Parameter | Effect |
|-----------|--------|
| **Style Weight ↑** | More artistic style, less content detail |
| **Style Weight ↓** | More content preserved, subtle style |
| **Content Weight ↑** | Preserves original content more |
| **Steps ↑** | More refined result, takes longer |
| **Image Size** | 256px = fast preview; 512px = final quality |

### Recommended Combos
- **Subtle style**: Style Weight 1e5, Content Weight 5
- **Balanced**: Style Weight 1e6, Content Weight 1 (default)
- **Heavy stylization**: Style Weight 1e7, Content Weight 0.5
                    """
                )

            with gr.TabItem("📖 About"):
                gr.Markdown(
                    """
## How Neural Style Transfer Works

Based on the seminal paper *"A Neural Algorithm of Artistic Style"* (Gatys et al., 2015).

### Architecture: VGG19
VGG19 is a deep CNN trained on ImageNet. Its intermediate layers capture:
- **Early layers** (conv_1, conv_2): textures, colours, brushstroke patterns → **Style**
- **Deep layers** (conv_4): high-level content, objects, structures → **Content**

### Loss Functions

**Content Loss**:
```
L_content = MSE(F_content, F_generated)
```
Minimises difference in deep feature maps.

**Style Loss** (Gram matrices):
```
G = F · Fᵀ  (feature correlation matrix)
L_style = Σ MSE(G_style, G_generated)
```
Gram matrices capture texture/style independent of spatial layout.

**Total Loss**:
```
L_total = α·L_content + β·L_style
```

### Optimisation
- Input: a **noisy or content image** (not model weights!)
- Optimiser: **L-BFGS** (second-order, fast convergence)
- Each step updates pixel values to minimise total loss

### Stack
PyTorch · VGG19 · torchvision · Gradio
                    """
                )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7863)
