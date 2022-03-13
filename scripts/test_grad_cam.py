import torch
import clip
from PIL import Image
from thesis.affordance.grad_cam import gradCAM
from thesis.utils.utils import *


if __name__ == "__main__":
    clip_model = "RN50" #@param ["RN50", "RN101", "RN50x4", "RN50x16"]
    saliency_layer = "layer4" #@param ["layer4", "layer3", "layer2", "layer1"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(clip_model, device=device, jit=False)

    image_path = './images/test_tabletop_real.png'
    blur = True
    img = Image.open(image_path)
    # img = Image.fromarray(cv2.imread(image_path))
    image_input = preprocess(img).unsqueeze(0).to(device)
    image_np = load_image(image_path, model.visual.input_resolution)

    image_caption = input("Type an instruction \n")
    while image_caption != "stop":
        text_input = clip.tokenize([image_caption]).to(device)
        attn_map = gradCAM(
            model.visual,
            image_input,
            model.encode_text(text_input).float(),
            getattr(model.visual, saliency_layer)
        )
        attn_map = attn_map.squeeze().detach().cpu().numpy()

        viz_attn(image_np, attn_map, blur)
        image_caption = input("Type an instruction \n")