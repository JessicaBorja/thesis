import torch
import clip
from PIL import Image
from thesis.grad_cam.grad_cam import gradCAM
from thesis.grad_cam.utils import *
from thesis.models.base_detector import BaseDetector
import cv2


class CLIPPointDetector(BaseDetector):
    def __init__(self,
                 resize: int,
                 clip_model: str,
                 saliency_layer: str,
                 blur: bool,
                 viz: bool):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model = clip_model
        self.saliency_layer = saliency_layer
        self.model, self.preprocess = clip.load(self.clip_model,
                                                device=self.device,
                                                jit=False)
        self.blur = blur
        self.resize = resize
        self.viz = viz
    
    def find_target(self, inputs: dict):
        '''
            inputs:
                rgb_obs: np.ndarray H, W, C
                caption: str
        '''
        img = Image.fromarray(inputs["rgb_obs"])

        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        image_caption = inputs["caption"]
        text_input = clip.tokenize([image_caption]).to(self.device)
        attn_map = gradCAM(
            self.model.visual,
            image_input,
            self.model.encode_text(text_input).float(),
            getattr(self.model.visual, self.saliency_layer)
        )
        attn_map = attn_map.squeeze().detach().cpu().numpy()
        
        # Original size
        image_np = np.asarray(img).astype(np.float32) / 255.
        attn_map = cv2.resize(attn_map, image_np.shape[:2])
        # Transforms size
        attn_map, pixel_max = self.getAttMap(image_np, attn_map)
        v, u  = pixel_max
        if(self.viz):
            frame = cv2.drawMarker(np.array(attn_map),
                        (u, v),
                        (0, 0, 0),
                        markerType=cv2.MARKER_CROSS,
                        markerSize=12,
                        thickness=2,
                        line_type=cv2.LINE_AA)
            cv2.imshow("Target", frame[:, :, ::-1])
            cv2.waitKey(1)
            # plt.plot(u, v,'x', color='black', markersize=12)
            # plt.imshow(attn_map)
            # plt.axis("off")
            # plt.show()
        return (u, v)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        # Normalize to [0, 1].
        x = x - x.min()
        if x.max() > 0:
            x = x / x.max()
        return x

    # Modified from: https://github.com/salesforce/ALBEF/blob/main/visualization.ipynb
    def getAttMap(self, img, attn_map):
        if self.blur:
            attn_map = filters.gaussian_filter(attn_map, 0.02*max(img.shape[:2]))
        attn_map = normalize(attn_map)
        pixel_max = np.unravel_index(attn_map.argmax(), attn_map.shape)[:2]
        
        cmap = plt.get_cmap('jet')
        attn_map_c = np.delete(cmap(attn_map), 3, 2)
        attn_map = 1*(1-attn_map**0.7).reshape(attn_map.shape + (1,))*img + \
                (attn_map**0.7).reshape(attn_map.shape+(1,)) * attn_map_c
        return attn_map, pixel_max

    def viz_attn(self, img, attn_map, pixel_max):
        img = cv2.resize(img, attn_map.shape[:2])
        attn_map, pixel_max = self.getAttMap(img, attn_map)
        y, x = pixel_max

        plt.plot(x, y,'x', color='black', markersize=12)
        plt.imshow(attn_map)
        plt.axis("off")
        plt.show()
        
    def load_image(self, img_path, resize=None):
        image = Image.open(img_path).convert("RGB")
        if resize is not None:
            image = image.resize((resize, resize))
        return np.asarray(image).astype(np.float32) / 255.