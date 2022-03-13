import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from thesis.models.core.affordance_pixel_module import AffordancePixelModule
from thesis.models.streams.one_stream_attention_lang_fusion_pixel import AttentionLangFusionPixel
from thesis.utils.utils import add_img_text, tt, blend_imgs,get_transforms, resize_pixel

class AffLangDetector(AffordancePixelModule):

    def __init__(self, cfg, transforms=None, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if transforms is not None:
            self.pred_transforms = get_transforms(transforms, self.in_shape[0])['transforms']
        else:
            self.pred_transforms = nn.Identity()

    def _build_model(self):
        self.attention = AttentionLangFusionPixel(
            stream_fcn=self.cfg.streams.name,
            in_shape=self.in_shape,
            cfg=self.cfg,
            device=self.device_type,
        )

    def predict(self, obs, goal=None, info=None):
        """ Run inference and return best pixel given visual observations.
            Args:
                obs(dict):
                    img: np.ndarray (H, W, C)  values between 0-255 uint8
                    lang_goal: str
                goal(str)
            Returns:
                (Tuple) nd.array: pixel position

        """
        # Get inputs
        img = np.expand_dims(obs["img"], 0)  # B, H, W, C
        img = tt(img, self.device)
        img = img.permute((0, 3, 1, 2))

        img = self.pred_transforms(img)

        lang_goal = goal if goal is not None else obs["lang_goal"]
        # Attention model forward pass.
        pick_inp = {'inp_img': img,
                    'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)
        pick_inp["img"] = img

        err = None
        if info is not None:
            _, err = self.attn_step(pick_inp, info, compute_err=True)

        logits = pick_conf.detach().cpu().numpy().squeeze()
        argmax = np.argmax(logits)
        argmax = np.unravel_index(argmax, shape=logits.shape)
        p0_pix = argmax[:2]
        
        affordance_heatmap_scale = 30
        pick_logits_disp = (logits * 255 * affordance_heatmap_scale).astype('uint8')
        pick_logits_disp_masked = np.ma.masked_where(pick_logits_disp < 0, pick_logits_disp)

        return {"softmax": pick_logits_disp,
                "pixel": (p0_pix[1], p0_pix[0]),
                "error": err}

    def viz_preds(self, inp, pred, out_size=(300, 300), waitkey=0):
        '''
            Arguments:
                inp(dict):
                    img(np.ndarray): between 0-1, shape= H, W, C
                    lang_goal(list): language instruction
                pred(dict): output of self.predict(inp)
        '''
        # frame = inp["img"][0].detach().cpu().numpy()
        # frame = (frame * 255).astype("uint8")
        # frame = np.transpose(frame, (1, 2, 0))
        # if frame.shape[-1] == 1:
        #     frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        frame = inp["img"].copy()
        pred_img = frame.copy()

        cm = plt.get_cmap('viridis')
        heatmap = cm(pred["softmax"])[:, :, [0,1,2]] * 255
        heatmap = heatmap.astype('uint8')

        frame = cv2.resize(frame, heatmap.shape[:2])
        heatmap = blend_imgs(frame.copy(), heatmap, alpha=0.7)

        pixel = pred["pixel"]
        pixel = resize_pixel(pixel, self.in_shape[:2], pred_img.shape[:2])
        # print(pred["error"], pred["pixel"], (x, y))
        pred_img = cv2.drawMarker(
                pred_img,
                (pixel[0], pixel[1]),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        pred_img = cv2.resize(pred_img, out_size, interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.resize(heatmap, out_size, interpolation=cv2.INTER_CUBIC)
        pred_img = pred_img.astype(float) / 255
        out_img = np.concatenate([pred_img, heatmap], axis=1)


        # Prints the text.
        out_img = add_img_text(out_img, inp["lang_goal"])
        cv2.imshow("img", out_img[:, :, ::-1])
        cv2.waitKey(waitkey)