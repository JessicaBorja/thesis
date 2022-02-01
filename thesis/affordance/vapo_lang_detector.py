from cv2 import imshow
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

from thesis.models.core.affordance_mask_module import AffordanceMaskModule
from thesis.models.streams.one_stream_attention_lang_fusion import AttentionLangFusion
from thesis.utils.utils import add_img_text, tt, blend_imgs
from thesis.utils.utils import get_transforms


class AffLangDetector(AffordanceMaskModule):

    def __init__(self, cfg, transforms=None, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if transforms is not None:
            self.pred_transforms = get_transforms(transforms, self.in_shape[0])['transforms']
        else:
            self.pred_transforms = nn.Identity()

    def _build_model(self):
        self.attention = AttentionLangFusion(
            stream_fcn=self.cfg.streams.name,
            in_shape=self.in_shape,
            cfg=self.cfg,
            device=self.device_type,
            output_dim=self.n_classes
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

        return {"pixel": (p0_pix[1], p0_pix[0]),
                "error": err}

    def viz_preds(self, inp):
        '''
            Arguments:
                inp(dict):
                    img(float.Tensor): between 0-1, shape= H, W, C
                    lang(list): language instruction
        '''
        return