from thesis.models.core.pick_module import PickModule
from thesis.models.streams.one_stream_attention_lang_fusion import AttentionLangFusion
from thesis.utils.utils import tt
import numpy as np
import os


class ClipLingUNetDetector(PickModule):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    def _build_model(self):
        self.attention = AttentionLangFusion(
            stream_fcn=self.cfg.streams.name,
            in_shape=self.in_shape,
            cfg=self.cfg,
            device=self.device_type,
        )

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']
        out = self.attention(inp_img, lang_goal, softmax=softmax)
        return out  # B, H, W

    def attn_step(self, frame, label, compute_err=False):
        inp_img = frame['img']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(compute_err, inp, out, label)

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
        img = tt(img, self.device) / 255  # 0 - 1
        img = img.permute((0, 3, 1, 2))
        lang_goal = goal if goal is not None else obs["lang_goal"]
        # Attention model forward pass.
        pick_inp = {'inp_img': img,
                    'lang_goal': [lang_goal]}
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

        return {"softmax": pick_logits_disp_masked,
                "pixel": (p0_pix[1], p0_pix[0]),
                "error": err}