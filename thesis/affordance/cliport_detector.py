from thesis.models.core.pick_module import PickModule
from thesis.models.streams.one_stream_attention_lang_fusion import OneStreamAttentionLangFusion
from thesis.utils.utils import tt
import numpy as np
import os


class ClipLingUNetDetector(PickModule):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    def _build_model(self):
        stream_fcn = 'clip_lingunet'
        self.attention = OneStreamAttentionLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=None, ## Set on clip_lingunet_lat to match clip preprocessing
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
        # p0, p0_theta = label['p0'], label['p0_theta']
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(compute_err, inp, out, label)

    def predict(self, obs, goal=None):
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
        _, err = self.attn_step(pick_inp, obs["label"], compute_err=True)

        pick_conf = pick_conf.detach().cpu().numpy().squeeze()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]

        return {"softmax": (pick_conf * 255).astype('uint8'),
                "pixel": (p0_pix[1], p0_pix[0]),
                "error": err}