import cv2
import matplotlib.pyplot as plt
import numpy as np

from thesis.models.core.pick_module import AffordanceModule
from thesis.models.streams.one_stream_attention_lang_fusion import AttentionLangFusion
from thesis.utils.utils import tt, blend_imgs


class ClipLingUNetDetector(AffordanceModule):

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    def _build_model(self):
        self.attention = AttentionLangFusion(
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
        img = tt(img, self.device) / 255  # 0 - 1
        img = img.permute((0, 3, 1, 2))

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

        return {"softmax": pick_logits_disp_masked,
                "pixel": (p0_pix[1], p0_pix[0]),
                "error": err}

    def viz_preds(self, inp):
        frame = inp["img"][0].detach().cpu().numpy()
        frame = (frame * 255).astype("uint8")
        frame = np.transpose(frame, (1, 2, 0))
        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        obs = {"img": frame.copy(),
               "lang_goal": inp["lang_goal"][0]}
        info = None  # labels
        pred = self.predict(obs, info=info)
        pred_img = frame.copy()

        cm = plt.get_cmap('viridis')
        heatmap = cm(pred["softmax"])[:, :, [0,1,2]] * 255
        heatmap = blend_imgs(frame.copy(), heatmap, alpha=0.7)

        pixel = pred["pixel"]
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

        new_size = (400, 400)
        pred_img = cv2.resize(pred_img, new_size, interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.resize(heatmap, new_size, interpolation=cv2.INTER_CUBIC)
        pred_img = pred_img.astype(float) / 255
        out_img = np.concatenate([pred_img, heatmap], axis=1)


        # Prints the text.
        font_scale = 0.6
        thickness = 2
        color = (0, 0, 0)
        x1, y1 = 10, 20
        text_label = obs["lang_goal"]
        (w, h), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        out_img = cv2.rectangle(out_img, (x1, y1 - 20), (x1 + w, y1 + h), color, -1)
        out_img = cv2.putText(
            out_img,
            text_label,
            org=(x1, y1),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=thickness,
        )
        cv2.imshow("img", out_img[:, :, ::-1])
        cv2.waitKey(0)