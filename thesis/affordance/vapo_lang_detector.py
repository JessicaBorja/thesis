import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import torch

from thesis.models.core.affordance_mask_module import AffordanceMaskModule
from thesis.models.streams.one_stream_attention_lang_fusion_mask import AttentionLangFusionMask
from thesis.utils.utils import add_img_text, tt, blend_imgs,get_transforms, resize_pixel, torch_to_numpy, overlay_flow
import affordance.utils.flowlib as flowlib


class VAPOAffLangDetector(AffordanceMaskModule):

    def __init__(self, cfg, transforms=None, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if transforms is not None:
            self.pred_transforms = get_transforms(transforms, self.in_shape[0])['transforms']
        else:
            self.pred_transforms = nn.Identity()

    def _build_model(self):
        self.attention = AttentionLangFusionMask(
            stream_fcn=self.cfg.streams.name,
            in_shape=self.in_shape,
            cfg=self.cfg,
            device=self.device_type,
            output_dim=self.n_classes
        )
        feature_dim = self.attention.attn_stream.decoder_channels[-1]
        self.center_direction_net = nn.Conv2d(feature_dim, 2, kernel_size=1, stride=1, padding=0, bias=False)

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
        preds = self.forward(pick_inp)
        err = None
        if info is not None:
            _, err = self.attn_step(pick_inp, info, compute_err=True)

        aff_mask = preds["affordance"].argmax(1)

        # Hough Voting
        # directions = preds["center_dirs"]
        # p0_pix, _, _ = self.pred_centers(aff_mask, directions)
        # p0_pix = [torch_to_numpy(o) for o in p0_pix[0]]
        # if(len(p0_pix) > 0):
        #     p0_pix = p0_pix[0]
        # else:
        #     p0_pix = np.array([0, 1])

        # Argmax
        aff_probs = preds["affordance"][:, 1] * aff_mask
        idx = aff_probs.reshape(1, -1).argmax(-1)
        idx = idx.detach().cpu().numpy()
        p0_pix = self.unravel_idx(idx, shape=aff_probs.shape[1:])[0][::-1]

        # softmax = preds["affordance"][:, 1].detach().cpu().numpy().squeeze()
        softmax = aff_probs.detach().cpu().numpy().squeeze()
        # (column, row) -> (u, v)
        center_dirs = preds["center_dirs"].detach().cpu().numpy()

        return {"softmax": softmax,
                "center_dirs": center_dirs,
                "pixel": p0_pix,
                "error": err}

    def viz_preds(self, inp, pred, out_shape=(300, 300), waitkey=0):
        '''
            Arguments:
                inp(dict):
                    img(np.ndarray): between 0-1, shape= H, W, C
                    lang_goal(list): language instruction
                pred(dict): output of self.predict(inp)
        '''
        frame = inp["img"].copy()
        frame = cv2.resize(frame, pred["softmax"].shape)
        pred_img = frame.copy()

        cm = plt.get_cmap('viridis')
        heatmap = cm(pred["softmax"])[:, :, [0,1,2]] * 255.0
        heatmap = blend_imgs(frame.copy(), heatmap, alpha=0.7)

        mask = np.zeros_like(pred["softmax"], dtype='uint8')
        mask[pred["softmax"]>0] = 1

        # Get centers and directions
        fg_mask = torch.tensor(mask).unsqueeze(0).cuda()
        center_dirs = torch.tensor(pred["center_dirs"]).cuda()
        centers, directions, _ = self.pred_centers(fg_mask, center_dirs)
        centers = [torch_to_numpy(o) for o in centers[0]]
        directions = torch_to_numpy(directions[0].permute(1, 2, 0))  # H x W x 2
        flow_img = flowlib.flow_to_image(directions)
        mask = mask * 255

        # Resize images
        mask = cv2.resize(mask, out_shape)
        frame = cv2.resize(frame, out_shape)
        if('img_label' in inp):
            label = cv2.resize(inp["img_label"].copy(), out_shape)
        else: 
            label = frame
        pred_img = cv2.resize(pred_img, out_shape, interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.resize(heatmap, out_shape, interpolation=cv2.INTER_CUBIC)     
        flow_img = cv2.resize(flow_img, out_shape)
        pred_img = overlay_flow(flow_img, pred_img, mask)

        centers_vals = [pred['softmax'][c[0], c[1]] for c in centers]
        max_val_idx = np.argmax(centers_vals)
        # Draw centers
        for i, c in enumerate(centers):
            c = resize_pixel(c, self.in_shape[:2], out_shape)
            u, v = int(c[1]), int(c[0])  # center stored in matrix convention
            if i == max_val_idx:
                color = (0, 0, 0)
            else:
                color = (255, 255, 255)
            pred_img = cv2.drawMarker(
                    pred_img,
                    (u, v),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=2,
                    line_type=cv2.LINE_AA,
                )


        pred_img = pred_img.astype(float) / 255
        label = label.astype(float) / 255
        flow_img = flow_img.astype(float) / 255
        out_img = np.concatenate([label, pred_img, heatmap, flow_img], axis=1)


        # Prints the text.
        out_img = add_img_text(out_img, inp["lang_goal"])
        cv2.imshow("img", out_img[:, :, ::-1])
        cv2.waitKey(waitkey)