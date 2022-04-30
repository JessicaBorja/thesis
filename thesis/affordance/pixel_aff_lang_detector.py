import cv2
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import logging

import thesis.models as models
from thesis.models.lang_fusion.one_stream_attention_lang_fusion_pixel import AttentionLangFusionPixel
from thesis.utils.utils import add_img_text, tt, blend_imgs,get_transforms, resize_pixel
from thesis.utils.losses import cross_entropy_with_logits

class PixelAffLangDetector(LightningModule):
    def __init__(self, cfg, in_shape=(200, 200, 3), transforms=None,
                 depth_dist=None, *args, **kwargs):
        super().__init__()
        self.loss_weights = cfg.loss_weights
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.total_steps = 0
        self.in_shape = in_shape
        self._batch_loss = []
        self.cmd_log = logging.getLogger(__name__)
        self.pred_depth = depth_dist is not None
        self.depth_est_dist = depth_dist

        if transforms is not None:
            self.pred_transforms = get_transforms(transforms, self.in_shape[0])['transforms']
        else:
            self.pred_transforms = nn.Identity()
        self._build_model()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        self.attention.train()

        frame, label = batch

        # Get training losses.
        step = self.total_steps + 1
        total_loss, err, info = self.attn_step(frame, label, compute_err=True)
        bs = frame["img"].shape[0]

        self.log('Training/total_loss', total_loss,
                 on_step=False, on_epoch=True,
                 batch_size=bs)
        for loss_fnc, value in info.items():
            self.log('Training/%s' % loss_fnc, value,
                     on_step=False, on_epoch=True)

        if self.pred_depth:
            self.log('Training/depth_err', err['depth'],
                    on_step=False, on_epoch=True,
                    batch_size=bs)

        self.total_steps = step
        return total_loss

    def validation_step(self, batch, batch_idx):
        self.attention.eval()

        frame, label = batch
        val_total_loss, err, info = self.attn_step(frame, label, compute_err=True)
        bs = frame["img"].shape[0]
        self.log('Validation/total_loss', val_total_loss,
                 on_step=False, on_epoch=True,
                 batch_size=bs)
        for loss_fnc, value in info.items():
            self.log('Validation/%s' % loss_fnc, value,
                     on_step=False, on_epoch=True,)

        # self.log('Validation/dist_err', err['px_dist'])
        # if self.pred_depth:
        #     self.log('Validation/depth_err', err['depth'],
        #             on_step=False, on_epoch=True,
        #             batch_size=bs)
        return dict(
            val_loss=val_total_loss,
            val_attn_dist_err=err['px_dist'],
            val_depth_err=err["depth"],
            n_imgs=batch[1]['p0'].shape[0],
        )

    def validation_epoch_end(self, all_outputs):
        # mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])        
        total_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_imgs = np.sum([v['n_imgs'] for v in all_outputs])
        mean_img_error = total_dist_err/total_imgs

        # self.log('Validation/total_loss', mean_val_total_loss)
        self.log('Validation/total_dist_err', total_dist_err)
        self.log('Validation/mean_dist_error', mean_img_error)

        if self.pred_depth:
            mean_val_depth_err = np.mean([v['val_depth_err'].item() for v in all_outputs])
            self.log('Validation/mean_val_depth_err', mean_val_depth_err)

        print("\nAttn Err - Dist: {:.2f}".format(total_dist_err))

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.attention.parameters(), lr=self.cfg.lr)
        return optim

    def _build_model(self):
        self.attention = AttentionLangFusionPixel(
            stream_fcn=self.cfg.streams.name,
            in_shape=self.in_shape,
            cfg=self.cfg,
            device=self.device_type,
        )
        if self.pred_depth:
            _depth_est = models.deth_est_nets[self.depth_est_dist]
            self.depth_est = _depth_est(self.in_shape, 1, self.cfg, self.device_type)

    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']
        out_aff, _info = self.attention(inp_img, lang_goal, softmax=softmax)
        out = {"aff": out_aff}
        if self.pred_depth:
            dist, _info = self.depth_est(inp_img, _info['text_enc'])
            out.update({"depth_dist": dist})
        return out  # B, H, W

    def attn_criterion(self, compute_err, inp, pred, label, reparametrize=False):
        inp_img = inp['inp_img']
        B = inp_img.shape[0]

        # B, H, W, 1
        label_size = (inp_img.shape[0], ) + inp_img.shape[2:]
        aff_label = torch.zeros(label_size)
        p0=label['p0'].detach().cpu().numpy()  # B, 2
        aff_label[np.arange(B), p0[:, 0], p0[:, 1]] = 1  # B, H, W

        # B, 1, H, W
        # aff_label = aff_label.permute((2, 0, 1))
        aff_label = aff_label.reshape(B, np.prod(aff_label.shape[1:]))  # B, H*W
        aff_label = aff_label.to(dtype=torch.float, device=pred['aff'].device)

        # Get loss.
        aff_loss = cross_entropy_with_logits(pred["aff"], aff_label)
        if self.pred_depth:
            gt_depth = label["depth"].unsqueeze(-1).float()
            depth_loss = self.depth_est.loss(pred['depth_dist'], gt_depth)
        else: 
            depth_loss = 0

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            # Pixel distance error
            pick_conf = self.attn_forward(inp)['aff'][:, :, :, 0]  # B, H, W 
            pick_conf = pick_conf.detach().cpu().numpy()
            indices = np.argmax(pick_conf.reshape((B,-1)), -1)
            p0_pix = unravel_idx(indices, shape=pick_conf.shape[1:])

            # Depth error
            depth_error = 0
            if self.pred_depth:
                sample = self.depth_est.sample(pred["depth_dist"], reparametrize)
                sample = sample.squeeze().detach().cpu().numpy()
                gt_depth = label["depth"].detach().cpu().numpy()
                depth_error = np.sum(np.abs(sample - gt_depth))
            err = {"px_dist": np.sum(np.linalg.norm(p0 - p0_pix, axis=1)),
                    "depth": depth_error}

        loss = self.loss_weights.aff * aff_loss
        loss += self.loss_weights.depth * depth_loss
        info = {"aff_loss": aff_loss,
                "depth_loss": depth_loss}
        return loss, err, info

    def attn_step(self, frame, label, compute_err=False, reparametrize=False):
        inp_img = frame['img']
        lang_goal = frame['lang_goal']
        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(compute_err, inp, out, label, reparametrize)

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
        out = self.attn_forward(pick_inp)
        pick_conf = out["aff"]
        pick_inp["img"] = img

        err = None
        if info is not None:
            _, err = self.attn_step(pick_inp, info, compute_err=True)

        # Get Aff point
        logits = pick_conf.detach().cpu().numpy().squeeze()
        argmax = np.argmax(logits)
        argmax = np.unravel_index(argmax, shape=logits.shape)
        p0_pix = argmax[:2]
        
        affordance_heatmap_scale = 30
        pick_logits_disp = (logits * 255 * affordance_heatmap_scale).astype('uint8')
        pick_logits_disp_masked = np.ma.masked_where(pick_logits_disp < 0, pick_logits_disp)

        # Get depth
        depth = self.depth_est.sample(out["depth_dist"])

        return {"softmax": pick_logits_disp,
                "pixel": (p0_pix[1], p0_pix[0]),
                "depth": depth,
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