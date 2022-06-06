import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import logging
from omegaconf import open_dict

from thesis.models.lang_fusion.aff_lang_depth_pixel import AffDepthLangFusionPixel
from thesis.utils.utils import add_img_text, tt, blend_imgs,get_transforms, resize_pixel, unravel_idx
from thesis.utils.losses import cross_entropy_with_logits

class PixelAffLangDetector(LightningModule):
    def __init__(self, cfg,
                 in_shape=(200, 200, 3),
                 transforms=None,
                 depth_dist=None,
                 depth_norm_values=None,
                 *args, **kwargs):
        super().__init__()
        self.loss_weights = cfg.loss_weights
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # append depth transforms to cfg
        with open_dict(cfg):
            cfg.depth_norm_values=depth_norm_values
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
        self.model.train()

        frame, label = batch

        # Get training losses.
        step = self.total_steps + 1
        total_loss, err, info = self.step(frame, label, compute_err=True)
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
        self.model.eval()

        frame, label = batch
        val_total_loss, err, info = self.step(frame, label, compute_err=True)
        bs = frame["img"].shape[0]
        self.log('Validation/total_loss', val_total_loss,
                 on_step=False, on_epoch=True,
                 batch_size=bs)
        for loss_fnc, value in info.items():
            self.log('Validation/%s' % loss_fnc, value,
                     on_step=False, on_epoch=True,)
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
        optim = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        return optim

    def _build_model(self):
        self.model = AffDepthLangFusionPixel(
            modules_cfg=[self.cfg.streams.vision_net,
                         self.cfg.streams.lang_enc,
                         self.depth_est_dist],
            in_shape=self.in_shape,
            cfg=self.cfg,
            device=self.device_type,
        )

    def forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']
        output, _info = self.model(inp_img, lang_goal, softmax=softmax)
        return output

    def criterion(self, compute_err, inp, pred, label, reparametrize=False):
        if self.model.depth_stream.normalized:
            depth_label = "normalized_depth"
        else:
            depth_label = "depth"

        # AFFORDANCE CRITERION #
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
            gt_depth = label[depth_label].unsqueeze(-1).float()
            depth_loss = self.model.depth_stream.loss(pred['depth_dist'], gt_depth)
        else: 
            depth_loss = 0

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            # Pixel distance error
            p0_pix, depth_sample, _ = self.model.predict(**inp)  # B, H, W 
            # Depth error
            depth_error = 0
            if self.pred_depth:
                # Depth sample is unormalized in predict
                unormalized_depth = label["depth"].detach().cpu().numpy()
                depth_error = np.sum(np.abs(depth_sample - unormalized_depth))
            err = {"px_dist": np.sum(np.linalg.norm(p0 - p0_pix, axis=1)),
                    "depth": depth_error}

        loss = self.loss_weights.aff * aff_loss
        loss += self.loss_weights.depth * depth_loss
        info = {"aff_loss": aff_loss,
                "depth_loss": depth_loss}
        return loss, err, info

    def step(self, frame, label, compute_err=False, reparametrize=False):
        inp_img = frame['img']
        lang_goal = frame['lang_goal']
        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        out = self.forward(inp, softmax=False)
        return self.criterion(compute_err, inp, out, label, reparametrize)

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
        p0_pix, depth, logits = self.model.predict(pick_inp)
        p0_pix = p0_pix.squeeze()
        depth = depth.squeeze()
        pick_inp["img"] = img

        err = None
        if info is not None:
            _, err = self.step(pick_inp, info, compute_err=True)

        # Get Aff mask
        affordance_heatmap_scale = 30
        pick_logits_disp = (logits * 255 * affordance_heatmap_scale).astype('uint8')
        pick_logits_disp_masked = np.ma.masked_where(pick_logits_disp < 0, pick_logits_disp)

        return {"softmax": pick_logits_disp,
                "pixel": (p0_pix[1], p0_pix[0]),
                "depth": depth,
                "error": err}

    def get_preds_viz(self, inp, pred, out_shape=(300, 300), waitkey=0):
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

        pred_img = cv2.resize(pred_img, out_shape, interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.resize(heatmap, out_shape, interpolation=cv2.INTER_CUBIC)
        pred_img = pred_img.astype(float) / 255
        out_img = np.concatenate([pred_img, heatmap], axis=1)


        # Prints the text.
        out_img = add_img_text(out_img, inp["lang_goal"])
        return out_img