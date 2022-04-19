import os
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import logging


class AffordancePixelModule(LightningModule):
    def __init__(self, cfg, in_shape=(200, 200, 3)):
        super().__init__()
        self.loss_weights = cfg.loss_weights
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.total_steps = 0
        self.in_shape = in_shape
        self._batch_loss = []
        self.cmd_log = logging.getLogger(__name__)
        self._build_model()
        self.save_hyperparameters()
    
    def _build_model(self):
        self.attention=None

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

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
        aff_loss = self.cross_entropy_with_logits(pred["aff"], aff_label)
        if self.pred_depth:
            gt_depth = label["depth"].unsqueeze(-1).float()
            depth_loss = self.depth_est.loss(pred, gt_depth)
        else: 
            depth_loss = 0

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            # Pixel distance error
            pick_conf = self.attn_forward(inp)['aff'][:, :, :, 0]  # B, H, W 
            pick_conf = pick_conf.detach().cpu().numpy()
            indices = np.argmax(pick_conf.reshape((B,-1)), -1)
            p0_pix = self.unravel_idx(indices, shape=pick_conf.shape[1:])
            err = {'px_dist': np.sum(np.linalg.norm(p0 - p0_pix, axis=1))}

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

    def unravel_idx(self, indices, shape):
        coord = []
        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = indices // dim

        coord = np.stack(coord[::-1], axis=-1)
        return coord

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
            self.log('Validation/%s' % loss_fnc, value,
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


        self.log('Validation/dist_err', err['px_dist'])
        self.log('Validation/total_loss', val_total_loss,
                 on_step=False, on_epoch=True,
                 batch_size=bs)
        for loss_fnc, value in info.items():
            self.log('Validation/%s' % loss_fnc, value,
                     on_step=False, on_epoch=True,)

        if self.pred_depth:
            self.log('Validation/depth_err', err['depth'],
                    on_step=False, on_epoch=True,
                    batch_size=bs)

        # return dict(
        #     val_loss=val_total_loss,
        #     val_attn_dist_err=err['px_dist'],
        #     val_depth_err=err["depth"],
        #     n_imgs=batch[1]['p0'].shape[0],
        # )

    # def validation_epoch_end(self, all_outputs):
    #     mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])        
    #     total_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
    #     total_imgs = np.sum([v['n_imgs'] for v in all_outputs])
    #     mean_img_error = total_dist_err/total_imgs

    #     self.log('Validation/total_loss', mean_val_total_loss)
    #     self.log('Validation/total_dist_err', total_dist_err)
    #     self.log('Validation/mean_dist_error', mean_img_error)

    #     if self.pred_depth:
    #         mean_val_depth_err = np.mean([v['val_depth_err'].item() for v in all_outputs])
    #         self.log('Validation/mean_val_depth_err', mean_val_depth_err)

    #     print("\nAttn Err - Dist: {:.2f}".format(total_dist_err))

    #     return dict(
    #         val_loss=mean_val_total_loss,
    #         total_dist_err=total_dist_err,
    #     )

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.attention.parameters(), lr=self.cfg.lr)
        return optim