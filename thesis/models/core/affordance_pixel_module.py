import os
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import logging


class AffordancePixelModule(LightningModule):
    def __init__(self, cfg, in_shape=(3, 200, 200)):
        super().__init__()
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.val_repeats = cfg.val_repeats
        self.total_steps = 0
        self.in_shape = in_shape
        self._batch_loss = []
        self.cmd_log = logging.getLogger(__name__)
        self._build_model()
        self.save_hyperparameters()

    def _build_model(self):
        self.attention = None
        raise NotImplementedError()

    def cross_entropy_with_logits(self, pred, labels, reduction='mean'):
        # Lucas found that both sum and mean work equally well
        x = (-labels * F.log_softmax(pred, -1))
        if reduction == 'sum':
            return x.sum()
        elif reduction == 'mean':
            return x.mean()
        else:
            raise NotImplementedError()

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

    def attn_criterion(self, compute_err, inp, out, label):

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
        aff_label = aff_label.to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = self.cross_entropy_with_logits(out, aff_label)

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)[:, :, :, 0]  # B, H, W 
            pick_conf = pick_conf.detach().cpu().numpy()
            indices = np.argmax(pick_conf.reshape((B,-1)), -1)
            p0_pix = self.unravel_idx(indices, shape=pick_conf.shape[1:])
            err = {
                'dist': np.sum(np.linalg.norm(p0 - p0_pix, axis=1)),
            }
        return loss, err

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
        loss0, err0 = self.attn_step(frame, label)
        total_loss = loss0
        self.log('Training/total_loss', total_loss, on_step=False, on_epoch=True)
        self.total_steps = step
        # self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def validation_step(self, batch, batch_idx):
        self.attention.eval()

        loss0 = 0
        assert self.val_repeats >= 1
        for i in range(self.val_repeats):
            frame, label = batch
            l0, err0 = self.attn_step(frame, label, compute_err=True)
            loss0 += l0
        loss0 /= self.val_repeats
        val_total_loss = loss0

        return dict(
            val_loss=val_total_loss,
            val_attn_dist_err=err0['dist'],
            n_imgs=batch[1]['p0'].shape[0],
        )

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        total_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_imgs = np.sum([v['n_imgs'] for v in all_outputs])
        mean_img_error = total_dist_err/total_imgs

        self.log('Validation/total_loss', mean_val_total_loss)
        self.log('Validation/total_dist_err', total_dist_err)
        self.log('Validation/mean_dist_error', mean_img_error)

        print("\nAttn Err - Dist: {:.2f}".format(total_dist_err))

        return dict(
            val_loss=mean_val_total_loss,
            total_dist_err=total_dist_err,
        )

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.attention.parameters(), lr=self.cfg.lr)
        return optim