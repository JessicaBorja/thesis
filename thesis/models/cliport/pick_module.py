import os
import numpy as np

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from cliport.utils import utils
import logging



class PickModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        utils.set_seed(0)
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this is bad for PL :(
        self.cfg = cfg
        self.val_repeats = cfg.train.val_repeats
        self.total_steps = 0
        self.in_shape = (200, 200, 3)
        self._batch_loss = []
        self.cmd_log = logging.getLogger(__name__)

        self._build_model()

    def _build_model(self):
        self.attention = None
        raise NotImplementedError()

    def forward(self, x):
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
        inp_img = inp['img']
        lang_goal = inp['lang_goal']
        out = self.attention(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_step(self, frame, label, compute_err=False):
        out = self.attn_forward(frame, softmax=False)
        return self.attn_criterion(compute_err, frame, out, label)

    def attn_criterion(self, compute_err, inp, out, label):
        # Get label.
        # theta = label['theta'].detach().cpu().numpy()
        # theta_i = theta / (2 * np.pi / self.attention.n_rotations)
        # theta_i = np.int32(np.round(theta_i)) % self.attention.n_rotations

        inp_img = inp['inp_img']
        # label_size = inp_img.shape[:2] + (self.attention.n_rotations,)
        # label[p[0], p[1], theta_i] = 1
        label_size = inp_img.shape[2:] + (1, )
        aff_label = torch.zeros(label_size)
        p0=label['p0'][0].detach().cpu().numpy()
        aff_label[p0[0], p0[1], 0] = 1

        aff_label = aff_label.permute((2, 0, 1))
        aff_label = aff_label.reshape(1, np.prod(aff_label.shape))
        aff_label = aff_label.to(dtype=torch.float, device=out.device)

        # Get loss.
        loss = self.cross_entropy_with_logits(out, aff_label)

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            pick_conf = self.attn_forward(inp)
            pick_conf = pick_conf.detach().cpu().numpy()
            argmax = np.argmax(pick_conf)
            argmax = np.unravel_index(argmax, shape=pick_conf.shape)
            p0_pix = argmax[:2]
            p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

            err = {
                'dist': np.linalg.norm(p0 - p0_pix, ord=1),
                #'theta': np.absolute((theta - p0_theta) % np.pi)
            }
        return loss, err

    def training_step(self, batch, batch_idx):
        self.attention.train()

        frame, label = batch

        # Get training losses.
        step = self.total_steps + 1
        loss0, err0 = self.attn_step(frame, label)
        total_loss = loss0
        self.log('tr/attn/loss', loss0)
        self.log('tr/loss', total_loss)
        self.total_steps = step
        # self.check_save_iteration()

        return dict(
            loss=total_loss,
        )

    def check_save_iteration(self):
        global_step = self.trainer.global_step
        if (global_step + 1) in self.save_steps:
            self.trainer.run_evaluation()
            val_loss = self.trainer.callback_metrics['val_loss']
            steps = f'{global_step + 1:05d}'
            filename = f"steps={steps}-val_loss={val_loss:0.8f}.ckpt"
            checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
            ckpt_path = os.path.join(checkpoint_path, filename)
            self.trainer.save_checkpoint(ckpt_path)

        if (global_step + 1) % 1000 == 0:
            self.save_last_checkpoint()

    def save_last_checkpoint(self):
        checkpoint_path = os.path.join(self.cfg['train']['train_dir'], 'checkpoints')
        ckpt_path = os.path.join(checkpoint_path, 'last.ckpt')
        self.trainer.save_checkpoint(ckpt_path)

    def log_stats(self, split, max_batch, batch_idx, loss, error):
        if batch_idx >= max_batch - 1:
            e_loss = 0 if len(self._batch_loss) == 0 else np.mean(self._batch_loss)
            self.cmd_log.info(
                "%s [epoch %4d]" % (split, self.current_epoch) + "loss: %.3f, error: %.3f" % (e_loss, error)
            )
            self._batch_loss = []
        else:
            self._batch_loss.append(loss.item())

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

        # Log metrics on command line
        # self.log_stats("validation", sum(self.trainer.num_val_batches), batch_idx, val_total_loss, err0)

        return dict(
            val_loss=val_total_loss,
            val_attn_dist_err=err0['dist'],
            # val_attn_theta_err=err0['theta'],
        )

    # def training_epoch_end(self, all_outputs):
    #     super().training_epoch_end(all_outputs)
    #     utils.set_seed(self.trainer.current_epoch+1)

    def validation_epoch_end(self, all_outputs):
        mean_val_total_loss = np.mean([v['val_loss'].item() for v in all_outputs])
        total_attn_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        # total_attn_theta_err = np.sum([v['val_attn_theta_err'] for v in all_outputs])
    
        self.log('vl/loss', mean_val_total_loss)
        self.log('vl/total_attn_dist_err', total_attn_dist_err)
        # self.log('vl/total_attn_theta_err', total_attn_theta_err)

        print("\nAttn Err - Dist: {:.2f}".format(total_attn_dist_err))

        return dict(
            val_loss=mean_val_total_loss,
            total_attn_dist_err=total_attn_dist_err,
            # total_attn_theta_err=total_attn_theta_err,
        )


    def configure_optimizers(self):
        optim = torch.optim.Adam(self.attention.parameters(), lr=self.cfg.train.lr)
        return optim

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path)['state_dict'])
        self.to(device=self.device_type)