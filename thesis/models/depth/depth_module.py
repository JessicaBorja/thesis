import numpy as np

import torch
from pytorch_lightning import LightningModule
import thesis.models as models
from thesis.models.language_encoders.clip_lang_encoder import CLIPLang
from thesis.datasets.transforms import NormalizeVectorInverse


class DepthModule(LightningModule):
    def __init__(self, cfg, in_shape=(200, 200, 3), depth_transforms=None):
        super().__init__()
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.in_shape = in_shape
        self.normalized = cfg.normalize_depth
        self.init_depth_transforms(depth_transforms)
        self._build_model()
        self.save_hyperparameters()
        self.text_enc = CLIPLang(self.device_type)

    def init_depth_transforms(self, depth_norm_values):
        self.depth_norm_inverse = NormalizeVectorInverse(depth_norm_values["mean"], depth_norm_values["std"])

    def _build_model(self):
        _depth_est = models.deth_est_nets[self.cfg.depth_dist]
        self.depth_est = _depth_est(self.in_shape, 1, self.cfg, self.device_type)

    def forward(self, inp):
        '''
            inp(dict):
                - 'img'(torch.Tensor): 
        '''
        text_enc = self.text_enc(inp['lang_goal'])
        dist, _info = self.depth_est(inp['img'], text_enc[0])
        out = {"depth_dist": dist}
        return out

    def predict(self, inp, transforms):
        inp_img = torch.tensor(inp['img']).permute((2, 0, 1)).unsqueeze(0).to(self.device)
        inp_img = transforms(inp_img)
        # dist, _info = self.depth_est(inp['img'], inp['lang_goal'])
        net_inp = {"img": inp_img, "lang_goal": inp["lang_goal"]}
        dist = self.forward(net_inp)
        sample = self.depth_est.sample(dist["depth_dist"])
        if self.normalized:
            sample = self.depth_norm_inverse(sample)
        sample = sample.squeeze().detach().cpu().numpy()
        return sample

    def criterion(self, pred, label, compute_err):
        if self.normalized:
            depth_label = "normalized_depth"
        else:
            depth_label = "depth"

        gt_depth = label[depth_label].unsqueeze(-1).float()
        depth_loss = self.depth_est.loss(pred, gt_depth)

        # Pixel and Rotation error (not used anywhere).
        err = {}
        if compute_err:
            # Always compute error w.r.t true depth (not normalized)
            sample = self.depth_est.sample(pred["depth_dist"])
            # Unnormalize to match real depth
            if self.normalized:
                sample = self.depth_norm_inverse(sample)
            sample = sample.squeeze().detach().cpu().numpy()
            unormalized_depth = label["depth"].unsqueeze(-1).float()
            unormalized_depth = label["depth"].detach().cpu().numpy()
            depth_error = np.sum(np.abs(sample - unormalized_depth))
            err = {"depth": depth_error}

        loss = depth_loss
        return loss, err

    def training_step(self, batch, batch_idx):
        frame, label = batch

        # Get training losses.
        pred = self.forward(frame)
        total_loss, err = self.criterion(pred, label, compute_err=True)
        bs = frame["img"].shape[0]

        self.log('Training/total_loss', total_loss,
                 on_step=False, on_epoch=True,
                 batch_size=bs)
        self.log('Training/depth_err', err['depth'],
                on_step=False, on_epoch=True,
                batch_size=bs)

        return total_loss

    def validation_step(self, batch, batch_idx):
        frame, label = batch

        pred = self.forward(frame)
        total_loss, err = self.criterion(pred, label, compute_err=True)
        bs = frame["img"].shape[0]


        self.log('Validation/total_loss', total_loss,
                 on_step=False, on_epoch=True,
                 batch_size=bs)
        self.log('Validation/depth_err', err['depth'],
                on_step=False, on_epoch=True,
                batch_size=bs)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optim