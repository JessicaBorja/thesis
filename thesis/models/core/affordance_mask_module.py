import numpy as np
import pytorch_lightning as pl
import torch
import logging

from affordance.hough_voting import hough_voting as hv
from affordance.utils.losses import (
    compute_dice_loss,
    compute_dice_score,
    compute_mIoU,
    CosineSimilarityLossWithMask,
    get_affordance_loss,
)


class AffordanceMaskModule(pl.LightningModule):
    def __init__(self, cfg, in_shape=(3, 200, 200), n_classes=2, *args, **kwargs):
        super().__init__()
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.n_classes = n_classes
        self.in_shape = in_shape

        self.optimizer_cfg = cfg.optimizer
        # Loss function
        self.affordance_loss = get_affordance_loss(cfg.loss, self.n_classes)
        self.center_loss = CosineSimilarityLossWithMask(weighted=True)
        self.loss_w = cfg.loss

        # Misc
        self.cmd_log = logging.getLogger(__name__)
        self._batch_loss = []
        self._batch_miou = []

        # Hough Voting stuff (this operates on CUDA only)
        self.hough_voting_layer = hv.HoughVoting(**cfg.hough_voting)
        print("hvl init")

        self._build_model()
        self.save_hyperparameters()

    def _build_model(self):
        self.attention = None
        raise NotImplementedError()

    def compute_aff_loss(self, preds, labels):
        # Preds = (B, C, H, W)
        # labels = (B, H, W)
        B, C, H, W = preds.shape
        if C == 1:
            # BCE needs B, H, W
            preds = preds.squeeze(1)
            labels = labels.float()
        ce_loss = self.affordance_loss(preds, labels)
        info = {"CE_loss": ce_loss}
        loss = self.loss_w.ce_loss * ce_loss

        # Add dice if required
        if self.loss_w.affordance.add_dice:
            if C == 1:
                # Dice needs B, C, H, W
                preds = preds.unsqueeze(1)
                labels = labels.unsqueeze(1)
            # label_spatial = pixel2spatial(labels.long(), H, W)
            dice_loss = compute_dice_loss(labels.long(), preds)
            info["dice_loss"] = dice_loss
            loss += self.loss_w.dice * dice_loss
        return loss, info

    def criterion(self, preds, labels, compute_err=False):
        # Activation fnc is applied in loss fnc hence, use logits
        # Affordance loss
        aff_loss, info = self.compute_aff_loss(preds["aff_logits"], labels["affordance"])
        info["aff_loss"] = aff_loss

        # Center prediction loss
        if self.n_classes > 2:
            bin_mask = torch.zeros_like(labels["affordance"])
            bin_mask[labels["affordance"] > 0] = 1
        else:
            bin_mask = labels["affordance"]
        center_loss = self.center_loss(preds["center_dirs"], labels["center_dirs"], bin_mask)
        info.update({"center_loss": center_loss})

        # Total loss
        total_loss = aff_loss + self.loss_w.centers * center_loss

        info["total_loss"] = total_loss
        return total_loss, info

    def forward(self, inp):
        out = self.attn_forward(inp, softmax=True)
        center_dir =  self.center_direction_net(out["decoder"])
        preds = {"aff_logits": out["affordance"],
                 "center_dirs": center_dir}

        return preds

    def attn_step(self, frame, label, compute_err=False):
        inp_img = frame['img']
        lang_goal = frame['lang_goal']
        B = inp_img.shape[0]

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        logits = self.attn_forward(inp, softmax=False)
        center_dir =  self.center_direction_net(logits["decoder"])
        preds = {"aff_logits": logits["affordance"],
                 "center_dirs": center_dir}

        total_loss, info = self.criterion(preds, label)
        info["miou"] = compute_mIoU(logits["affordance"], label["affordance"])
        info["dice_score"] = compute_dice_score(logits["affordance"], label["affordance"])

        if compute_err:
            probs = self.attn_forward(inp, softmax=True)
            aff_mask = torch.argmax(probs["affordance"], dim=1)  # [N x H x W]
            _c_info = self.pred_centers(aff_mask, center_dir)
            p0=label['p0'].detach().cpu().numpy()  # B, 2
            pt_preds = []
            for pred in _c_info[0]:
                if(len(pred) > 0):
                    p0_pred = pred.detach().cpu().numpy()
                else:
                    p0_pred = np.array([[1.0, 0.0]])
                pt_preds.append(p0_pred)
            p0_pred = np.stack(pt_preds)
            info['err'] = np.sum(np.linalg.norm(p0 - p0_pred, axis=1))
        return total_loss, info

    def attn_forward(self, inp, softmax=False):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']
        output = self.attention(inp_img, lang_goal, softmax=softmax)
        return output

    # Center prediction
    def pred_centers(self, aff_mask, directions):
        """
        Take the direction vectors and foreground mask to get hough voting layer prediction
        :param aff_mask (torch.tensor, int64): [N x H x W]
        :param directions (torch.tensor, float32): [N x 2 x H x W]

        :return object_centers (list(torch.tensor), int64)
        :return directions (torch.tensor, float32): [N x 2 x H x W] normalized directions
        :return initial_masks (torch.tensor): [N x 1 x H x W]
            - range: int values indicating object mask (0 to n_objects)
        """
        # x.shape = (B, C, H, W)
        if self.n_classes > 2:
            if len(aff_mask.shape) == 4:
                bin_mask = aff_mask.any(1)
            else:
                bin_mask = torch.zeros_like(aff_mask)
                bin_mask[aff_mask > 0] = 1
        else:
            bin_mask = aff_mask

        with torch.no_grad():
            # Center direction
            directions /= torch.norm(directions, dim=1, keepdim=True).clamp(min=1e-10)
            directions = directions.float()
            initial_masks, num_objects, object_centers_padded = self.hough_voting_layer(
                (bin_mask == 1).int(), directions
            )

        # Compute list of object centers
        batch_centers = []
        for i in range(initial_masks.shape[0]):
            centers_padded = object_centers_padded[i]
            centers_padded = centers_padded.permute((1, 0))[: num_objects[i], :]
            object_centers = []
            for obj_center in centers_padded:
                if torch.norm(obj_center) > 0:
                    # cast to int for pixel
                    object_centers.append(obj_center.long())
            if(len(object_centers) > 0):
                batch_centers.append(torch.stack(object_centers))
            else:
                batch_centers.append([])
        return batch_centers, directions, initial_masks

    def log_stats(self, split, max_batch, batch_idx, loss, miou):
        if batch_idx >= max_batch - 1:
            e_loss = 0 if len(self._batch_loss) == 0 else np.mean(self._batch_loss)
            e_miou = 0 if len(self._batch_miou) == 0 else np.mean(self._batch_miou)
            self.cmd_log.info(
                "%s [epoch %4d]" % (split, self.current_epoch) + "loss: %.3f, mIou: %.3f" % (e_loss, e_miou)
            )
            self._batch_loss = []
            self._batch_miou = []
        else:
            self._batch_loss.append(loss.item())
            self._batch_miou.append(miou.item())

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, labels = batch
        total_loss, info = self.attn_step(x, labels)

        # Metrics
        mIoU = info["miou"]

        # Log metrics
        self.log_stats("Training", sum(self.trainer.num_val_batches), batch_idx, total_loss, mIoU)
        for k, v in info.items():
            self.log("Training/%s" % k, v, on_step=False, on_epoch=True)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch

        # Predictions
        total_loss, info = self.attn_step(x, labels, compute_err=False)

        # Metrics
        mIoU = info["miou"]
        dice_score = info["dice_score"]

        # Log metrics
        self.log_stats("Validation", sum(self.trainer.num_val_batches), batch_idx, total_loss, mIoU)
        for k, v in info.items():
            self.log("Validation/%s" % k, v, on_step=False, on_epoch=True)

        return dict(
            val_loss=total_loss,
            val_miou=mIoU,
            val_dice_score=dice_score,
            # val_attn_dist_err=info['err'],
            n_imgs=labels['p0'].shape[0],
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_cfg)
        return optimizer
