from black import out
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import cv2
import torch
import affordance.utils.flowlib as flowlib
import numpy as np
import pytorch_lightning as pl
import torch
import logging

from thesis.models.lang_fusion.one_stream_attention_lang_fusion_mask import AttentionLangFusionMask
from thesis.utils.utils import add_img_text, tt, blend_imgs,get_transforms, resize_pixel, torch_to_numpy, overlay_flow
from thesis.utils.utils import unravel_idx
from thesis.affordance.hough_voting import hough_voting as hv
from thesis.utils.losses import (
    compute_dice_score,
    compute_dice_loss,
    compute_mIoU,
    CosineSimilarityLossWithMask,
    get_ce_loss,
)

class MaskAffLangDetector(pl.LightningModule):
    def __init__(self, cfg, transforms=None, in_shape=(200, 200, 3), n_classes=2, *args, **kwargs):
        super().__init__()
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.n_classes = n_classes
        self.in_shape = in_shape

        self.optimizer_cfg = cfg.optimizer
        # Loss function
        self.ce_loss = get_ce_loss(cfg.loss, self.n_classes)
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

    def criterion(self, preds, labels):
        # Activation fnc is applied in loss fnc hence, use logits
        # Affordance loss
        aff_loss, info = self.compute_aff_loss(preds["affordance"], labels["affordance"])
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

    def forward(self, inp, softmax=True):
        out = self.attn_forward(inp, softmax=softmax)
        center_dir =  self.center_direction_net(out["decoder_out"][-1])
        preds = {"affordance": out["affordance"],
                 "center_dirs": center_dir}
        return preds

    def attn_step(self, frame, label, compute_err=False):
        inp_img = frame['img']
        lang_goal = frame['lang_goal']
        B = inp_img.shape[0]

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        preds = self.forward(inp, softmax=False)

        total_loss, info = self.criterion(preds, label)
        info["miou"] = compute_mIoU(preds["affordance"], label["affordance"])
        info["dice_score"] = compute_dice_score(preds["affordance"], label["affordance"])

        if compute_err:
            probs = self.attn_forward(inp, softmax=True) # [B x N x H x W]
            aff_mask = torch.argmax(probs["affordance"], 1)
            aff_probs = probs["affordance"][:, 1] * aff_mask

            indices = torch.argmax(aff_probs.reshape(B, -1), -1)
            indices = indices.detach().cpu().numpy()
            p0_pred = unravel_idx(indices, shape=aff_probs.shape[1:])
            p0=label['p0'].detach().cpu().numpy()  # B, 2
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
        :param aff_mask (torch.tensor, int64): [N x H x W], binary {0, 1}
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
            directions = directions.contiguous()
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
        bs = x["img"].shape[0]
        for k, v in info.items():
            self.log("Training/%s" % k, v,
                     on_step=True, on_epoch=True,
                     batch_size=bs)

        return total_loss

    def validation_step(self, val_batch, batch_idx):
        x, labels = val_batch

        # Predictions
        total_loss, info = self.attn_step(x, labels, compute_err=True)

        # Metrics
        mIoU = info["miou"]
        dice_score = info["dice_score"]

        # Log metrics
        bs = x["img"].shape[0]
        self.log_stats("Validation", sum(self.trainer.num_val_batches), batch_idx, total_loss, mIoU)
        for k, v in info.items():
            self.log("Validation/%s" % k, v,
                      on_step=True, on_epoch=True,
                      batch_size=bs)

        return dict(
            val_loss=total_loss,
            val_miou=mIoU,
            val_dice_score=dice_score,
            val_attn_dist_err=info['err'],
            n_imgs=labels['p0'].shape[0],
        )

    def validation_epoch_end(self, all_outputs):
        total_dist_err = np.sum([v['val_attn_dist_err'] for v in all_outputs])
        total_imgs = np.sum([v['n_imgs'] for v in all_outputs])
        mean_img_error = total_dist_err/total_imgs

        self.log('Validation/total_dist_err', total_dist_err)
        self.log('Validation/mean_dist_error', mean_img_error)

        print("\nAttn Err - Dist: {:.2f}".format(total_dist_err))

        return dict(
            total_dist_err=total_dist_err,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **self.optimizer_cfg)
        return optimizer

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
        p0_pix = unravel_idx(idx, shape=aff_probs.shape[1:])[0][::-1]

        # softmax = preds["affordance"][:, 1].detach().cpu().numpy().squeeze()
        softmax = aff_probs.detach().cpu().numpy().squeeze()
        # (column, row) -> (u, v)
        center_dirs = preds["center_dirs"].detach().cpu().numpy()

        return {"softmax": softmax,
                "center_dirs": center_dirs,
                "pixel": p0_pix,
                "error": err}

    def get_preds_viz(self, inp, pred, out_shape=(300, 300), waitkey=0):
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
        return out_img