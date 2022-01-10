from cliport.utils import utils
from thesis.agents.cliport.pick_module import PickModule
from thesis.agents.cliport.attention import OneStreamAttentionLangFusion
import numpy as np


class ClipLingUNetDetector(PickModule):

    def __init__(self, cfg):
        super().__init__(cfg)

    def _build_model(self):
        stream_fcn = 'clip_lingunet'
        self.attention = OneStreamAttentionLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
    
    def attn_forward(self, inp, softmax=True):
        inp_img = inp['inp_img']
        lang_goal = inp['lang_goal']
        out = self.attention(inp_img, lang_goal, softmax=softmax)
        return out

    def attn_step(self, frame, label, backprop=True, compute_err=False):
        inp_img = frame['img']
        lang_goal = frame['lang_goal']

        inp = {'inp_img': inp_img, 'lang_goal': lang_goal}
        # p0, p0_theta = label['p0'], label['p0_theta']
        out = self.attn_forward(inp, softmax=False)
        return self.attn_criterion(compute_err, inp, out, label)

    def act(self, obs, info, goal=None):  # pylint: disable=unused-argument
        """Run inference and return best action given visual observations."""
        # Get heightmap from RGB-D images.
        img = self.test_ds.get_image(obs)
        lang_goal = info['lang_goal']

        # Attention model forward pass.
        pick_inp = {'inp_img': img, 'lang_goal': lang_goal}
        pick_conf = self.attn_forward(pick_inp)
        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0_pix = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / pick_conf.shape[2])

        # Pixels to end effector poses.
        # hmap = img[:, :, 3]
        # p0_xyz = utils.pix_to_xyz(p0_pix, hmap, self.bounds, self.pix_size)
        # p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))

        return {
            # 'pose0': (np.asarray(p0_xyz), np.asarray(p0_xyzw)),
            'pick': [p0_pix[0], p0_pix[1], p0_theta],
        }