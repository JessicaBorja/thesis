import numpy as np
from thesis.utils.utils import resize_pixel
import cv2

class BaseAgent:
    def __init__(self, env):
        self._env = env
        _info = self.env.robot.get_observation()[-1]
        self.origin = np.array(_info["tcp_pos"])
        self.target_orn = np.array(_info["tcp_orn"])

    @property
    def env(self):
        return self._env

    def reset_position(self):
        return self.move_to(self.origin,
                            self.target_orn,
                            1)

    def viz_img(rgb_img, center_pixel, output_size=(300, 300), lang_goal=""):
        old_shape = rgb_img.shape[:2]
        pixel = resize_pixel(center_pixel, old_shape, output_size)
        out_img = cv2.drawMarker(
                rgb_img.copy(),
                (pixel[0], pixel[1]),
                (0, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=12,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
        out_img = cv2.resize(out_img, output_size)
        if(lang_goal != ""):
            # Prints the text.
            font_scale = 0.6
            thickness = 2
            color = (0, 0, 0)
            x1, y1 = 10, 20
            (w, h), _ = cv2.getTextSize(lang_goal, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            out_img = cv2.rectangle(out_img, (x1, y1 - 20), (x1 + w, y1 + h), color, -1)
            out_img = cv2.putText(
                out_img,
                lang_goal,
                org=(x1, y1),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=thickness,
            )

        cv2.imshow("orig img",out_img[:, :, ::-1])
        cv2.waitKey(1)

    def move_to(self, target_pos, target_orn=None, gripper_action=None):
        '''
            Move to the specified location in world coordinates
        '''
        curr_info = self.env.robot.get_observation()[-1]
        if(target_orn is None):
            target_orn = np.array(curr_info["tcp_orn"])
        if(gripper_action is None):
            gripper_action = curr_info["gripper_action"]
        action = [target_pos.copy(),
                  target_orn.copy(),
                  gripper_action]
        last_pos = target_pos.copy()
        # When robot is moving and far from target
        ns, r, d, info = self.env.step(action)
        curr_pos = np.array(info["robot_info"]["tcp_pos"])
        while(np.linalg.norm(curr_pos - target_pos) > 0.01 and np.linalg.norm(last_pos - curr_pos) > 0.001):
            last_pos = curr_pos
            ns, r, d, info = self.env.step(action)             
            curr_pos = np.array(info["robot_info"]["tcp_pos"])

            cv2.imshow("obs", ns["rgb_obs"]["rgb_static"][:, :, ::-1])
            cv2.waitKey(1)
        return ns