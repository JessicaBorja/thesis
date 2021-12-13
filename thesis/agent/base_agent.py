import numpy as np


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
        return ns