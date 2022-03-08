from thesis.agents.base_agent import BaseAgent
from thesis.utils.utils import get_abspath
from calvin_agent.models.play_lmp import PlayLMP
import os


class PlayLMPAgent(BaseAgent):
    def __init__(self, env, checkpoint=None):
        super().__init__(env)
        if checkpoint:
            self.policy = self.load_policy(**checkpoint)
        else:
            self.policy = PlayLMP()
    
    def load_policy(self, train_folder, model_name, **kwargs):
        # Load model
        train_folder = get_abspath(train_folder)
        checkpoint = os.path.join(train_folder, model_name)
        if(os.path.isfile(checkpoint)):
            model = PlayLMP.load_from_checkpoint(checkpoint)
            model = model.to(self.device)
            model.freeze()
            self.logger.info("Model successfully loaded: %s" % checkpoint)
        else:
            self.logger.info("No checkpoint file found, loading untrained model: %s" % checkpoint)
        return model

    def step(self, obs, step):
        return self.policy.step(obs, step)