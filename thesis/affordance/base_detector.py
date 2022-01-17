from threading import currentThread
import torch

class BaseDetector:
    def __init__(self, *args, **kwargs):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def cuda(self):
        pass

    def eval(self):
        pass

    def load_from_checkpoint(self, *args, **kwargs):
        pass

    def find_target(self, inputs: dict):
        pass