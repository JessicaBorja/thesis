from setuptools import setup

setup(name='thesis',
      version='1.0',
      description='Python Distribution Utilities',
      packages=['thesis'],
      install_requires=[
          'sklearn',
          'opencv-python',
          'pytorch-lightning',
          'pybullet',
          'hydra-core',
          'numpy-quaternion',
          'hydra-colorlog',
          'pypng',
          'tqdm',
          'wandb',
          'omegaconf',
          'matplotlib',
          'sentence-transformers',
          'segmentation_models_pytorch',
          'ftfy']
     )
