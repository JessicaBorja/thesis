from setuptools import setup

setup(name='thesis',
      version='1.0',
      description='Python Distribution Utilities',
      packages=['thesis'],
      install_requires=[
          'opencv-python',
          'pybullet',
          'hydra-core',
          'numpy-quaternion',
          'hydra-colorlog',
          'pypng',
          'tqdm',
          'wandb',
          'omegaconf',
          'matplotlib']
     )
