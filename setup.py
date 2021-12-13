from setuptools import setup

setup(name='thesis',
      version='1.0',
      description='Python Distribution Utilities',
      packages=['thesis'],
      install_requires=[
          'opencv-python(==4.5.3.56)',
          'pybullet(==3.1.7)',
          'hydra-core',
          'numpy-quaternion',
          'hydra-colorlog',
          'pypng',
          'tqdm',
          'wandb',
          'omegaconf',
          'matplotlib']
     )
