# Installation
- Setup a conda environment by running:
```
git clone https://github.com/JessicaBorja/thesis.git
cd thesis/
conda create -n thesis python=3.8
conda activate thesis
```

Install pytorch
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

Install VREnv
```
cd VREnv/
pip install pybullet
pip install -e .
```

Install repo
```
pip install -e .
```

# Training Affordance Model


# Testing / Evaluation
## Visualizing the predictions on the dataset

## Testing affordance + model-based given language annotation

## Evaluation model-based + model-free with affordance