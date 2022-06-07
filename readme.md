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
## Find normalization values for depth
Script: [find_norm_values.py](./scripts/find_norm_values.py)

### Running on a cluster:
```
cd run_on_cluster
python cluster.py -v "env" --train_file "../scripts/find_norm_values.py" --no_clone  -j norm_values --data_dir DATASET_ABS_PATH
```

## Train model
Script: [train_affordance.py](./train_affordance.py)

Config: [train_affordance.yaml](./config/train_affordance.yaml)

# Testing / Evaluation
## Visualizing the predictions on the dataset
Script: [test_affordance.py](./scripts/test_affordance.py)

Config: [test_affordance.yaml](./config/test_affordance.yaml)

Usage:
```
python test_affordance.py checkpoint.train_folder=AFFORDANCE_TRAIN_FOLDER aff_detection.dataset.data_dir=DATASET_TO_TEST_PATH
```

## Testing move to a point given language annotation
Script: [test_move_to_pt.py](./scripts/test_move_to_pt.py)

Config: [cfg_calvin.yaml](./config/cfg_calvin.yaml)

## Evaluation model-based + model-free with affordance
Script: [evaluate_policy.py](./thesis/evaluation/evaluate_policy.py)

### Running on a cluster:
```
cd run_on_cluster
python slurm_eval.py -v thesis --dataset_path DATA_TO_EVAL_PATH --train_folder POLICY_TRAIN_FOLDER --checkpoint CHECKPOINT_PATH/epoch=N --aff_lmp
```
