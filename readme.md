# Installation
Setup a conda environment by running:
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
git clone https://github.com/JessicaBorja/VREnv/
cd VREnv/
pip install pybullet
pip install -e .
```

Install thesis repo
```
cd ..
pip install -e .
```

# Training Affordance Model
## Generate affordance dataset

The dataset will automatically be generated in /home/USER/datasets. To change this add the following line:

paths.datasets=DATASETS_PATH

The paths configuration is taken from [general_paths.yaml](./config/paths/general_paths.yaml)

### Training
```
python cluster.py -v "thesis" --train_file "../thesis/affordance/dataset_creation/data_labeler_lang.py" --no_clone -j labeling \\
paths.parent_folder=/home/USER/ \\
play_data_dir=/work/dlclarge2/meeso-lfp/calvin_data/task_D_D/training \\
dataset_name=calvin_langDepthEndPt/training \\
output_cfg.single_split=training \\
output_size.static=200 \\
output_size.gripper=86 \\
labeling=simulation_lang \\
mask_on_close=True \\
```
### Validation
```
python cluster.py -v "thesis" --train_file "../thesis/affordance/dataset_creation/data_labeler_lang.py" --no_clone -j labeling \\
paths.parent_folder=/home/USER/ \\
play_data_dir=/work/dlclarge2/meeso-lfp/calvin_data/task_D_D/validation \\
dataset_name=calvin_langDepthEndPt/validation \\
output_cfg.single_split=validation \\
output_size.static=200 \\
output_size.gripper=86 \\
labeling=simulation_lang \\
mask_on_close=True \\
```
### Merging the datasets
The directories to merge are specified in [cfg_merge_dataset.yaml](./config/cfg_merge_dataset.yaml). They can be relative to the "thesis" directory or absolute paths. By default the script outputs to the parent of the first directory in the list of cfg_merge_dataset.yaml

```
python merge_datasets.py
```

## Find normalization values for depth prediction
Script: [find_norm_values.py](./scripts/find_norm_values.py)

### Running on a cluster:
```
cd run_on_cluster
python cluster.py -v "thesis" --train_file "../scripts/find_norm_values.py" --no_clone  -j norm_values --data_dir /DATASETS_PATH/calvin_langDepthEndPt/
```

## Train model
### Running on a cluster:
```
cd run_on_cluster
python cluster.py -v "thesis" --train_file "../train_affordance.py" --no_clone \\
-j aff_model \\
paths.parent_folder=~/ \\
run_name=WANDB_RUN_NAME \\
aff_detection=VISUAL_LANGENC_LABEL_TYPE \\
aff_detection.streams.lang_enc=LANGENC \\ dataset_name=calvin_langDepthEndPt \\
aff_detection.model_cfg.freeze_encoder.lang=True \\
aff_detection.model_cfg.freeze_encoder.aff=False \\
wandb.logger.group=aff_exps
```
Available configurations for aff_detection:
- rn18_bert_pixel
- rn18_clip_pixel (rn18 visual encoder + clip language)
- rn50_bert_pixel
- rn50_clip_pixel (clip for visual and language)

Available language encoders:
- clip
- bert
- distilbert
- sbert

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
python slurm_eval.py -v thesis --dataset_path DATA_TO_EVAL_PATH \\
--train_folder POLICY_TRAIN_FOLDER \\
--checkpoint EPOCH_NR | NONE \\
--aff_train_folder AFF_TRAIN_FOLDER
```

When the "--checkpoint" argument is not specified,  the evaluation will run for all epochs in POLICY_TRAIN_FOLDER.

The results are stored in POLICY_TRAIN_FOLDER the following way:
- If the evaluation ran with an affordance model: POLICY_TRAIN_FOLDER/evaluacion/Hulc_Aff/date/time.
- If the evaluation ran using only Hulc: POLICY_TRAIN_FOLDER/evaluation/Hulc/date/time

Here we show an example.
```
python slurm_eval.py -v thesis --hours 23 -j hulc_eval_all --dataset_path /work/dlclarge2/meeso-lfp/calvin_data/task_D_D --train_folder /home/borjadi/logs/lfp/2022-07-13/23-15-01_gcbc_50 --aff_train_folder /work/dlclarge2/borjadi-workspace/logs/thesis/aff_ablation/2022-06-20/14-31-59_aff_ablation
```

Optional flags:
 - --save_viz (flag to save images of rollouts)
 - --cameras=high_res (flag to change camera cfg to save images)
