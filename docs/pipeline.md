# Learning from play
## Process the real world data
Processing script to transform real world data into simulation format

## Create dataset split
Generate a data split on split.json and compute statistics.yaml (lfp):
- split_dataset.py

## Generate auto_lang_ann (LangAnnotationApp):
- get_annotations.py

copy the corresponding language annotation folder to the desired dataset.
If we modify manually the data split, re-compute statistics.yaml

# Dataset
Dataset should contain the following files:
- statistics.yaml
- split.json
- ep_start_end_ids.npy
- lang_[LANG_MODEL]/auto_lang_ann.npy
- .hydra/*

## Train affordance model and depth estimation

## Train learning from play

# Affordance learning
## Label the affordances
Get affordance, depth and pose labels (affordance)
- data_labeler_lang.py

Get normalization values (thesis)
- find_norm_values.py

Train depth estimation model (thesis)
- train_depth.py

Train the affordance model (thesis)
- train_affordance.py