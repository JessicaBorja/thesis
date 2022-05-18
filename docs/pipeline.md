# Learning from play pipeline
## Process the real world data
Processing script to transform real world data into simulation format
- [process_real_data](../lfp/utils/preprocess_real_data.py)
```
python process_real_data.py --dataset_root SRC_DATA_PATH --output_dir DEST_DATA_PATH
```

## Convert to 15 hz
Render teleop data in 15 hz with render_low_freq
- [render_low_freq](../lfp/utils/render_low_freq.py)

```
python render_low_freq.py --dataset_root SRC_DATA_PATH --output_dir DEST_DATA_PATH
```

## Create dataset split
Generate a data split on split.json and compute statistics.yaml:
- [split_dataset.py](../lfp/utils/split_dataset.py)

```
 python split_dataset.py --dataset_root FULL_DATA_PATH
```

If we modify the data split manually, then we need to re-compute statistics.yaml

## Generate auto_lang_ann (LangAnnotationApp):
- [get_annotations.py](../../LanguageAnnotationApp/scripts/get_annotations.py)

```
 python get_annotations.py dataset_dir=FULL_DATASET_DIR database_path=FULL_DATABASE_PATH
```

FULL_DATASET_DIR: Absolute path of directory where the preprocessed dataset is (output of process_real_data). This is, original data before reducing the frequency
FULL_DATABASE_PATH: Absolute path of direction where database storing annotations for dataset in FULL_DATASET_DIR is.

This script will produce multiple ouputs inside a folder named [annotations](../../LanguageAnnotationApp/annotations/) for the different types of dataset:
    - original 30hz in [annotations](../../LanguageAnnotationApp/annotations/)
    - reduced15hz in [15hz](../../LanguageAnnotationApp/annotations/15hz)
    - repeated15hz in [15hz_repeated](../../LanguageAnnotationApp/annotations/15hz_repeated/)

Copy the language annotation folder to the corresponding dataset.

## Train learning from play
- [training.py](../lfp/training.py)

# Dataset folder content
Dataset should contain the following files/folders:
- statistics.yaml
- split.json
- ep_start_end_ids.npy
- .hydra/

If additionally using language:
- lang_[LANG_MODEL]/auto_lang_ann.npy


#  Affordance model and depth prediction
## Label the affordances
Get affordance, depth and pose labels (affordance)
data_labeler_lang.py

Get normalization values (thesis)
[find_norm_values.py](../scripts/find_norm_values.py)

Train depth estimation model (thesis)
[train_depth.py](../train_depth.py)

Train the affordance model (thesis)
[train_affordance.py](../train_affordance.py)