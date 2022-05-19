import shutil
from pathlib import Path
from typing import Any
import glob
import argparse
import os
import numpy as np
from tqdm import tqdm
from vr_env.utils.utils import to_relative_action


def get_file_list(data_dir, extension=".npz", sort_list=False):
    """retrieve a list of files inside a folder"""
    dir_path = Path(data_dir)
    dir_path = dir_path.expanduser()
    assert dir_path.is_dir(), f"{data_dir} is not a valid dir path"
    file_list = []
    for x in dir_path.iterdir():
        if x.is_file() and extension in x.suffix:
            file_list.append(x)
        elif x.is_dir():
            file_list.extend(get_file_list(x, extension))
    if sort_list:
        file_list = sorted(file_list, key=lambda file: file.name)
    return file_list

class FixDatasetTransitions:
    def __init__(self, cfg) -> None:
        self.src = Path(cfg.src).expanduser("~/")
        assert self.src.is_dir(), "The path of the src dataset must be a dir"

        self.dest = Path(cfg.dest).expanduser("~/")
        self.dest.mkdir(parents=True, exist_ok=True)

        ep_start_end_ids = np.load(self.src / "ep_start_end_ids.npy")
        self.ep_start_end_ids = ep_start_end_ids[ep_start_end_ids[:, 0].argsort()]
        self.set_step_to_file()
        
        # lang ann files
        self.lang_ann_files = self.get_auto_lang_ann()

    def get_auto_lang_ann(self):
        search_str = os.path.join(self.src.as_posix()) + "*/auto_lang_ann.npy"
        lang_ann_files = glob.glob(search_str)
        return lang_ann_files

    def set_step_to_file(self):
        """Create mapping from step to file index"""
        step_to_file = {}
        file_list = get_file_list(self.src)
        for file in file_list:
            step = int(file.stem.split("_")[-1])
            step_to_file[step] = file
        self.step_to_file = step_to_file

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        new_i = 0
        new_ep_lens = []
        new_ep_start_end_ids = []
        for start, end in tqdm(self.ep_start_end_ids):
            # Load following frames if they have not been loaded
            new_start = new_i
            filepath = self.step_to_file[start]
            current_frame = dict(np.load(filepath))
            for step in tqdm(range(start, end)):
                filepath = self.step_to_file[step + 1]
                next_frame = dict(np.load(filepath))
                # (obs, prev_action), (next_obs, action) -> (obs, action)
                new_frame = {}
                # Get obs from current frame
                for key, value in current_frame.items():
                    if "action" not in key:
                        new_frame[key] = value

                # Get action from next frame
                new_frame["actions"] = next_frame["actions"]
                new_frame["rel_actions"] = to_relative_action(new_frame["actions"], new_frame["robot_obs"])

                current_frame = next_frame
                np.savez(self.dest / f"episode_{new_i:07d}.npz", **new_frame)
                new_i += 1
            new_end = new_i - 1
            new_ep_len = new_end - new_start + 1
            new_ep_start_end_ids.append((new_start, new_end))
            new_ep_lens.append(new_ep_len)
        
        # Save new files
        np.save(self.dest / "ep_lens.npy", new_ep_lens)
        np.save(self.dest / "ep_start_end_ids.npy", new_ep_start_end_ids)
        if (self.src / "statistics.yaml").is_file():
            shutil.copy(self.src / "statistics.yaml", self.dest)
        if (self.src / ".hydra").is_dir():
            shutil.copytree(self.src / ".hydra", self.dest / ".hydra")

def main(cfg):
    parser = argparse.ArgumentParser(description="Parse parameters")
    parser.add_argument("--src", type=str)
    parser.add_argument("--dest", type=str)
    args = parser.parse_args()
    analysis_class = FixDatasetTransitions(args)
    analysis_class()


if __name__ == "__main__":
    main()