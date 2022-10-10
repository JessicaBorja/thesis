import json
import os
import hydra
import argparse

def get_abspath(path_str):
    path_str = os.path.expanduser(path_str)
    if not os.path.isabs(path_str):
        hydra_cfg = hydra.utils.HydraConfig().cfg
        if hydra_cfg is not None:
            cwd = hydra.utils.get_original_cwd()
        else:
            cwd = os.getcwd()
        path_str = os.path.join(cwd, path_str)
        path_str = os.path.abspath(path_str)
    return path_str

def main(json_file):
    with open(json_file) as f:
        data = json.load(f)
    best_model = max(data, key=lambda v: data[v]['avg_seq_len'])
    print(best_model)
    print(data[best_model]['avg_seq_len'])
    print(data[best_model]['chain_sr'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str)
    
    args = parser.parse_args()

    json_file = get_abspath(args.file)
    main(json_file)
