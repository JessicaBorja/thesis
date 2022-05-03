import json
import os

from thesis.utils.utils import get_abspath


def main(json_file):
    with open(json_file) as f:
        data = json.load(f)
    best_model = max(data, key=lambda v: data[v]['avg_seq_len'])
    print(best_model)
    print(data[best_model]['avg_seq_len'])
    print(data[best_model]['chain_sr'])

if __name__ == "__main__":
    json_file = get_abspath("./hydra_outputs/results.json")
    main(json_file)
