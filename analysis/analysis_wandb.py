from collections import defaultdict
import json
import os
import re
from typing import DefaultDict, Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import seaborn as sns
import wandb

plt.rc("text", usetex=True)

sns.set(style="white", font_scale=2)
plt.rcParams["font.size"] = 50


def plot_data(data, ax, label, color="gray", stats_axis=0):
    mean = np.mean(data, axis=stats_axis)[:, -1]
    min_values = np.min(data, axis=stats_axis)[:, -1]
    max_values = np.max(data, axis=stats_axis)[:, -1]

    smooth_window = 1
    mean = np.array(pd.Series(mean).rolling(smooth_window, min_periods=smooth_window).mean())
    min_values = np.array(pd.Series(min_values).rolling(smooth_window, min_periods=smooth_window).mean())
    max_values = np.array(pd.Series(max_values).rolling(smooth_window, min_periods=smooth_window).mean())

    steps = np.array(data)[0, :, 0].astype(int)
    ax.plot(steps, mean, label=label, color=color)
    # ax.fill_between(steps, max_values, min_values, color=color, alpha=0.10)
    return ax

# Data is a list
def plot_experiments(
    data,
    show=True,
    save=True,
    save_name="error",
    save_folder="./analysis/figures/",
    x_lim=None,
    x_label="epochs",
    y_label="error",
):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), sharey=True)
    # ax.set_title("Evaluation")

    cm = plt.get_cmap("rainbow")
    colors = cm(np.linspace(0, 1, len(data)))
    # colors = [[0, 0, 1, 1], [1, 0, 0, 0.8]]
    for experiment, c in zip(data, colors):
        name, exp_data = experiment
        ax = plot_data(exp_data, ax, label=name, color=c, stats_axis=0)

    ax.set_xlabel(x_label.title())
    ax.set_ylabel(y_label.title())
    ax.set_xlim(xmin=0, xmax=x_lim)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    ax.set_ylim(0)
    ax.legend(loc="upper right")

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if show:
        plt.show()
    if save:
        fig.savefig(os.path.join(save_folder, "%s.png" % save_name), bbox_inches="tight", pad_inches=0)


class WandbPlots:
    def __init__(
        self,
        experiments: Dict,
        track_metrics: List[str],
        load_from_file: bool = True,
        show: bool = True,
        save_dir: str = "./analysis/figures",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = os.path.abspath(save_dir)
        json_filepath = os.path.join(self.save_dir, "exp_data.json")
        self.metrics = [m.split("/")[-1] for m in track_metrics]
        print("searching data in %s" % json_filepath)
        if os.path.isfile(json_filepath) and load_from_file:
            print("File found, loading data from previous wandb fetch")
            with open(json_filepath, "r") as outfile:
                data = json.load(outfile)
        else:
            print("No file found, loading data wandb..")
            data = self.read_from_wandb(experiments, track_metrics)
        self.data = data
        self.plot_stats(show)

    def read_from_wandb(self, experiments, wandb_metrics):
        # Get the runs
        _api = wandb.Api()
        runs = {}
        for exp_name, exp_id in experiments.items():
            runs[exp_name] = []
            wandb_run = _api.run(exp_id)
            runs[exp_name] = wandb_run
        # Get the data from the runs
        data = {}
        for exp_name, run in runs.items():
            exp_data = defaultdict(list)
            run_data = defaultdict(list)  # metric: (timestep, episode, value)
            for row in run.scan_history():
                for metric in wandb_metrics:
                    if metric in row:
                        _row_data = [row["epoch"], row[metric]]
                        _metric = metric.split("/")[-1]
                        run_data[_metric].append(_row_data)
            # List of run data for each metric
            for metric in self.metrics:
                exp_data[metric].append(run_data[metric])
            data[exp_name] = exp_data

        output_path = os.path.join(self.save_dir, "exp_data.json")
        with open(output_path, "w") as outfile:
            json.dump(data, outfile, indent=2)
        return data

    def plot_stats(self, show):
        for metric in self.metrics:
            metric_data = {
                "data": [], "min_x_value": np.inf,
            }

            for exp_name, exp_data in self.data.items():
                truncated_data = self.truncate_data(exp_data[metric], min_data_axs=20)
                metric_data["data"].append([exp_name, truncated_data])

                # Update crop values
                if truncated_data[0][-1][0] < metric_data["min_x_value"]:
                    metric_data["min_x_value"] = truncated_data[0][-1][0]

            # Crop to experiment with least episodes
            x_lim = metric_data["min_x_value"]
            x_label = "epoch"
            save_name = "%s_by_%s" % (metric, x_label)
            plot_experiments(
                metric_data["data"],
                show=show,
                save=True,
                save_name=save_name,
                save_folder=self.save_dir,
                x_label=x_label,
                x_lim=x_lim,
                y_label=metric.replace("_", " ").capitalize(),
            )

    def truncate_data(self, data, min_data_axs=None):
        """
        data(list):
            contains as elements different run results
            of the same experiment
            len() = n_experiments
            - Each element is another list with columns:
                columns:[timesteps, episode, metric_value]
        """
        epoch_axis = 1
        if min_data_axs is None:
            min_data_axs = min([np.array(d)[:, 0][-1] for d in data])
        croped_data = data[:min_data_axs]
        return croped_data


if __name__ == "__main__":
    # Tabletop Rand
    runs_visual = {
        "x": "jessibd/aff_lang_thesis/hcmulu5z",
        "y": "jessibd/aff_lang_thesis/3npitgti",
    }

    # Generalization
    runs_lang = {
        "x": "jessibd/aff_lang_thesis/3npitgti",
        "y": "jessibd/aff_lang_thesis/3npitgti",
    }

    run_info = {"visual_enc_ablation": runs_visual,
                "lang_enc_abletion": runs_lang}

    metrics = ["Validation/mean_dist_error", "Validation/mean_depth_error"]
    for exp_name, runs in run_info.items():
        analysis = WandbPlots(
            runs, metrics, load_from_file=True, show=False, save_dir="./analysis/figures/%s" % exp_name
        )
