import os
from typing import Dict
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import glob
import seaborn as sns
import re
import pathlib

from tomlkit import string

plt.rc("text", usetex=True)
sns.set(style="white", font_scale=2)
plt.rcParams["font.size"] = 40

class SheetPlots:
    def __init__(
        self,
        experiment: string,
        data: pd.DataFrame,
        show: bool = True,
        save_dir: str = "./analysis/figures",
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = os.path.abspath(save_dir)
        self.name = exp_name
        self.df = data
        self.plot_stats(show)

    def plot_stats(self, show):
        columns = [c for c in df.columns if "task" in c]
        x = np.arange(len(columns))

        percentages = np.array([100, 50, 25])
        # cm = plt.get_cmap("jet")
        # colors = cm(np.linspace(0, 1, len(percentages)))
        colors = ['#0b5394', '#cc0000',  '#34a853']
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), sharey=True)

        for i, (_, row) in enumerate(self.df.iterrows()):
            s = row["Method"]
            doted = "Baseline" in s
            percentage = int(s[s.find("(")+1:s.find("%)")])
            c = colors[np.where( percentage == percentages)[0].item()]
            linestyle = '--' if doted else '-'
            ax.plot(x, row[columns], linewidth=3, markersize=14,
                    linestyle=linestyle,
                    marker='o',
                    label=s.replace("%", "\%"),
                    color=c)
            if i == 0:
                for i, v in enumerate(row[columns]):
                    ax.text(i, v+0.05, "%0.2f" % v, ha="center", color=c, fontsize=20)
        
        ax.set_xticks(x)
        ax.set_xticklabels(columns)
        x_label = "Sequential tasks"
        y_label = "Success rate"
        ax.set_xlabel(x_label.title())
        ax.set_ylabel(y_label.title())
        ax.set_xlim(xmin=-0.5, xmax=len(columns))
        ax.set_ylim(0, 1)
        ax.grid(axis='y')
        ax.legend(loc="upper right", prop={'size': 20})
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if show:
            plt.show()
        fig.savefig(os.path.join(self.save_dir, "%s_lineplot.png" % self.name), bbox_inches="tight", pad_inches=0)

if __name__ == "__main__":
    data_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = os.path.join(data_dir, "data")
    results_file = glob.glob(data_dir + "/*.csv")[0]
    df = pd.read_csv(results_file)
    data_top = df.head() 

    exp_rows = list(np.where(df.isna().any(axis=1))[0])
    exp_names = list(df['Method'][exp_rows])

    exp_rows.append(None)
    experiments = {name: df[exp_rows[i] + 1: exp_rows[i + 1]] 
                   for i, name in enumerate(exp_names)}
    print(experiments)
    for exp_name, data in experiments.items():
        analysis = SheetPlots(
            exp_name, data, show=False, save_dir="./analysis/figures"
        )
