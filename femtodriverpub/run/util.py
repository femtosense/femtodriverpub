import numpy as np
from copy import copy
from typing import Dict, Tuple, Union, List


def process_single_outputs(
    outputs: Dict[str, Dict],
    no_save: bool = False,
    no_plot: bool = False,
    model_name: str = "",
) -> List[str]:
    """if the model has a single output, save each runner's to file
    do some other summary analysis
    returns fnames of output files
    """

    import matplotlib.pyplot as plt

    # bail out if multiple output variables
    for k, v in outputs.items():
        if len(v) > 1:
            return None

    # split out output files for easy handoff to others
    fnames = []
    output_var = None
    for k, v in outputs.items():
        assert len(v) == 1
        vals = next(iter(v.values()))
        output_var = next(iter(v.keys()))
        fname = f"output_{k}.npy"
        fnames.append(fname)
        if not no_save:
            np.save(fname, vals)

    # no comparisons to do
    if len(outputs) == 1:
        return None, None

    # make diff histogram (over entire run)
    diffs = {}
    idxs = {}
    maxdiff = 1
    to_del = []
    for idx_a, runner_a in enumerate(outputs):
        for idx_b, runner_b in enumerate(outputs):
            out_a = outputs[runner_a][output_var]
            out_b = outputs[runner_b][output_var]
            ax_idx = (idx_a, idx_b)
            idxs[(runner_a, runner_b)] = ax_idx
            if runner_a != runner_b and (runner_b, runner_a) not in diffs:
                diff = out_a - out_b
                diffs[(runner_a, runner_b)] = diff
                maxdiff = max(maxdiff, np.abs(np.max(diff.flatten())))
            else:
                to_del.append((idx_a, idx_b))

    if no_plot:
        return fnames, diffs

    fig, axes = plt.subplots(len(outputs), len(outputs), figsize=(10, 10))
    for comparison, diff in diffs.items():
        plt_idx = idxs[comparison]
        ax = axes[plt_idx]
        ax.hist(diff.flatten(), bins=50)
        a, b = comparison
        if np.max(diff.flatten()) == 0:
            ax.set_title(f"{a} - {b}\n(PERFECT MATCH)")
            pass
        else:
            ax.set_xlim(-maxdiff, maxdiff)
            ax.set_title(f"{a} - {b}")

    for ax_idx in to_del:
        fig.delaxes(axes[ax_idx])

    plt.suptitle(f"{model_name} output diff distributions for each runner pair")
    plt.tight_layout()
    plt.savefig("output_hist.png")

    # make max error/step line plot

    # only plot vs the lowest level platform, not all-to-all
    LEVELS = {
        "hardware": 1,
        "fasmir": 2,
        "fmir": 3,
        "fqir": 4,
        "dummy": 5,
    }

    def complt(comp_a, comp_b):
        return LEVELS[comp_a] < LEVELS[comp_b]

    def compeq(comp_a, comp_b):
        return LEVELS[comp_a] == LEVELS[comp_b]

    # determine lowest level platform
    lowest = "fqir"
    for platform in outputs:
        if complt(platform, lowest):
            lowest = platform

    def _mean_or_max_diff_over_time(ax, mean_or_max):
        if mean_or_max == "mean":
            fn = np.mean
        elif mean_or_max == "max":
            fn = np.max

        for comparison, diff in diffs.items():
            comp_a, comp_b = comparison
            if comp_a == lowest:
                max_diff_by_frame = fn(np.abs(diff), axis=1)
                ax.plot(np.arange(len(max_diff_by_frame)), max_diff_by_frame)
        ax.legend(diffs.keys())
        ax.set_title(f"{mean_or_max} differences per frame for different comparisons")

    fig, axes = plt.subplots(2, 1, figsize=(20, 7))
    _mean_or_max_diff_over_time(axes[0], "mean")
    _mean_or_max_diff_over_time(axes[1], "max")
    plt.savefig(f"output_diff_over_time.png")

    return fnames, diffs
