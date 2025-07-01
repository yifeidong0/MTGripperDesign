#!/usr/bin/env python
"""
Pretty summary plots (mean ± std) for several TensorBoard runs.

Changes vs. previous version
----------------------------
✓ Nicer colours     – matplotlib colormap 'Set2'.
✓ Larger legend     – fontsize=11, frameon=True.
✓ One figure        – 3 sub-plots laid out horizontally.
"""
import pathlib
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from tensorboard.backend.event_processing import event_accumulator

# ----------------------------------------------------------------------
ROOT   = pathlib.Path("data/runs/panda_1")           # adjust if needed
TAGS   = [
    "rollout/ep_rew_mean",
    "rollout/success_rate",
]
LABELS = {
    "rollout/ep_rew_mean":  "Episode reward",
    "rollout/success_rate": "Success rate",
}
LEGEND_ORDER = ["PPO_scratch", "PPO_finetune", "PPOReg_finetune"]
# COLOR_FOR_EXP = {}  # exp_name -> colour
# # Prettier colour cycle (Set2 has 8 harmonious pastels)
PALETTE = get_cmap("Set2").colors
# for idx, exp in enumerate(LEGEND_ORDER):
#     COLOR_FOR_EXP[exp] = PALETTE[idx % len(PALETTE)]
COLOR_FOR_EXP = {
    "PPO_scratch":      PALETTE[1],  # soft red-orange
    "PPO_finetune":     PALETTE[0],  # lime green
    "PPOReg_finetune":  PALETTE[2],  # teal-green
}
        
ALPHA  = 0.25                                         # transparency of std band
# ----------------------------------------------------------------------

def load_scalars(event_file, tags):
    ea = event_accumulator.EventAccumulator(str(event_file))
    ea.Reload()
    out = {}
    for tag in tags:
        if tag in ea.Tags()["scalars"]:
            evs   = ea.Scalars(tag)
            steps = np.fromiter((e.step  for e in evs), dtype=np.int64)
            vals  = np.fromiter((e.value for e in evs), dtype=np.float32)
            out[tag] = pd.DataFrame({"step": steps, "value": vals})
    return out

# 1. Gather raw per-file data -------------------------------------------------
raw = defaultdict(lambda: defaultdict(list))  # raw[exp][tag] -> list[DF]
for exp_dir in ROOT.iterdir():
    if not exp_dir.is_dir():
        continue
    for ev_file in exp_dir.glob("events.out.tfevents.*"):
        for tag, df in load_scalars(ev_file, TAGS).items():
            raw[exp_dir.name][tag].append(df)

# 2. Aggregate (align -> mean, std) ------------------------------------------
stats = defaultdict(dict)                            # stats[exp][tag] -> DF
for exp, tag_dict in raw.items():
    for tag, dfs in tag_dict.items():
        # union of all time steps
        union_steps = sorted(set(itertools.chain.from_iterable(df["step"] for df in dfs)))
        aligned = np.full((len(dfs), len(union_steps)), np.nan, dtype=np.float32)
        for i, df in enumerate(dfs):
            aligned[i, :] = (
                df.set_index("step")
                  .reindex(union_steps)["value"]
                  .to_numpy()
            )
        stats[exp][tag] = pd.DataFrame({
            "step": union_steps,
            "mean": np.nanmean(aligned, axis=0),
            "std" : np.nanstd(aligned,  axis=0),
        })


# 4. Plot: one figure, three sub-plots ---------------------------------------
fig, axes = plt.subplots(
    nrows=1, ncols=len(TAGS),
    figsize=(15, 4),          # wider canvas
    sharex=False
)

for col, tag in enumerate(TAGS):
    ax = axes[col]
    for exp, tag_dict in stats.items():
        if tag not in tag_dict:
            continue
        df = tag_dict[tag]
        colour = COLOR_FOR_EXP[exp]
        ax.plot(df["step"], df["mean"], label=exp, linewidth=2, color=colour)
        ax.fill_between(
            df["step"], df["mean"]-df["std"], df["mean"]+df["std"],
            alpha=ALPHA, color=colour
        )
    ax.set_title(LABELS.get(tag, tag), fontsize=22)
    ax.set_xlabel("Environment steps", fontsize=22)
    ax.tick_params(axis='both', labelsize=22)
    if tag == "rollout/ep_rew_mean":
        ax.set_ylim(-70, 0)  # adjust y-limits for better visibility
    ax.grid(True, alpha=0.3)

# # One legend for the whole figure, centred below sub-plots
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(
#     handles, labels,
#     loc="upper center", ncol=len(handles),
#     fontsize=11, frameon=True
# )

handles_labels = {
    exp: axes[0].plot([], [], label=exp, color=COLOR_FOR_EXP[exp])[0]
    for exp in LEGEND_ORDER if exp in COLOR_FOR_EXP
}
fig.legend(
    [handles_labels[exp] for exp in LEGEND_ORDER if exp in handles_labels],
    LEGEND_ORDER,
    loc="upper center",
    ncol=len(handles_labels),
    fontsize=24,
    frameon=True,
    bbox_to_anchor=(0.5, 1.3),   # <-- move legend higher (Y=1.15 is above top)
)

fig.tight_layout(rect=[0, 0.05, 1, 1])   # leave room bottom/top
plt.savefig("panda_runs_summary.png", bbox_inches='tight')
# plt.show()

# Optional: fig.savefig("panda_runs_summary.png", dpi=300)
