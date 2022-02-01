import numpy as np
import os
import matplotlib.pyplot as plt
from time import sleep

RESULTS_DIR = "_results"

classifiers = [
    "COPK",
    "PCK",
    "COPK-S",
    "PCK-S",
]

db_names = set({})
scores_dict = dict({})
iteration_dict = dict({})


for f_name in os.listdir(RESULTS_DIR):
    db_name, params = f_name.split("_C")

    if db_name not in db_names:
        db_names.add(db_name)
        scores_dict[db_name] = {}
        iteration_dict[db_name] = {}

    const_r, type = params.split("_")
    const_r = float(const_r)
    type = type.split('.')[0]

    # print(db_name, const_r, type)

    scores = np.load(os.path.join(RESULTS_DIR, f_name))

    if 'iter' in type:
        iteration_dict[db_name][const_r] = scores
    else:
        scores_dict[db_name][const_r] = scores

for db_name in db_names:
    print(db_name)
    db_dict = scores_dict[db_name]

    bw = 8
    fig = plt.figure(figsize=(bw * 1.618, bw))
    grid = fig.add_gridspec(3, 1)

    plt.title(db_name)

    for i, cr in enumerate([0.001, 0.005, 0.01]):
        print(cr)
        scores = db_dict[cr]

        divider = np.arange(1, scores.shape[1]+1)
        cs = np.cumsum(scores, axis=1)
        cumsum_scores =  cs / divider[np.newaxis, :]

        # SCORES
        ax = fig.add_subplot(grid[i, :])
        ax.set_ylabel("Adjusted Rand Index")
        ax.set_xlabel("Chunks")

        for _ in scores:
            plt.plot(_, ':', lw=1)

        ax.set_prop_cycle(None)

        for clf, _ in zip(classifiers, cumsum_scores):
            plt.plot(_, lw=1.2, label=clf)

        ax.set_xlim(1, scores.shape[1]+1)
        ax.set_ylim(0.0, 1.0)
        ax.grid(ls=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.legend(frameon=False, ncol=4)
    plt.tight_layout()
    # plt.savefig("foo.png")
    plt.savefig(f"foo_{db_name}.png")
    plt.clf()
    plt.close()
