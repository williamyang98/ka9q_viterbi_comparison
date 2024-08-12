import argparse
import json
import numpy as np
from sample_loader import Sample, load_samples_from_json
from util import unique, get_si_scale

def main():
    parser = argparse.ArgumentParser(
        prog="plot_data",
        description="Plot benchmark data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("filename", nargs="?", default="./data/benchmark.json", help="Filename for benchmark data")
    parser.add_argument("--cpu-name", default=None, type=str, help="Name of cpu results came from")
    args = parser.parse_args()

    with open(args.filename, "r") as fp:
        json_text = fp.read()
    json_data = json.loads(json_text)
    samples = load_samples_from_json(json_data)

    names = list(unique((s.name for s in samples)))
    kr_list = list(unique(((s.K,s.R) for s in samples)))
    norm_name = "sse_u8"
    sorted_names = ["ka9q", "spiral", "sse_u16", "avx_u16", "sse_u8", "avx_u8"]
    sorted_names.extend((name for name in names if name not in set(sorted_names)))

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mpl_ticker

    column_width = 0.75
    bar_width = column_width/len(sorted_names)

    plt.style.use('ggplot')
    figsize = (10,5)
    fig, ax = plt.subplots(figsize=figsize)
    kr_norms = {}
    for (K,R) in kr_list:
        sample = next(s for s in samples if s.K == K and s.R == R and s.name == norm_name)
        kr_norms[(K,R)] = np.mean(sample.update_ns)
    for name_index, name in enumerate(sorted_names):
        kr_samples = {(s.K,s.R): s for s in samples if s.name == name}
        x_offset = name_index*bar_width
        x = []
        y_avg = []
        y_std = []
        for column_index, key in enumerate(kr_list):
            if not key in kr_samples:
                continue
            sample = kr_samples[key]
            value_norm = kr_norms[key]
            K, R = key
            x.append(column_index + x_offset - column_width/2 + bar_width/2)
            values = value_norm / sample.update_ns
            y_avg.append(np.mean(values))
            y_std.append(np.std(values))
        ax.bar(x, y_avg, bar_width, label=name, edgecolor="black")
        ax.errorbar(x, y_avg, y_std, fmt="none", linewidth=0.5, capsize=3, color="black")
    ax.set_ylim([0,2])
    ax.set_xticks(np.arange(len(kr_list)))
    ax.set_xticklabels([f"K={K} R={R}" for K,R in kr_list])
    ax.legend(loc="upper right")
    ax.set_ylabel("Relative symbol update rate")
    ax.yaxis.set_major_locator(mpl_ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mpl_ticker.MultipleLocator(0.1))
    plt.grid(which="minor", axis="y", color="black", linestyle="--", linewidth=0.5)
    plt.grid(which="major", axis="y", color="black", linestyle="-", linewidth=0.5)
    title = "Performance of Viterbi decoder symbol update"
    if args.cpu_name:
        title += f" ({args.cpu_name})"
    plt.suptitle(title, x=0.125, ha="left")
    plt.title(f"Normalised to {norm_name}", fontsize=10, loc="left")
    plt.show()

    fig, ax = plt.subplots(figsize=figsize)
    kr_norms = {}
    for (K,R) in kr_list:
        sample = next(s for s in samples if s.K == K and s.R == R and s.name == norm_name)
        kr_norms[(K,R)] = np.mean(sample.chainback_ns)
    for name_index, name in enumerate(sorted_names):
        kr_samples = {(s.K,s.R): s for s in samples if s.name == name}
        x_offset = name_index*bar_width
        x = []
        y_avg = []
        y_std = []
        for column_index, key in enumerate(kr_list):
            if not key in kr_samples:
                continue
            sample = kr_samples[key]
            value_norm = kr_norms[key]
            K, R = key
            x.append(column_index + x_offset - column_width/2 + bar_width/2)
            values = value_norm / sample.chainback_ns
            y_avg.append(np.mean(values))
            y_std.append(np.std(values))
        ax.bar(x, y_avg, bar_width, label=name, edgecolor="black")
        ax.errorbar(x, y_avg, y_std, fmt="none", linewidth=0.5, capsize=3, color="black")
    ax.set_ylim([0,2])
    ax.set_xticks(np.arange(len(kr_list)))
    ax.set_xticklabels([f"K={K} R={R}" for K,R in kr_list])
    ax.legend(loc="upper right")
    ax.set_ylabel("Relative chainback bit rate")
    ax.yaxis.set_major_locator(mpl_ticker.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(mpl_ticker.MultipleLocator(0.1))
    plt.grid(which="minor", axis="y", color="black", linestyle="--", linewidth=0.5)
    plt.grid(which="major", axis="y", color="black", linestyle="-", linewidth=0.5)
    title = "Performance of Viterbi decoder chainback"
    if args.cpu_name:
        title += f" ({args.cpu_name})"
    plt.suptitle(title, x=0.125, ha="left")
    plt.title(f"Normalised to {norm_name}", fontsize=10, loc="left")
    plt.show()

if __name__ == '__main__':
    main()
