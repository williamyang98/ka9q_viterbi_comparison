import argparse
import json
import numpy as np
from sample_loader import Sample, load_samples_from_json
from util import unique, get_si_scale

def main():
    parser = argparse.ArgumentParser(
        prog="tabulate_data",
        description="Tabulate benchmark data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("filename", help="Filename for benchmark data")
    args = parser.parse_args()

    with open(args.filename, "r") as fp:
        json_text = fp.read()
    json_data = json.loads(json_text)
    samples = load_samples_from_json(json_data)

    names = list(unique((s.name for s in samples)))
    kr_list = list(unique(((s.K,s.R) for s in samples)))

    print("## Update symbol rate")
    print("| K | R | {0} |".format(" | ".join(names)))
    print("| {0} |".format(" | ".join(["---"]*(len(names)+2))))
    for (K, R) in kr_list:
        kr_samples = {s.name: s for s in samples if (s.K == K and s.R == R)}
        values = []
        for name in names:
            if name in kr_samples:
                sample = kr_samples[name]
                symbol_rate = sample.total_output_symbols / (sample.update_ns*1e-9)
                avg = np.mean(symbol_rate)
                std = np.std(symbol_rate)
                prefix, scale = get_si_scale(avg)
                avg = avg/scale
                std = std/scale
                values.append(f"{avg:.3g}±{std:.2g}{prefix}")
            else:
                values.append("---")
        print("| {0} | {1} | {2} |".format(K, R, " | ".join(values)))

    print()
    print("## Chainback bit rate")
    print("| K | R | {0} |".format(" | ".join(names)))
    print("| {0} |".format(" | ".join(["---"]*(len(names)+2))))
    for (K, R) in kr_list:
        kr_samples = {s.name: s for s in samples if (s.K == K and s.R == R)}
        values = []
        for name in names:
            if name in kr_samples:
                sample = kr_samples[name]
                chainback_rate = sample.total_input_bytes*8 / (sample.chainback_ns*1e-9)
                avg = np.mean(chainback_rate)
                std = np.std(chainback_rate)
                prefix, scale = get_si_scale(avg)
                avg = avg/scale
                std = std/scale
                values.append(f"{avg:.3g}±{std:.2g}{prefix}")
            else:
                values.append("---")
        print("| {0} | {1} | {2} |".format(K, R, " | ".join(values)))

if __name__ == '__main__':
    main()
