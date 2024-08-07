import argparse
import json
import numpy
import matplotlib.pyplot as plt
from sample_loader import Sample, load_samples_from_json

def main():
    parser = argparse.ArgumentParser(
        prog="plot_data",
        description="Plot benchmark data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("filename", help="Filename for benchmark data")
    args = parser.parse_args()

    with open(args.filename, "r") as fp:
        json_text = fp.read()
    json_data = json.loads(json_text)
    samples = load_samples_from_json(json_data)

    print(samples)

if __name__ == '__main__':
    main()
