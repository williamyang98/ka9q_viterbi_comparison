import numpy as np

class Sample:
    def __init__(self, v):
        self.name = v["name"]
        self.K = v["K"]
        self.R = v["R"]
        self.poly = np.array(v["poly"])
        self.total_input_bytes = v["total_input_bytes"]
        self.total_transmit_bits = v["total_transmit_bits"]
        self.total_output_symbols = v["total_output_symbols"]
        self.sampling_time = v["sampling_time"]
        self.minimum_samples = v["minimum_samples"]
        self.total_samples = v["total_samples"]
        self.init_ns = np.array(v["init_ns"])
        self.update_ns = np.array(v["update_ns"])
        self.chainback_ns = np.array(v["chainback_ns"])
        self.total_bits = v["total_bits"]
        self.total_bit_errors = v["total_bit_errors"]
        self.bit_error_rate = v["bit_error_rate"]

def load_samples_from_json(json_data):
    return [Sample(v) for v in json_data]
