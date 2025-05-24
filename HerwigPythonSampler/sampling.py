import json
import sys
# fix paths to libraries. issue with pybind I guess
sys.path.insert(0, '/mnt/data-slow/Herwig/venv_herwig/lib/python3.10/site-packages')
# sys.path.append('/mnt/data-slow/herwig/python/madnis')

#import cpp module provided by pybind11
import herwig_python

import Dataset
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import glob
import re
import logging
from tqdm import tqdm
import settings, Sampler
from FlowSampler import FlowSampler
import traceback

in_files = glob.glob('*.in')
current_process = "UNKNOWN"
if not in_files:
    print("No .in files found in the current directory.")
else:
    most_recent_file = max(in_files, key=os.path.getmtime)
    pattern = r'do Factory:Process (.+)'
    with open(most_recent_file, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                current_process = match.group(1)

os.makedirs("PythonSampler", exist_ok=True)
os.makedirs(f"PythonSampler/{current_process}", exist_ok=True)

log_file = f"PythonSampler/{current_process}/log.txt"
logger = logging.getLogger('main')
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()
file_handler.setLevel(logging.INFO)
console_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)
formatter = logging.Formatter('%(asctime)s-%(levelname)s: %(message)s', datefmt='%H:%M:%S')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.setLevel(logging.INFO)


process_info = {}
try:
    with open(f"PythonSampler/{current_process}/process_info.json", "r") as f:
        process_info = json.load(f)
except FileNotFoundError:
    pass

sampler = None

def train(python_sampler, n_dims, channel_count, matrix_name, bin_number, bin_count, channel_selection_dim):
    global sampler
    try:
        if sampler is None:
            sampler = Sampler.Sampler(python_sampler, n_dims, channel_count, bin_count, channel_selection_dim, current_process)
        sampler.train(bin_number, matrix_name)
    except Exception as e:
        # traceback.print_exc()
        raise e

    

def load(python_sampler, n_dims, channel_count, matrix_name, bin_number, bin_count, channel_selection_dim, ref_weight):
    global sampler
    try:
        if sampler is None:
            sampler = Sampler.Sampler(python_sampler, n_dims, channel_count, bin_count, channel_selection_dim, current_process)
        sampler.load(matrix_name, ref_weight)
    except Exception as e:
        # traceback.print_exc()
        raise e



def generate(n_cache):
    global sampler
    try:
        if sampler is None:
            raise ValueError("Sampler not initialized?!")
        return sampler.generate(n_cache)
    except Exception as e:
        # traceback.print_exc()
        raise e


def finalize():
    global sampler
    try:
        if sampler is None:
            raise ValueError("Sampler not initialized?!")
        sampler.finalize()
    except Exception as e:
        # traceback.print_exc()
        raise e

# def generate():
#     global samplers, stored_x, stored_prob, stored_func_vals, current_idx, max_weight, reference_weights, current_active_matrix_element, g_python_sampler

#     sampler = samplers[current_active_matrix_element]
#     reference_weight = reference_weights[current_active_matrix_element]

#     # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 5 * 2))
#     # axes = axes.flatten()
#     # axes[0].set_title("NF Probability distribution")
#     # axes[1].set_title("ME cross section distribution")
#     # axes[2].set_title("Weights distribution")

#     if current_idx >= len(stored_x):
#         n_cache = 50000
#         x, prob, func_vals = sampler.sample(n_cache, numpy=True)
#         weights = func_vals / prob

#         # axes[0].hist(prob, bins=50)
#         # axes[1].hist(func_vals, bins=50)
#         # axes[2].hist(weights, bins=50)
#         # plt.savefig(sampler.basepath + "/weights.png")
#         # plt.close(fig)


#         zero_func_vals = np.where(func_vals == 0)[0]
#         zero_weights = np.where(weights == 0)[0]
#         # max_weight = weights.max().item()
#         max_weight = reference_weight


#         nonzero_weights = weights[weights > 0]
#         unweighting_efficiency = weights.mean()/reference_weight*100

#         est_accepted_points = nonzero_weights.sum()/(reference_weight*0.5)/n_cache*100*(float(len(zero_func_vals))/n_cache)


#         stored_x = x.tolist()
#         stored_prob = prob.tolist()
#         stored_func_vals = func_vals.tolist()
#         current_idx = 0
#         estimated_integral = sampler._integrate(func_vals, prob, n_cache)
#         logger.info(f"Caching. Est. Integ.:{estimated_integral['integral']:.3f}+-{estimated_integral['error']:.5f}\n"
#                     f"Ref. weight: {max_weight:.3f}, Median {np.median(weights):.3f}, Mean: {np.mean(weights):.3f}\n"
#                     f"Zero: {float(len(zero_func_vals))/n_cache*100:.2f}%, Est. Acc: {est_accepted_points:.3f}%, Unweighting efficiency: {unweighting_efficiency:.3f}")
        
        
#     ret_tuple = (stored_x[current_idx], stored_prob[current_idx], stored_func_vals[current_idx], max_weight)
#     current_idx += 1
#     return ret_tuple


