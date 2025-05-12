import json
import sys
# fix paths to libraries. issue with pybind I guess
sys.path.insert(0, '/home/finn/.pyenv/versions/3.10.16/envs/madnis/lib/python3.10/site-packages')
sys.path.append('/mnt/data-slow/herwig/python/madnis')

#import cpp module provided by pybind11
import herwig_python

import Dataset

import time
import numpy as np
import os
import glob
import re
import logging
from tqdm import tqdm
import settings
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

samplers = {}
reference_weights = {}
current_active_matrix_element = ""
g_python_sampler = None

def train(python_sampler, n_dims, channel_count, matrix_name, bin_number, bin_count, channel_selection_dim):
    global samplers, current_active_matrix_element, reference_weights, g_python_sampler
    g_python_sampler = python_sampler
    if bin_number == 0:
        logger.info(f"Starting training for {bin_count} diagrams for process {current_process}")
        process_info["start_time"] = time.time()
        process_info["bin_count"] = bin_count
        process_info["n_channels"] = channel_count
        process_info["channels"] = {}
    logger.info(f"Learning diagram {bin_number} of {bin_count}: {matrix_name}, PS Dim: {n_dims}, Channels: {channel_count}")
    current_active_matrix_element = matrix_name
    start_time = time.time()

    do_train = not os.path.exists(f"PythonSampler/{current_process}/{matrix_name}/best_model.pth") or settings.ALWAYS_RETRAIN
    if do_train:
        ps_points = np.random.rand(int(settings.INITIAL_POINTS), n_dims)
        cross_sections = python_sampler.dSigDRMatrix(ps_points)
        reference_weights[matrix_name] = max(cross_sections)
    else:
        reference_weights[matrix_name] = 1.0
    ps_sampling_time = time.time() - start_time

    integrator = FlowSampler(python_sampler.dSigDRMatrix,f"PythonSampler/{current_process}/{matrix_name}" ,n_dims, channel_count, matrix_name=matrix_name,
                                                current_process_name =current_process, single_channel=not settings.SPLIT_BY_CHANNELS
                                                ,channel_selection_dim=channel_selection_dim)
    if not do_train:
        logger.info(f"Loading existing model for {matrix_name}")
        integrator.load()
        
        
    else:
        integrator.prepare_data(ps_points, cross_sections)
        integrator.train(verbose=True)
        integrator.save()
        process_info[matrix_name] = integrator.meta
        if bin_number == bin_count - 1:
            process_info["end_time"] = time.time()
            process_info["total_time"] = process_info["end_time"] - process_info["start_time"]
            logger.info(f"Training finished for {bin_count} diagrams for process {current_process}")
            logger.info(f"Total time: {process_info['total_time']:.2f} seconds")
            # sum up total remaining channels
            total_remaining_channels = 0
            for matrix_name, matrix_info in process_info["channels"].items():
                total_remaining_channels += matrix_info["remaining_channel_count"]
            logger.info(f"Total trained flows: {total_remaining_channels}")
        with open(f"PythonSampler/{current_process}/process_info.json", "w") as f: 
            json.dump(process_info, f, indent=4)

    
    samplers[matrix_name] = integrator

    

def load(python_sampler, n_dims, channel_count, matrix_name, bin_number, bin_count, channel_selection_dim, ref_weight):
    global samplers, reference_weights, current_active_matrix_element, g_python_sampler
    g_python_sampler = python_sampler
    current_active_matrix_element = matrix_name
    if matrix_name in samplers:
        logger.info(f"Diagram {matrix_name} already loaded for process {current_process}")
        return

    reference_weights[matrix_name] = ref_weight
    print(f"Loading diagram {matrix_name} for process {current_process}")
    try:
        integrator = FlowSampler(python_sampler.dSigDRMatrix,f"PythonSampler/{current_process}/{matrix_name}" ,n_dims, channel_count, matrix_name=matrix_name,
                                                current_process_name =current_process, single_channel=not settings.SPLIT_BY_CHANNELS,
                                                channel_selection_dim=channel_selection_dim)
        integrator.load()
        samplers[matrix_name] = integrator
    except Exception as e:
        traceback.print_exc()
        raise e



stored_x = []
stored_prob = []
stored_func_vals = []
max_weight = 0
current_idx = 0
def generate():
    global samplers, stored_x, stored_prob, stored_func_vals, current_idx, max_weight, reference_weights, current_active_matrix_element, g_python_sampler

    sampler = samplers[current_active_matrix_element]
    reference_weight = reference_weights[current_active_matrix_element]
    if current_idx >= len(stored_x):
        n_cache = 50000
        x, prob, func_vals = sampler.sample(n_cache, numpy=True)
        weights = func_vals / prob
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 5 * 2))
        # axes = axes.flatten()
        # axes[0].hist(prob, bins=50)
        # axes[0].set_title("Probability distribution")
        # axes[1].hist(func_vals, bins=50)
        # axes[1].set_title("Function values distribution")
        # axes[2].hist(weights, bins=50)
        # axes[2].set_title("Weights distribution")
        # plt.savefig(sampler.basepath + "/weights.png")
        # plt.close(fig)

        zero_func_vals = np.where(func_vals == 0)[0]
        zero_weights = np.where(weights == 0)[0]
        # max_weight = weights.max().item()
        max_weight = reference_weight


        nonzero_weights = weights[weights > 0]
        unweighting_efficiency = weights.mean()/reference_weight*100

        est_accepted_points = nonzero_weights.sum()/(reference_weight*0.5)/n_cache*100*(float(len(zero_func_vals))/n_cache)


        stored_x = x.tolist()
        stored_prob = prob.tolist()
        stored_func_vals = func_vals.tolist()
        current_idx = 0
        estimated_integral = sampler._integrate(func_vals, prob, n_cache)
        logger.info(f"Caching. Est. Integ.:{estimated_integral['integral']:.3f}+-{estimated_integral['error']:.5f}\n"
                    f"Ref. weight: {max_weight:.3f}, Median {np.median(weights):.3f}, Mean: {np.mean(weights):.3f}\n"
                    f"Zero: {float(len(zero_func_vals))/n_cache*100:.2f}%, Est. Acc: {est_accepted_points:.3f}%, Unweighting efficiency: {unweighting_efficiency:.3f}")
        
        
    ret_tuple = (stored_x[current_idx], stored_prob[current_idx], stored_func_vals[current_idx], max_weight)
    current_idx += 1
    return ret_tuple
