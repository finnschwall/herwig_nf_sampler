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
from FlowSampler import SingleChannelFlowSampler
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

def train(python_sampler, n_dims, channel_count, matrix_name, bin_number, bin_count):
    global sampler
    if bin_number == 0:
        logger.info(f"Starting training for {bin_count} diagrams for process {current_process}")
        process_info["start_time"] = time.time()
        process_info["bin_count"] = bin_count
        process_info["n_channels"] = channel_count
        process_info["channels"] = {}
    logger.info(f"Learning diagram {bin_number} of {bin_count}: {matrix_name}, PS Dim: {n_dims}, Channels: {channel_count}")

    start_time = time.time()
    ps_points = np.random.rand(int(settings.INITIAL_POINTS), n_dims)
    # n_dims = n_dims -1 #remove channel selection dimension
    cross_sections = python_sampler.dSigDRMatrix(ps_points)
    
    ps_sampling_time = time.time() - start_time

    integrator = SingleChannelFlowSampler(python_sampler.dSigDRMatrix,f"PythonSampler/{current_process}/{matrix_name}" ,n_dims, 1, matrix_name=matrix_name,
                                                current_process_name =current_process, single_channel=True)
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
    sampler = integrator
    # try:
    #     x, prob, func_vals = integrator.sample(10000, numpy=True)
    #     # func_vals/=10
    #     for i in range(len(x)):
    #         python_sampler.dSigDRRun(x[i], prob[i], func_vals[i])
    # except Exception as e:
    #     traceback.print_exc()
    #     raise e
    
g_python_sampler = None
def load(python_sampler, n_dims, channel_count, matrix_name, bin_number, bin_count):
    global sampler, g_python_sampler
    g_python_sampler = python_sampler
    try:
        integrator = SingleChannelFlowSampler(python_sampler.dSigDRMatrix,f"PythonSampler/{current_process}/{matrix_name}" ,n_dims, 1, matrix_name=matrix_name,
                                                current_process_name =current_process, single_channel=True)
        integrator.load()
        sampler = integrator
    except Exception as e:
        traceback.print_exc()
        raise e
    # print(integrator.integrate(2000))
    # print(integrator.integrate(2000))
    # print(integrator.integrate(2000))
    # from madnis.integrator import Integrator, Integrand
    # integrand = Integrand(integrator.model[0].matrix_callback, input_dim=n_dims)
    # mad_integ = Integrator(integrand, flow = integrator.model[0].model.to("cpu"))
    # sampler = mad_integ
    # result, error = mad_integ.integrate(100)
    # print(f"Integration result: {result:.5f} +- {error:.5f}")



stored_x = []
stored_prob = []
stored_func_vals = []
max_weight = 0
current_idx = 0
def generate():
    global sampler, stored_x, stored_prob, stored_func_vals, current_idx, max_weight
    if current_idx >= len(stored_x):
        x, prob, func_vals = sampler.sample(2000, numpy=True)
        weights = func_vals / prob
        max_weight = weights.max().item()
        # unweighted_indices = []
        # valid_indices = np.where(weights > 0)[0]
        # if len(valid_indices) > 0:
        #     for idx in valid_indices:
        #         # Accept with probability proportional to weight / max_weight
        #         if np.random.uniform(0, 1) < weights[idx] / max_weight:
        #             unweighted_indices.append(idx)
        # x = x[unweighted_indices]
        # prob = prob[unweighted_indices]
        # func_vals = func_vals[unweighted_indices]
        stored_x = x.tolist()
        stored_prob = prob.tolist()
        stored_func_vals = func_vals.tolist()
        current_idx = 0
        estimated_integral = sampler._integrate(func_vals, prob, 2000)
        logger.info(f"Sampling new points. Est. Integ.:{estimated_integral['integral']:.3f}, Max weight: {max_weight:.3f}")
        
        
    ret_tuple = (stored_x[current_idx], stored_prob[current_idx], stored_func_vals[current_idx], max_weight)
    current_idx += 1
    return ret_tuple
    
    try:
        x, prob, channel_weights = sampler.sample(1000)
        for i in range(len(x)):
            # print(f"Sampled point: {x[i]}, Probability: {prob[i]}, Channel weights: {channel_weights[i]}")
            g_python_sampler.dSigDRRun(x[i], prob[i], channel_weights[i])
    except Exception as e:
        traceback.print_exc()
        raise e
    return sampler.sample(n_samples)