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
from MadnisFlow import MadnisMultiChannelIntegrator, MadnisFlow


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

def train(python_sampler, n_dims, channel_count, matrix_name, bin_number, bin_count):
    if bin_number == 0:
        logger.info(f"Starting training for {bin_count} diagrams for process {current_process}")
        process_info["start_time"] = time.time()
        process_info["bin_count"] = bin_count
        process_info["n_channels"] = channel_count
        process_info["channels"] = {}
    logger.info(f"Learning diagram {bin_number} of {bin_count}: {matrix_name}, PS Dim: {n_dims}, Channels: {channel_count}")

    start_time = time.time()
    ps_points = np.random.rand(int(settings.INITIAL_POINTS), n_dims)
    n_dims = n_dims -1 #remove channel selection dimension
    cross_sections = python_sampler.dSigDRMatrix(ps_points)
    total_cross_section = np.sum(cross_sections)
    ps_sampling_time = time.time() - start_time

    data_preprocessor = Dataset.ChannelDataPreprocessor(channel_count)
    channel_phase_space, channel_cross_sections = data_preprocessor.split_by_channel(ps_points, cross_sections, int(settings.CHANNEL_SELECTION_DIM))
    tot_cross_section_per_channel = np.array([np.sum(i) for i in channel_cross_sections])
    channel_weights = tot_cross_section_per_channel / total_cross_section
    logger.info(f"Channel weights: {channel_weights}")
    expected_weight = 1 / channel_count
    drop_threshold = float(settings.CHANNEL_DROP_THRESHOLD)*expected_weight
    if np.sum(channel_weights > drop_threshold) > 0:
        dropped_weights = channel_weights[channel_weights < drop_threshold]
        logger.info(f"Dropping channels {np.where(channel_weights < drop_threshold)[0]} with weight < {max(dropped_weights):.4f} (Exp. Weight: {expected_weight:.4f})")
        #recalculate to make sure weights sum to 1
        tot_cross_section_per_channel = np.where(channel_weights > drop_threshold, tot_cross_section_per_channel, 0)
        total_cross_section = np.sum(tot_cross_section_per_channel)
        channel_weights = tot_cross_section_per_channel / total_cross_section
        logger.info(f"New channel weights: {channel_weights}")
    process_info["channels"][matrix_name] = {}
    process_info["channels"][matrix_name]["channel_weights"] = channel_weights.tolist()
    process_info["channels"][matrix_name]["remaining_channel_count"] = int(np.sum(channel_weights > drop_threshold))
    process_info["channels"][matrix_name]["total_cross_section"] = total_cross_section
    process_info["channels"][matrix_name]["ps_sampling_time"] = ps_sampling_time
    process_info["channels"][matrix_name]["channel_cross_sections"] = tot_cross_section_per_channel.tolist()

    
    # flow_trainer.train_multi_channel(ps_points, cross_sections, )
    
    # data_preprocessor = Dataset.ChannelDataPreprocessor(channel_count)
    # datasets = data_preprocessor.get_datasets(ps_points, cross_sections, int(config["TRAINING_PARAMETERS"]["channel_selection_dim"]))

    
    flows = []
    last_trainer = None
    for i in tqdm(range(len(channel_phase_space))):
        if channel_weights[i] == 0:
             continue
        
        basepath = f"PythonSampler/{current_process}/{matrix_name}"
        os.makedirs(basepath, exist_ok=True)
        basepath = os.path.join(basepath, f"channel_{i}")
        os.makedirs(basepath, exist_ok=True)
        flow_trainer = MadnisFlow(channel_phase_space[i], channel_cross_sections[i], 
                                              python_sampler.dSigDRMatrix, basepath, n_dims, i,
                                                len(channel_phase_space),single_channel=True)
        logger.info(f"Training channel {i} of {channel_count}. # of points: {len(channel_phase_space[i])}, tot. cross section: {np.sum(channel_cross_sections[i]):.2e}")
        flow_trainer.train()
        flows.append(flow_trainer.model)
        last_trainer = flow_trainer
        
    multichannel_integrator = MadnisMultiChannelIntegrator(flows, last_trainer.matrix_callback, channel_weights)
    result = multichannel_integrator.integrate(sample_size=1000)
    print(result)

    exit()
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
    

def load(python_sampler, n_dims):
    python_sampler_instance.setup_base(python_sampler, n_dims)
    return python_sampler_instance.load()