#!/usr/bin/env python3
import os
import subprocess
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

folder = None
def run_herwig_command(folder, iterations=100, command="Herwig run LEP-Matchbox.run -N1000"):
    """
    Run the Herwig command a specified number of times in the given folder.
    
    Args:
        folder (str): Directory to execute commands in
        iterations (int): Number of times to run the command
        command (str): The Herwig command to execute
    """
    print(f"Changing to directory: {folder}")
    os.chdir(folder)
    
    print(f"Running '{command}' {iterations} times...")
    for i in range(iterations):
        print(f"Execution {i+1}/{iterations}")
        try:
            subprocess.run(command + f" -s {i}", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command: {e}")
            continue

def parse_dat_files(folder):
    """
    Find and parse all .dat files in the runstats folder.
    
    Args:
        folder (str): Base directory containing the runstats folder
        
    Returns:
        dict: Dictionary with filenames as keys and parsed data as values
    """
    runstats_dir = os.path.join(folder, "runstats")
    if not os.path.exists(runstats_dir):
        print(f"Error: runstats directory not found at {runstats_dir}")
        return {}
    
    all_data = {}
    dat_files = glob.glob(os.path.join(runstats_dir,"**" ,"*.dat"), recursive=True)
    
    if not dat_files:
        print("No .dat files found in runstats directory")
        return {}
    
    print(f"Found {len(dat_files)} .dat files in {runstats_dir}")
    
    for dat_file in dat_files:
        filename = os.path.basename(dat_file)
        print(f"Parsing {filename}")
        
        with open(dat_file, 'r') as f:
            content = f.read()
            
        json_strs = content.split("\n{")
        json_objects = []
        for json_str in json_strs:
            try:
                json_obj = json.loads(json_str)
                json_objects.append(json_obj)
            except json.JSONDecodeError as e:
                json_str = "{" + json_str
                try:
                    json_obj = json.loads(json_str)
                    json_objects.append(json_obj)
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON object: {e}")
                    print(f"Problematic JSON string: {json_str}")
                    continue
        
        all_data[filename] = json_objects
    
    return all_data

def create_histograms(data):
    global folder
    """
    Create histograms for each category in the data.
    
    Args:
        data (dict): Dictionary with filenames as keys and parsed data as values
    """
    for filename, json_objects in data.items():
        if not json_objects:
            print(f"No data to plot for {filename}")
            continue
            
        # Get all metrics except Timestamp
        metrics = [key for key in json_objects[0].keys() if key != "Timestamp"]
        
        # Calculate grid size (roughly square layout)
        n_plots = len(metrics)
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        
        # Create a large figure for all histograms
        plt.figure(figsize=(n_cols * 5, n_rows * 4))
        
        for i, metric in enumerate(metrics, 1):
            values = [obj.get(metric) for obj in json_objects if metric in obj]
            if not values:
                continue
                
            plt.subplot(n_rows, n_cols, i)
            plt.hist(values, bins='auto', alpha=0.7)
            plt.title(metric)
            
            # Add statistics
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                plt.annotate(f"Mean: {mean_val:.4g}\nStd: {std_val:.4g}", 
                             xy=(0.05, 0.95),
                             xycoords='axes fraction',
                             va='top',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save the figure with the same name as the file but with .png extension
        output_filename = os.path.splitext(filename)[0] + ".png"
        output_filename = os.path.join(folder,"runstats" ,output_filename)
        plt.savefig(output_filename)
        print(f"Saved histogram plot: {output_filename}")
        plt.close()

def main():
    global folder
    parser = argparse.ArgumentParser(description="Run Herwig commands and analyze results")
    parser.add_argument("--folder", type=str, default=".", 
                        help="Folder to execute commands in (default: current directory)")
    parser.add_argument("--iterations", "-n", type=int, default=20,
                        help="Number of times to run the Herwig command (default: 100)")
    parser.add_argument("--command", type=str, default="Herwig run LEP-Matchbox.run -N1000",
                        help="Herwig command to execute (default: 'Herwig run LEP-Matchbox.run -N1000')")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip running Herwig and only analyze existing .dat files")
    
    args = parser.parse_args()
    
    # Convert relative path to absolute
    folder = os.path.abspath(args.folder)
    print(f"Using folder: {folder}")
    
    if not args.analyze_only:
        run_herwig_command(folder, args.iterations, args.command)
    
    # Parse .dat files and create histograms
    data = parse_dat_files(folder)
    if data:
        create_histograms(data)
    else:
        print("No data to analyze. Exiting.")

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    print(f"Total execution time: {end_time - start_time}")