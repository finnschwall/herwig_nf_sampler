import time
import settings
import numpy as np
import os
import json
import traceback
import logging
import matplotlib.pyplot as plt
import torch
from FlowSamplerMadnis import FlowSampler

logger = logging.getLogger('main')

def visualize_metrics(data, figsize=(15, 12)):
    metrics_config = [
        ('integral', 'integral_error', 'Integral'),
        ('error', 'error_error', 'Error'),
        ('effective_sample_size', 'effective_sample_size_error', 'Effective Sample Size'),
        ('unweighting_efficiency', 'unweighting_efficiency_error', 'Unweighting Efficiency'),
        ('weight_mean', 'weight_mean_error', 'Weight Mean'),
        ('max_weight', 'max_weight_error', 'Max Weight'),
        ('loss', None, 'Loss')  # No error available for loss
    ]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    axes = axes.flatten()
    
    # Iteration indices
    iterations = range(len(data))
    
    # Plot each metric
    for i, (metric_key, error_key, title) in enumerate(metrics_config):
        ax = axes[i]
        
        # Extract values
        values = [float(d[metric_key]) for d in data]
        
        if error_key and error_key in data[0]:
            # Plot with error bars
            errors = [float(d[error_key]) for d in data]
            ax.errorbar(iterations, values, yerr=errors, 
                       marker='o', capsize=5, capthick=2, 
                       linewidth=2, markersize=6)
        else:
            # Plot without error bars
            ax.plot(iterations, values, marker='o', linewidth=2, markersize=6)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis for scientific notation if values are very small
        if max(values) < 0.01:
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Remove unused subplots
    for i in range(len(metrics_config), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.suptitle('Metrics Evolution Across Iterations', fontsize=16, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.93)
    plt.savefig('metrics_evolution.png')
    plt.close(fig)
    return fig, axes

class Sampler:
    def __init__(self, python_sampler, n_dims, channel_count, bin_count, channel_selection_dim, current_process):
        self.python_sampler = python_sampler
        self.n_dims = n_dims
        self.channel_count = channel_count
        self.bin_count = bin_count
        self.current_process = current_process

        self.cur_bin_number = -1
        self.channel_selection_dim = channel_selection_dim
        self.samplers = {}
        self.reference_weight = None
        self.current_idx = 0
        self.process_info = {}
        self.current_active_matrix_element = "UNKNOWN"
        self.reference_weights = {}
        self.tot_points = 0
        self.accepted_points = 0

        self.zero_count = 0

        self.axes = None
        self.fig=None

        self.n_train_step = 0
        self.basepath = f"PythonSampler/{self.current_process}"

        self.metrics = []

        self.ps_points = None
        self.cross_sections = None

    def train_step(self, integrator):
        n_cache = settings.PER_STAGE_POINTS
        temp_basepath = self.basepath + f"/step_{self.n_train_step}"
        integrator.basepath = temp_basepath
        if not os.path.exists(temp_basepath):
            os.makedirs(temp_basepath)
        if self.n_train_step==0:
            self.ps_points = np.random.rand(n_cache, self.n_dims)
            self.cross_sections = self.python_sampler.dSigDRMatrix(self.ps_points)
        else:
            weights, x, prob, func_vals, n_cache = integrator.get_weights(n_cache)
            self.ps_points = np.append(self.ps_points[::settings.SKIP_PARAM], x, axis=0)
            self.cross_sections = np.append(self.cross_sections[::settings.SKIP_PARAM], func_vals, axis=0)
            # ps_points = x
            # cross_sections = func_vals#self.python_sampler.dSigDRMatrix(x)
            plt.hist(weights, bins=1000)#, density=True)
            plt.yscale('log')
            plt.savefig(temp_basepath + "/weights.png")
            plt.close()
            plt.cla()
            plt.clf()
            plt.hist(func_vals, bins=1000)#, density=True)
            plt.yscale('log')
            plt.savefig(temp_basepath + "/func_vals.png")
            plt.close()
            plt.cla()
            plt.clf()
            plt.hist(prob, bins=1000)#, density=True)
            plt.yscale('log')
            plt.savefig(temp_basepath + "/prob.png")
            plt.close()
            plt.cla()
            plt.clf()
            
        print("LEN:",len(self.ps_points))
        self.n_train_step += 1
        

        integrator.prepare_data(self.ps_points, self.cross_sections)
        metrics = integrator.train(verbose=True, epochs=settings.PER_STAGE_EPOCHS)
        result = {}
        for key in metrics[0].keys():
            if key in ['zero_weights']:
                continue
            values = [d[key] for d in metrics]
            result[key] = np.mean(values)
            if key != 'loss':
                result[key + '_error'] = np.std(values) / np.sqrt(len(values))
        self.metrics.append(result)

        integrator.save()

        if self.n_train_step == settings.NUM_STAGES:
            self.reference_weights[integrator.matrix_name] = max(func_vals)
            visualize_metrics(self.metrics)
            integrator.basepath = self.basepath
            integrator.save()
        else:
            self.train_step(integrator)
        
        # self.reference_weights[matrix_name] = max(cross_sections)


    def train(self, bin_number, matrix_name):
        torch.manual_seed(settings.SEED)
        np.random.seed(settings.SEED)
        self.cur_bin_number = bin_number
        self.basepath = f"PythonSampler/{self.current_process}/{matrix_name}"
        do_train = not os.path.exists(f"{self.basepath}/best_model.pth") or settings.ALWAYS_RETRAIN
        if not do_train:
            logger.info(f"Loading existing model for {matrix_name}")
        else:
            logger.info(f"Training new model for {matrix_name}")

        if bin_number == 0:
            logger.info(f"Starting training for {self.bin_count} diagrams for process {self.current_process}")
            self.process_info["start_time"] = time.time()
            self.process_info["bin_count"] = self.bin_count
            self.process_info["n_channels"] = self.channel_count
            self.process_info["channels"] = {}
            self.process_info["ME"] = matrix_name
            if do_train:
                settings_dic = {}
                for key in settings.__dict__:
                    if not key.startswith("__") and key.isupper():
                        settings_dic[key] = settings.__dict__[key]
                with open(f"PythonSampler/{self.current_process}/settings.json", "w") as f:
                    json.dump(settings_dic, f, indent=4)
        logger.info(f"Learning diagram {bin_number} of {self.bin_count}: {matrix_name}, PS Dim: {self.n_dims}, Channels: {self.channel_count}")
        self.current_active_matrix_element = matrix_name
        start_time = time.time()


        if do_train and not settings.MULTI_STAGE_TRAINING:
            ps_points = np.random.rand(int(settings.INITIAL_POINTS), self.n_dims)
            cross_sections = np.array(self.python_sampler.dSigDRMatrix(ps_points))

            if settings.NORMALIZE_CROSS_SECTIONS:
                print(f"Max cross section: {np.max(cross_sections)}")
                cross_sections = cross_sections / np.max(cross_sections)
                

            zero_mask = cross_sections == 0
            num_zeros = np.sum(zero_mask)
            print(f"Total len {len(cross_sections)}, zeros: {num_zeros}, fraction: {num_zeros/len(cross_sections):.2f}")

            num_to_remove = int(settings.ZERO_POINT_REMOVAL * num_zeros)
            zero_indices = np.where(zero_mask)[0]
            remove_indices = np.random.choice(zero_indices, size=num_to_remove, replace=False)
            remove_mask = np.zeros(len(cross_sections), dtype=bool)
            remove_mask[remove_indices] = True
            keep_mask = ~remove_mask
            
            ps_points = ps_points[keep_mask]
            cross_sections = cross_sections[keep_mask]
            print(f"Remaining: {len(cross_sections)}")


            # zero_mask = cross_sections == 0
            # ps_points = ps_points[~zero_mask]
            # cross_sections = cross_sections[~zero_mask]
            # cross_sections = np.array(cross_sections)/max(cross_sections)
            
            self.reference_weights[matrix_name] = max(cross_sections)
        else:
            self.reference_weights[matrix_name] = 1.0
        ps_sampling_time = time.time() - start_time
        self.process_info["sampling_time"] = ps_sampling_time

        integrator = FlowSampler(self.python_sampler.dSigDRMatrix,self.basepath ,self.n_dims, self.channel_count, matrix_name=matrix_name,
                                                    current_process_name =self.current_process, single_channel=not settings.SPLIT_BY_CHANNELS
                                                    ,channel_selection_dim=self.channel_selection_dim)
        
        train_start_time = time.time()
        if not do_train:
            
            integrator.load()
        else:
            if settings.MULTI_STAGE_TRAINING:
                self.train_step(integrator)
            else:
                integrator.prepare_data(ps_points, cross_sections)
                integrator.train(verbose=True)
                integrator.save()
                # self.process_info[matrix_name] = integrator.meta
            if bin_number == self.bin_count - 1:
                self.process_info["end_time"] = time.time()
                self.process_info["total_time"] = self.process_info["end_time"] - self.process_info["start_time"]
                logger.info(f"Training finished for {self.bin_count} diagrams for process {self.current_process}")
                logger.info(f"Total time: {self.process_info['total_time']:.2f} seconds")
                # sum up total remaining channels
                total_remaining_channels = 0
                for matrix_name, matrix_info in self.process_info["channels"].items():
                    total_remaining_channels += matrix_info["remaining_channel_count"]
                logger.info(f"Total trained flows: {total_remaining_channels}")
            # with open(f"PythonSampler/{self.current_process}/process_info.json", "w") as f: 
            #     json.dump(self.process_info, f, indent=4)
        self.process_info["train_time"] = time.time() - train_start_time
        
        self.samplers[matrix_name] = integrator
        second_start_time = time.time()
        self.generate_n(settings.FINAL_SAMPLE_SIZE)
        second_end_time = time.time()
        self.process_info["generation_time"] = second_end_time - second_start_time
        with open(f"PythonSampler/{self.current_process}/process_info.json", "w") as f: 
            json.dump(self.process_info, f, indent=4)


    def load(self, matrix_name, ref_weight):
        self.current_active_matrix_element = matrix_name
        if matrix_name in self.samplers:
            logger.info(f"Diagram {matrix_name} already loaded for process {self.current_process}")
            return

        self.reference_weights[matrix_name] = ref_weight
        print(f"Loading diagram {matrix_name} for process {self.current_process}")
        try:
            integrator = FlowSampler(self.python_sampler.dSigDRMatrix,f"PythonSampler/{self.current_process}/{matrix_name}" ,self.n_dims, self.channel_count, matrix_name=matrix_name,
                                                    current_process_name =self.current_process, single_channel=not settings.SPLIT_BY_CHANNELS,
                                                    channel_selection_dim=self.channel_selection_dim)
            integrator.load()
            self.samplers[matrix_name] = integrator

        except Exception as e:
            traceback.print_exc()
            raise e
        
    def generate_n(self, n, n_cache = 100000):
        """Generate at least n accepted points from the current active matrix element."""
        if self.accepted_points >= n:
            with open("runstats/runstats.dat", "w") as f:
                f.write("{")
                f.write(f'"ZeroCount":{self.zero_count},\n')
                f.write(f'"PointsAttempted":{self.tot_points}')
                f.write("}")

            return
        while self.accepted_points < n:
            print(f"{self.accepted_points}/{n}, Tot: {self.tot_points}")
            sampler = self.samplers[self.current_active_matrix_element]
            reference_weight = self.reference_weights[self.current_active_matrix_element]

            weights, x, prob, func_vals, n_cache = sampler.get_weights(n_cache)
            # prob = prob/1e10
            non_zero_weights = weights[weights != 0]
            with open("runstats/data.csv", "a") as f:
                for i in non_zero_weights:
                    f.write(f"{i}\n")
            with open("runstats/func_vals.csv", "a") as f:
                for i in func_vals[func_vals != 0]:
                    f.write(f"{i}\n")
            with open("runstats/prob.csv", "a") as f:
                for i in prob[func_vals != 0]:
                    f.write(f"{i}\n")

            zero_weights = np.where(weights == 0)[0]
            self.zero_count += len(zero_weights)
            self.accepted_points += len(non_zero_weights)
            self.tot_points += n_cache
            self.generate_n(n)

    def generate(self, n_cache):
        self.tot_points += n_cache
        sampler = self.samplers[self.current_active_matrix_element]
        reference_weight = self.reference_weights[self.current_active_matrix_element]

        weights, x, prob, func_vals, n_cache = sampler.get_weights(n_cache)
        zero_func_vals = np.where(func_vals == 0)[0]
        zero_weights = np.where(weights == 0)[0]
        # max_weight = weights.max().item()
        max_weight = reference_weight

        percentile = np.percentile(weights, 99.9)
        print(f"Percentile: {percentile:.3f}")
        if percentile < reference_weight or reference_weight == 1.0:
            print(f"Replacing reference weight {reference_weight:.3f} with percentile {percentile:.3f}")
            max_weight = percentile
            reference_weight = percentile


        nonzero_weights = weights[weights > 2]
        unweighting_efficiency = weights.mean()/reference_weight*100

        est_accepted_points = nonzero_weights.sum()/(reference_weight*0.5)/n_cache*100*(float(len(zero_func_vals))/n_cache)


        stored_x = x.tolist()
        stored_prob = prob.tolist()
        stored_func_vals = func_vals.tolist()
        alpha_i= None
        estimated_integral = sampler._integrate(func_vals, prob, n_cache, alpha_i=alpha_i)
        # logger.info(f"Caching. Est. Integ.:{estimated_integral['integral']:.3f}+-{estimated_integral['error']:.5f}\n"
        #             f"Ref. weight: {max_weight:.3f}, Median {np.median(weights):.3f}, Mean: {np.mean(weights):.3f}\n"
        #             f"Zero: {float(len(zero_func_vals))/n_cache*100:.2f}%, Est. Acc: {est_accepted_points:.3f}%, Unweighting efficiency: {unweighting_efficiency:.3f}")
        print(f"I: {estimated_integral['integral']:.3f}+-{estimated_integral['error']:.5f}\n"
              f"Ref. weight: {max_weight:.3f}, Median {np.median(weights):.3f}, Mean: {np.mean(weights):.3f}\n"
              f"Zero: {float(len(zero_func_vals))/n_cache*100:.2f}%, Unweighting efficiency: {unweighting_efficiency:.3f}%, Tot. Points: {self.tot_points}")
        
        if self.axes is None:
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 5 * 2))
            axes = axes.flatten()
            axes[0].set_title("NF Probability distribution")
            axes[1].set_title("ME cross section distribution")
            axes[2].set_title("Weights distribution")
            axes[3].set_title("Weights distribution (>0)")
            axes[2].set_yscale("log")
            axes[3].set_yscale("log")
            self.axes=axes
            self.fig =fig

        self.axes[0].hist(prob, bins=50, alpha=0.5, density=True)
        self.axes[1].hist(func_vals, bins=50, alpha=0.5, density=True)
        self.axes[2].hist(weights, bins=50, alpha=0.5)
        self.axes[3].hist((weights/max_weight)[weights/max_weight <= 1], bins=50, alpha=0.5)
        
        print(f"Discarded weights: {np.sum(weights/max_weight >= 1)}")
        print(np.mean((weights/max_weight)[weights/max_weight <= 1])*100)

        ret_tuple = (stored_x, stored_prob, stored_func_vals, len(stored_x))


        return ret_tuple
    
    def finalize(self):
        plt.savefig(self.samplers[self.current_active_matrix_element].basepath + "/weights.png")
        plt.close(self.fig)
        pass