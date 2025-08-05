from abc import abstractmethod
import copy
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional
import json

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from nn import Flow
import Dataset
import settings

import logging
logger = logging.getLogger('main')

import sys
sys.path.append("/home/finn/.pyenv/versions/madnis/lib/python3.10/site-packages/")



class FlowSampler:
    def __init__(self, cpp_integrand, basepath, n_dims,
                 channel_count, single_channel=True, channel_number=None,
                 current_process_name="UNKNOWN", matrix_name="UNKNOWN",
                 channel_selection_dim=None):
        self.basepath = basepath
        #one dim is channel selection dim
        if single_channel:
            self.n_dims = n_dims
        else:
            self.n_dims = n_dims - 1
        self.channel_selection_dim = channel_selection_dim
        self.channel_count = channel_count
        self.single_channel = single_channel
        self.integrand = cpp_integrand
        
        self.channel_number = channel_number
        self.device = "cuda" if settings.USE_CUDA else "cpu"

        if self.device == "cuda":
            "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.basepath, exist_ok=True)

        self.current_process_name = current_process_name
        self.matrix_name = matrix_name

        self.dataset = None
        self.model = None
        self.channel_weights = None


        self.flow_kwargs = {
            # "spline_type": "quadratic",
            # "bins": 5,
            "permutations": settings.COUPLING_CONSTRUCTOR,
            "blocks":settings.NUM_COUPLING_BLOCKS,
            "units" : settings.UNITS_PER_SUBNET,
            "layers": settings.SUBNET_LAYERS,
            # "min_bin_derivative" : 1e-2

        }

        if self.single_channel:
            # permutations="random", blocks=16, bins=12
            self.model = Flow(dims_in=self.n_dims, uniform_latent=True, **self.flow_kwargs).to(self.device)
            self.best_model = copy.deepcopy(self.model)
            # self.best_model = Flow(dims_in=self.n_dims, uniform_latent=True).to(self.device)
            
        else:
            self.model = Flow(dims_in=self.n_dims, uniform_latent=True, dims_c=1).to(self.device)
            self.best_model = copy.deepcopy(self.model)

    
    def matrix_callback(self, x, channel=None):
        x=x.to("cpu")
        if self.single_channel:
            # matrix_list = x.tolist()
            matrix_list = np.array(x.tolist())
            has_unbound = np.any(matrix_list< 0) or np.any(matrix_list > 1)
            if has_unbound:
                out_of_bounds = matrix_list[(matrix_list < 0) | (matrix_list > 1)]
                logger.warning(f"Unbound values in input: {matrix_list[out_of_bounds]}")
                matrix_list = np.clip(matrix_list, 0, 1)
            result = self.integrand(matrix_list)
            result_tensor = torch.tensor(result)
            return result_tensor
        else:
            if channel is None:
                channel = torch.zeros((x.shape[0],))
            else:
                if isinstance(channel, torch.Tensor):
                    channel = channel.float()/ self.channel_count
                elif isinstance(channel, int):
                    channel_tensor_size = (x.shape[0],)
                    channel = torch.full(channel_tensor_size, channel / self.channel_count)
                else:
                    raise ValueError("Channel must be a tensor or an integer.")
            x = torch.cat((x[:, :self.channel_selection_dim], channel.unsqueeze(1), x[:, self.channel_selection_dim:]), dim=1)
            matrix_list = x.tolist()
            result = self.integrand(matrix_list)
            result_tensor = torch.tensor(result)
            return result_tensor
        
    
    def prepare_data(self, phase_space_points, cross_sections):
        if self.single_channel:
            cross_sections = cross_sections / np.sum(cross_sections)
            self.dataset = Dataset.PhaseSpaceDataset(phase_space_points, cross_sections, device=self.device)
        else:
            preprocessor = Dataset.ChannelDataPreprocessor(self.channel_count)
            tot_cross_section = np.sum(cross_sections)
            phase_space_points, cross_sections = preprocessor.split_by_channel(
                phase_space_points,
                cross_sections,
                channel_selection_dim=self.channel_selection_dim
            )            
            tot_cross_section_per_channel = np.array([np.sum(i) for i in cross_sections])
            channel_weights = tot_cross_section_per_channel / tot_cross_section

            expected_weight = 1 / self.channel_count
            print(f"Expected weight: {expected_weight}")

            print(channel_weights)


            drop_threshold = float(settings.CHANNEL_DROP_THRESHOLD)*expected_weight
            
            print(drop_threshold)
            
            if np.sum(channel_weights < drop_threshold) > 0:
                dropped_weights = channel_weights[channel_weights < drop_threshold]
                logger.info(f"Dropping channels {np.where(channel_weights < drop_threshold)[0]} with weight < {max(dropped_weights):.4f} (Exp. Weight: {expected_weight:.4f})")
                #recalculate to make sure weights sum to 1
                tot_cross_section_per_channel = np.where(channel_weights > drop_threshold, tot_cross_section_per_channel, 0)
                total_cross_section = np.sum(tot_cross_section_per_channel)
                channel_weights = tot_cross_section_per_channel / total_cross_section
                logger.info(f"New channel weights: {channel_weights}")

            self.channel_weights = channel_weights

            channel_num_arrs = []
            for i in range(self.channel_count):
                channel_number = (i+0.5) / self.channel_count
                channel_num_arr = np.full((cross_sections[i].shape[0],), channel_number)
                channel_num_arrs.append(channel_num_arr)
            combined_cross_section = np.concatenate(cross_sections)
            combined_phase_space = np.concatenate(phase_space_points)
            channel_numbers = np.concatenate(channel_num_arrs)
            channel_weights = np.concatenate([np.full((cross_sections[i].shape[0],), self.channel_weights[i]) for i in range(self.channel_count)])
            
        
            self.dataset = Dataset.PhaseSpaceChannelDataset(combined_phase_space, combined_cross_section, channel_numbers, channel_weights, device=self.device)
            # logger.info(f"Channel weights: {self.channel_weights}")
            

            


    def train(self, batch_size = None, epochs = None, lr = None, verbose: bool = False) -> Tuple[Flow, float, List[float]]:
        if not epochs:
            epochs = settings.TRAINING_EPOCHS
        if not batch_size:
            batch_size = settings.BATCH_SIZE
        if not lr:
            lr = settings.LEARNING_RATE
        
        flow = self.model
        flow_best = self.best_model


        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        # self.plot_dims(file_name="phase_space_distrib_before_training.png")
        
        best_loss = float('inf')
        tot_losses = []
        
        flow.eval()
        with torch.no_grad():
            untrained_losses = []
            if self.single_channel:
                for phase_space, weight in loader:
                    log_prob = flow.log_prob(phase_space)
                    weighted_loss = -(log_prob * weight).mean()
                    untrained_losses.append(weighted_loss.item())
            else:
                for phase_space, weight, channel_number, channel_weight in loader:
                    channel_number = channel_number.unsqueeze(1)
                    log_prob = flow.log_prob(phase_space, c=channel_number)
                    weighted_loss = -(log_prob * weight*channel_weight).mean()
                    untrained_losses.append(weighted_loss.item())
            untrained_loss = sum(untrained_losses) / len(untrained_losses)
            tot_losses.append(untrained_loss)
            
        if settings.OPTIMIZER == 'adam':
            optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
        elif settings.OPTIMIZER == 'adamw':
            optimizer = torch.optim.AdamW(flow.parameters(), lr=lr)
        elif settings.OPTIMIZER == 'rmsprop':
            optimizer = torch.optim.RMSprop(flow.parameters(), lr=lr)
        elif settings.OPTIMIZER == 'sgd':
            optimizer = torch.optim.SGD(flow.parameters(), lr=lr)
        progress_bar = tqdm(range(epochs), desc="Training", unit="epoch", disable=not verbose)
        if settings.LIVE_TRAINING_PLOT:
            plt.ion()
        plt_epochs = [0]
        metrics = [self.metrics(50000)]
        metrics[-1]["loss"] = tot_losses[0]
        total_elems = len(metrics[0].keys())
        fig, axes = plt.subplots(nrows=(total_elems+1)//2, ncols=2, figsize=(20, 5 *2))
        axes = axes.flatten()
        lines = []
        for i , key in enumerate(metrics[0].keys()):
            # axes[i].set_title(key)
            axes[i].set_xlabel('Training Iteration')
            axes[i].set_ylabel(key)
            line, = axes[i].plot(plt_epochs, metrics[0][key], label=key, marker="o")
            lines.append(line)
            axes[i].legend()
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.set_title('Training Loss')
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Loss')
        # ax.grid(True)
        # line, = ax.plot(plt_epochs,tot_losses)
        self.model = flow_best


        monitor_data = {"total_grad_norm":[], "max_grad":[], "jac_mean":[], "jac_max":[], "jac_std":[], "jac_min": [], "log_prob_mean": [], "log_prob_std": [],
                        "abs_log_prob_max": []}
        jac = None
        for epoch in progress_bar:
            flow.train()
            epoch_losses = []
            if self.single_channel:
                for phase_space, weight in loader:
                    optimizer.zero_grad()
                    log_prob, jac = flow.log_prob(phase_space, return_jacobian=True)

                    # print(f"Log prob: Mean {log_prob.mean().item():.3e}, Max {log_prob.max().item():.3e}, Min {log_prob.min().item():.3e}, Std {log_prob.std().item():.3e}")
                    # print(f"Jacobian: Mean {jac.mean().item():.3e}, Max {jac.max().item():.3e}, Min {jac.min().item():.3e}, Std {jac.std().item():.3e}")
                    # print(f"Weight: Mean {weight.mean().item():.3e}, Max {weight.max().item():.3e}, Min {weight.min().item():.3e}, Std {weight.std().item():.3e}")

                    weighted_loss = -(log_prob * weight).mean()

                    # lambd = 0.00005
                    # weighted_loss = -(log_prob * weight).mean() + lambd * (log_prob.abs().mean() )**2

                    weighted_loss.backward()

                    # normalized_weight = weight / weight.sum()
                    # loss = -(normalized_weight * log_prob).sum()
                    # loss.backward()
                    
                    optimizer.step()
                    epoch_losses.append(weighted_loss.item())
            else:
                for phase_space, weight, channel_number, channel_weight in loader:
                    optimizer.zero_grad()
                    channel_number = channel_number.unsqueeze(1)
                    log_prob = flow.log_prob(phase_space, c=channel_number)
                    weighted_loss = -(log_prob * weight* channel_weight).mean()
                    weighted_loss.backward()
                    optimizer.step()
                    epoch_losses.append(weighted_loss.item())
            total_grad_norm = 0
            cur_max_grad = 0
            for name, param in flow.named_parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2)
                    total_grad_norm += param_grad_norm.item() ** 2
                    cur_max_grad = max(cur_max_grad, param_grad_norm.item())

                        
            total_grad_norm = total_grad_norm ** 0.5
            monitor_data["total_grad_norm"].append(total_grad_norm)
            monitor_data["max_grad"].append(cur_max_grad)
            
            epoch_loss = sum(epoch_losses) / len(epoch_losses)
            tot_losses.append(epoch_loss)
            if verbose:
                progress_bar.set_postfix(loss=f"{epoch_loss:.3e}")
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                flow_best.load_state_dict(flow.state_dict())
                self.model = flow_best
            if settings.COLLECT_TRAINING_INTEGRATION_METRICS:
                res_1000 = self.repeat_integrate(1000, 3)
                self.integ_metrics["effective_sample_sizes"].append(res_1000["effective_sample_size"])
                self.integ_metrics["unweighting_efficiencies"].append(res_1000["unweighting_efficiency"])
                self.integ_metrics["zero_count"].append(res_1000["zero_count"])
                self.integ_metrics["variance_1000_samples"].append(res_1000["error"])
                self.integ_metrics["variance_50_samples"].append(self.repeat_integrate(50)["error"])
                self.integ_metrics["variance_100_samples"].append(self.repeat_integrate(100)["error"])
                line.set_xdata(epochs)
            plt_epochs.append(epoch+1)
            try:
                metrics.append(self.metrics(20000, epoch=epoch+1))
                metrics[-1]["loss"] = epoch_loss
                for i, key in enumerate(metrics[-1].keys()):
                    y_vals = [metric[key] for metric in metrics]
                    lines[i].set_ydata(y_vals)
                    lines[i].set_xdata(plt_epochs)
                    axes[i].relim()
                    axes[i].autoscale_view()
            except Exception as e:
                logger.error(f"Error during metrics calculation: {e}")

            fig.canvas.draw()
            fig.canvas.flush_events()
            # line.set_ydata(tot_losses)
            # line.set_xdata(plt_epochs)
            # ax.relim()
            # ax.autoscale_view()
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            for key, value in flow.monitor_data.items():
                if key not in monitor_data:
                    monitor_data[key] = []
                monitor_data[key].append(value)
                                # print(f"Mean {jac.mean().item():.3f}, Max {jac.max().item():.3f}, Min {jac.min().item():.3f}, Std {jac.std().item():.3f}")
            if jac == None:
                jac = torch.zeros_like(log_prob)
            monitor_data["jac_max"].append(jac.mean().item())
            monitor_data["jac_min"].append(jac.min().item())
            monitor_data["jac_std"].append(jac.std().item())
            monitor_data["jac_mean"].append(jac.mean().item())

            monitor_data["log_prob_mean"].append(log_prob.mean().item())
            monitor_data["log_prob_std"].append(log_prob.std().item())
            monitor_data["abs_log_prob_max"].append(torch.abs(log_prob).max().item())
            # print(f"Mean {log_prob.mean().item():.3f}, Max {log_prob.max().item():.3f}, Min {log_prob.min().item():.3f}, Std {log_prob.std().item():.3f}")

        num_plots = len(monitor_data)
        ncols = 2
        nrows = (num_plots + ncols - 1) // ncols
        plt.figure(figsize=(12, 6 * nrows))
        for i, (key, values) in enumerate(monitor_data.items(), 1):
            plt.subplot(nrows, ncols, i)
            plt.plot(values)
            plt.title(key)
            plt.xlabel("Epoch")
            plt.ylabel(key)
        plt.tight_layout()  # Prevents overlapping of subplots
        plt.savefig(os.path.join(self.basepath, "all_metrics_plot.png"))
        plt.close()


        self.model = copy.deepcopy(flow_best)
        self.best_model = copy.deepcopy(flow_best)
        self.model = flow_best
        self.losses = tot_losses

        if settings.COLLECT_TRAINING_INTEGRATION_METRICS:
            self.plot_integral(label="Trained model 1", close_plot=False, save=False)
            self.plot_integral(label="Trained model 2", close_plot=False, save=False)
            self.plot_integral(label="Trained model 3", close_plot=False, save=False)
            self.plot_integral(label="Trained model 3", close_plot=False, save=False)
            self.plot_integral(label="Trained model 3", close_plot=False, save=False)
            self.plot_integral(label="Trained model 3", close_plot=False, save=False)
            self.plot_integral(label="Trained model 4", close_plot=True, save=True)
            self.plot_integration_metrics()
        
        if settings.PLOT_DIMS:
            self.plot_dims()
        
        plot_path = os.path.join(self.basepath, "loss_plot.png")
        plt.savefig(plot_path)
        if settings.LIVE_TRAINING_PLOT:
            plt.ioff()
        plt.close()
        return metrics
    
    def get_weights(self, n_points: int = 1000000):
        n_cache = n_points
        if settings.SPLIT_BY_CHANNELS:
            # channel_idx = torch.randint(0, self.channel_count, (n_cache,))
            probs = torch.tensor(self.channel_weights)
            channel_idx = torch.multinomial(probs, n_cache, replacement=True)
            c = (channel_idx.float() + 0.5) / self.channel_count
            c= c.to(self.device).unsqueeze(1)
            x, prob, func_vals = self.sample(n_cache, c=c, numpy=True)
            alpha_i = self.channel_weights[channel_idx]
        else:
            x, prob, func_vals = self.sample(n_cache, numpy=True)
            # alpha_i = np.ones((n_cache, self.channel_count))
            alpha_i = np.ones(n_cache)
        n_cache = len(x)
        weights =  alpha_i*func_vals / (prob)

        return weights, x, prob, func_vals, n_cache

    def sample(self, n_samples, return_prob=True, numpy=False, force_nonzero=False, max_attempts=5, only_sample=False, c=None):
        with torch.no_grad():
            if self.single_channel:
                x, prob = self.model.sample(
                    n_samples,
                    return_prob=True,
                    # return_latent=True,
                    device=self.device
                )
            else:
                if c is None:
                    outputs = []
                    for i in range(self.channel_count):
                        channel_number = (i+0.5) / self.channel_count
                        c = torch.full((n_samples,), channel_number).to(self.device).unsqueeze(1)
                        outputs.append(self.model.sample(
                            n_samples,
                            return_prob=True,
                            c=c,
                            device=self.device
                        ))
                    x = torch.cat([out[0] for out in outputs])
                    prob = torch.cat([out[1] for out in outputs])
                    # raise ValueError("Channel number must be provided for multi-channel sampling.")
                if isinstance(c, float):
                    c = torch.full((n_samples,), c).to(self.device).unsqueeze(1)
                x, prob = self.model.sample(
                    n_samples,
                    return_prob=True,
                    c=c,
                    device=self.device
                )
        if only_sample:
            if return_prob:
                if not numpy:
                    return x, prob
                else:
                    return x.cpu().numpy(), prob.cpu().numpy()#, latent.cpu().numpy()
            if not numpy:
                return x
            else:
                return x.cpu().numpy()
        # print(f"Prob < 1: {torch.sum(prob < 0.5).item()}/{n_samples} ({torch.sum(prob < 0.5).item()/n_samples*100:.2f}%)")
        # x = x[prob > 0.5]
        # prob = prob[prob > 0.5]
        # n_samples = len(x)

        func_vals = self.matrix_callback(x, self.channel_number)
        zero_count = torch.sum(func_vals == 0).item()
        if zero_count == n_samples:
            raise ValueError("All function values from sampling are zero!")
        # Handle force_nonzero logic if needed
        if force_nonzero and zero_count > 0:
            # Get initial indices of non-zero values
            nonzero_mask = func_vals != 0
            nonzero_indices = torch.where(nonzero_mask)[0]
            # Create final arrays with only non-zero elements initially
            final_x = x[nonzero_indices]
            final_prob = prob[nonzero_indices]
            final_func_vals = func_vals[nonzero_indices]
            remaining_samples = n_samples - len(nonzero_indices)
            attempt = 0
            while remaining_samples > 0 and attempt < max_attempts:
                attempt += 1
                # Calculate how many additional samples to generate
                # Use dynamic scaling based on observed zero ratio
                zero_percent = zero_count / n_samples
                if zero_percent >= 1.0:
                    # Avoid division by zero
                    additional_samples = remaining_samples * 10
                else:
                    # Generate at least twice as many samples as needed, accounting for zero ratio
                    additional_samples = remaining_samples * int(1 / (1 - zero_percent)) * 2
                # Ensure we generate at least remaining_samples
                additional_samples = max(remaining_samples, additional_samples)
                # Sample more points
                with torch.no_grad():
                    new_x, new_prob = self.model.sample(
                        additional_samples,
                        return_prob=True,
                        device=self.device
                    )
                new_func_vals = self.matrix_callback(new_x, self.channel_number)
                new_nonzero_mask = new_func_vals != 0
                new_nonzero_indices = torch.where(new_nonzero_mask)[0]
                if len(new_nonzero_indices) > 0:
                    samples_to_take = min(len(new_nonzero_indices), remaining_samples)
                    final_x = torch.cat([final_x, new_x[new_nonzero_indices[:samples_to_take]]])
                    final_prob = torch.cat([final_prob, new_prob[new_nonzero_indices[:samples_to_take]]])
                    final_func_vals = torch.cat([final_func_vals, new_func_vals[new_nonzero_indices[:samples_to_take]]])
                    remaining_samples -= samples_to_take
                zero_count = torch.sum(new_func_vals == 0).item()
            if remaining_samples > 0:
                raise ValueError(f"Could not generate {n_samples} non-zero samples after {max_attempts} attempts")
            # Ensure we have exactly n_samples
            if len(final_x) > n_samples:
                final_x = final_x[:n_samples]
                final_prob = final_prob[:n_samples]
                final_func_vals = final_func_vals[:n_samples]
            x = final_x
            prob = final_prob
            func_vals = final_func_vals
            self.actual_sample_size = len(final_x)+zero_count
        if return_prob:
            if not numpy:
                return x, prob, func_vals
            else:
                return x.cpu().numpy(), prob.cpu().numpy(), func_vals.cpu().numpy()
        else:
            if not numpy:
                return x, func_vals
            else:
                return x.cpu().numpy(), func_vals.cpu().numpy()
            
    def _metrics(self, prob, func_vals, n_samples):
        metrics = self._integrate(func_vals, prob, n_samples)
        weights = func_vals / prob
        
        zero_func_vals = np.where(func_vals == 0)[0]
        zero_weights = np.where(weights == 0)[0]

        #hack to get stable weight 
        reference_weight = np.percentile(weights,99)#weights.max()

        nonzero_weights = weights[weights > 0]
        unweighting_efficiency = weights.mean()/reference_weight*100
        weight_mean = weights.mean()

        est_accepted_points = nonzero_weights.sum()/(reference_weight*0.5)/n_samples*100*(float(len(zero_func_vals))/n_samples)
        # metrics["est_accepted_points"] = est_accepted_points

        metrics["unweighting_efficiency"] = unweighting_efficiency
        metrics["weight_mean"] = weight_mean
        metrics["max_weight"] = reference_weight
        metrics["zero_weights"] = len(zero_weights)
        metrics["prob_mean"] = prob.mean().item()
        metrics["prob_std"] = prob.std().item()
        metrics["prob_max"] = abs(prob).max().item()
        to_del = ["ess","zero_count", ]
        for i in to_del:
            if i in metrics:
                del metrics[i]
        return metrics
        

    def metrics(self, n_samples, epoch=None):

        if self.single_channel:
            x, prob, func_vals = self.sample(n_samples, return_prob=True, numpy=True)
            if epoch is not None and (epoch % 5 == 0 or epoch == 1):
                fig, ax1 = plt.subplots()
                color = 'tab:blue'
                n1, bins, patches = ax1.hist(prob, bins=50, color=color, alpha=0.7)
                ax1.set_yscale('log')
                ax1.set_xlabel('Value')
                ax1.set_ylabel('Probability', color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                ax2 = ax1.twinx()
                color = 'tab:red'
                # ax2.hist(func_vals, bins=30, color=color, alpha=0.7)
                n2, _, patches = ax2.hist(func_vals, bins=bins, color=color, alpha=0.7)  # Using same bins
                ax2.set_yscale('log')
                ax2.set_ylabel('Function Values', color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                plt.savefig(os.path.join(self.basepath,f"epoch:{epoch}_func_vals_prob_hist.png"))
                plt.close(fig)
            return self._metrics(prob, func_vals, n_samples)
        else:
            metric_list = []
            for i in range(self.channel_count):
                channel_number = (i+0.5) / self.channel_count
                x, prob, func_vals = self.sample(n_samples, return_prob=True, numpy=True, c=channel_number)
                metric_list.append(self._metrics(prob, func_vals, n_samples))
            # Combine metrics from all channels
            combined_metrics = {}
            combined_metrics["max_weight"] = np.max([metric["max_weight"] for metric in metric_list])
            combined_metrics["weight_mean"] = np.mean([metric["weight_mean"] for metric in metric_list])
            combined_metrics["unweighting_efficiency"] = np.mean([metric["unweighting_efficiency"] for metric in metric_list])
            combined_metrics["zero_weights"] = np.sum([metric["zero_weights"] for metric in metric_list])
            return combined_metrics
        # x, prob, func_vals = self.sample(n_samples, return_prob=True, numpy=True)
        


    def _integrate(self, func_vals, prob, sample_size, alpha_i = None):
        assert sample_size != 0, "Sample size cannot be zero"
        assert np.sum(prob) != 0, "Summed probability cannot be zero"
        zero_mask = func_vals == 0
        if alpha_i is None:
            weights = func_vals / prob
            weights = np.where(prob == 0, 0, func_vals / prob)
        else:
            weights = func_vals * alpha_i / (prob)
            weights = np.where(prob == 0, 0, func_vals * alpha_i/ (prob))
        integral = np.sum(weights) / sample_size
        error = np.sqrt(np.var(weights) / sample_size)
        if np.sum(weights) == 0:
            normalized_weights = np.zeros_like(weights)
            ess=0
            unweighting_efficiency = 0
        else:
            normalized_weights = weights / weights.sum()
            ess = 1.0 / (normalized_weights ** 2).sum().item()
            unweighting_efficiency = weights.mean()/weights.max().item()
        normalized_weights = weights / weights.sum()
        return {
            "integral": integral.item(),
            "error": error.item(),
            "ess": ess,
            "effective_sample_size": ess,
            "unweighting_efficiency": unweighting_efficiency,
            "zero_count": zero_mask.sum(),
        }

    
    def integrate(self, sample_size):
        x, prob, func_vals = self.sample(sample_size,return_prob=True, numpy=True)
        return self._integrate(func_vals, prob, sample_size)
    
    def repeat_integrate(self, sample_size, n_times=10):
        results = []
        for i in range(n_times):
            x, prob, func_vals = self.sample(sample_size,return_prob=True, numpy=True)
            result = self._integrate(func_vals, prob, sample_size)
            results.append(result)
        results = {
            "integral": np.mean([r["integral"] for r in results]),
            "error": np.std([r["integral"] for r in results]),
            "ess": np.mean([r["ess"] for r in results]),
            "effective_sample_size": np.mean([r["effective_sample_size"] for r in results]),
            "unweighting_efficiency": np.mean([r["unweighting_efficiency"] for r in results]),
            "zero_count": np.mean([r["zero_count"] for r in results]),
        }
        return results
    
    def plot_integral(self, sample_size=500, plot_points=50, file_name="integral_convergence.png", 
                      close_plot=True, label="Integral Means", save=True):
        x, prob, func_vals = self.sample(sample_size,return_prob=True, numpy=True, force_nonzero=False)  
        means = []
        errors = []
        len_per_iter = sample_size // plot_points
        len_per_iter_corr = len_per_iter#self.actual_sample_size // plot_points
        for i in range(plot_points):
            end = (i + 1) * len_per_iter
            result = self._integrate(func_vals[0:end], prob[0:end], len_per_iter_corr*(i+1))
            means.append(result["integral"])
            errors.append(result["error"])
        means = np.array(means)
        errors = np.array(errors)
        x_values = range(1, plot_points + 1)
        def fit_function(x, a, b):
            return a * np.sqrt(x) + b
        popt, pcov = curve_fit(fit_function, x_values[2:], errors[2:])
        a, b = popt
        fit_line = fit_function(x_values, a, b)
        # plt.plot(x_values, fit_line, 'r-', label=f"Fit: $a \\cdot \\sqrt{{x}} + b$\na={a:.2f}, b={b:.2f}")
        plt.plot(x_values, means, 'o-', label=f'{result["integral"]:.3f}+-{result["error"]:.3f}')#label=label)
        # plt.errorbar(x_values, means, yerr=errors, fmt='o', capsize=5, label="Integral Means")
        plt.fill_between(x_values, means - errors, means + errors, color='blue', alpha=0.1)

        plt.title("Integral Means with Error Bars")
        plt.xlabel(f"i * {len_per_iter} points")
        plt.ylabel("Mean Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.basepath, file_name))
        if close_plot:
            plt.close()

    def save(self):
        model_path = os.path.join(self.basepath, "best_model.pth")
        torch.save(self.model.state_dict(), model_path)
        if not self.single_channel:
            param_dic = {
                "channel_weights": list(self.channel_weights),
            }
            param_path = os.path.join(self.basepath, "params.json")
            with open(param_path, "w") as f:
                json.dump(param_dic, f)

    
    def load(self):
        if self.single_channel:
            self.model = Flow(dims_in=self.n_dims, uniform_latent=True, **self.flow_kwargs).to(self.device)
        else:
            self.model = Flow(dims_in=self.n_dims, uniform_latent=True, dims_c=1, **self.flow_kwargs).to(self.device)
        path = os.path.join(self.basepath, "best_model.pth")
        self.model.load_state_dict(torch.load(path))
        json_path = os.path.join(self.basepath, "params.json")
        if not self.single_channel:
            with open(json_path, "r") as f:
                param_dic = json.load(f)
            self.channel_weights = np.array(param_dic["channel_weights"])


    def plot_integration_metrics(self, img_name="integration_metrics.png"):
        total_elems = len(self.integ_metrics.keys())
        fig, axes = plt.subplots(nrows=(total_elems+1)//2, ncols=2, figsize=(20, 5 *2))
        axes = axes.flatten()
        for i , key in enumerate(self.integ_metrics.keys()):
            axes[i].plot(self.integ_metrics[key], label=key, marker="o")
            axes[i].set_xlabel('Training Iteration')
            axes[i].set_ylabel(key)
            axes[i].legend()
        # for j in range(self.n_dims, len(axes)):
        #     fig.delaxes(axes[j])
        fig.tight_layout()  # Adjust layout to prevent overlap
        plt.savefig(self.basepath+"/"+img_name)
        plt.close()

    def _plot_dims(self,cross_sections,phase_space_points, n_points=None, c=None, file_name="phase_space_distrib.png"):
        if n_points is None:
            n_points = len(cross_sections[:500000])
        samples, prob = self.sample(n_points, c=c, return_prob=True, numpy=True, only_sample=True)

        n_rows = (self.n_dims + 3) // 4
        fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        
        for dim in range(self.n_dims):
            ax = axes[dim]
            ax.hist(phase_space_points[:500000,dim],weights=cross_sections[:500000], histtype="step", label="training data", bins=50, density=True)
            ax.hist(samples[:,dim],  histtype="step", label="generated", bins=50, density=True)
            # ax.hist(latent[:,dim],weights=prob ,histtype="step", label="generated", bins=50)
            ax.set_title(f'Dimension {dim}')
            ax.legend()
        for j in range(self.n_dims, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.basepath+"/"+file_name)
        plt.close()

    def plot_dims(self, n_points=None, file_name="phase_space_distrib.png"):
        if self.single_channel:
            phase_space_points = self.dataset.phase_space.cpu().numpy()
            cross_sections = self.dataset.cross_sections.cpu().numpy()
            self._plot_dims(cross_sections, phase_space_points, n_points=n_points, file_name=file_name)
        else:
            os.makedirs(self.basepath+"/channel_plots", exist_ok=True)
            for i in range(self.channel_count):
                channel_number = (i+0.5) / self.channel_count
                phase_space_points = self.dataset.phase_space[self.dataset.channel_numbers == channel_number].cpu().numpy()
                cross_sections = self.dataset.cross_sections[self.dataset.channel_numbers == channel_number].cpu().numpy()
                self._plot_dims(cross_sections, phase_space_points, n_points=n_points, c=channel_number,file_name=f"/channel_plots/channel_{i}_{file_name}")
        


