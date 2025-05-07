from abc import abstractmethod
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from nn import Flow
import Dataset
from AbstractSampler import AbstractSampler
import settings

import logging
logger = logging.getLogger('main')

# new abstract class to unify all flow based samplers
class FlowSampler:
    def __init__(self, cpp_integrand, basepath, n_dims,
                 channel_count, single_channel=True, channel_number=None,
                 current_process_name="UNKNOWN", matrix_name="UNKNOWN"):
        self.basepath = basepath
        self.n_dims = n_dims
        self.channel_selection_dim = settings.CHANNEL_SELECTION_DIM
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

        self.phase_space_points = None
        self.cross_sections = None
        self.model = None
        self.channel_weights = None

        self.meta = {}
    
    def matrix_callback(self, x, channel=None):
        # if channel is None:
        #     channel = torch.zeros((x.shape[0],))
        # else:
        #     if isinstance(channel, torch.Tensor):
        #         channel = channel.float()/ self.channel_count
        #     elif isinstance(channel, int):
        #         channel_tensor_size = (x.shape[0],)
        #         channel = torch.full(channel_tensor_size, channel / self.channel_count)
        #     else:
        #         raise ValueError("Channel must be a tensor or an integer.")
        x=x.to("cpu")
        # x = torch.cat((x[:, :self.channel_selection_dim], channel.unsqueeze(1), x[:, self.channel_selection_dim:]), dim=1)
        matrix_list = x.tolist()
        result = self.integrand(matrix_list)
        result_tensor = torch.tensor(result)
        return result_tensor
    
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


    def plot_dims(self, n_points=None, file_name="phase_space_distrib.png"):
        n_rows = (self.n_dims + 3) // 4
        fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        if n_points is None:
            n_points = len(self.cross_sections)
        samples = self.sample(n_points, return_prob=False, numpy=True)[0]
        for dim in range(self.n_dims):
            ax = axes[dim]
            bins = np.linspace(-5, 4, 50)
            ax.hist(self.phase_space_points[:,dim],weights=self.cross_sections, histtype="step", label="training data", bins=50, density=True)
            # ax.hist(x[:,dim],weights=y, histtype="step", label="training data", bins=50, density=True)
            ax.hist(samples[:,dim],  histtype="step", label="generated", bins=50, density=True)
            ax.set_title(f'Dimension {dim}')
            ax.legend()
        for j in range(self.n_dims, len(axes)):
            fig.delaxes(axes[j])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.basepath+"/"+file_name)
        plt.close()

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def sample(self, n_samples, return_prob=False, numpy=False):
        # needs to return 
        pass

    @abstractmethod
    def prepare_data(self, phase_space_points, cross_sections):
        pass

class SingleChannelFlowSampler(FlowSampler):
    def prepare_data(self, phase_space_points, cross_sections):
        self.phase_space_points = phase_space_points
        self.cross_sections = cross_sections

    # def sample(self, n_samples, return_prob=True, numpy=False, force_nonzero=True):
    #     with torch.no_grad():
    #             x, prob = self.model.sample(
    #                     n_samples,
    #                     return_prob=True,
    #                     device=self.device
    #                 )
    #     func_vals = self.matrix_callback(x, self.channel_number)
    #     zero_count = torch.sum(func_vals == 0).item()
    #     if zero_count == n_samples:
    #         raise ValueError("All function values from sampling are zero!") 
    #     if return_prob:
    #         if not numpy:
    #             return x, prob, func_vals
    #         else:
    #             return x.cpu().numpy(), prob.cpu().numpy(), func_vals.cpu().numpy()
    #     else:
    #         if not numpy:
    #             return x, func_vals
    #         else:
    #             return x.cpu().numpy(), func_vals.cpu().numpy()

    def sample(self, n_samples, return_prob=True, numpy=False, force_nonzero=False, max_attempts=5, only_sample=False):
        """
        Sample from the model and compute function values.
        
        Args:
            n_samples: Number of samples to generate
            return_prob: Whether to return probability values
            numpy: Whether to convert tensors to numpy arrays
            force_nonzero: If True, ensures all function values are non-zero
            max_attempts: Maximum number of sampling attempts when force_nonzero is True
            
        Returns:
            Samples and function values (and probabilities if return_prob is True)
        """
        with torch.no_grad():
            x, prob = self.model.sample(
                n_samples,
                return_prob=True,
                device=self.device
            )
        #exponentiate prob
        # prob = torch.exp(prob)
        if only_sample:
            if return_prob:
                if not numpy:
                    return x, prob
                else:
                    return x.cpu().numpy(), prob.cpu().numpy()
            if not numpy:
                return x
            else:
                return x.cpu().numpy()
        
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
        
        # Return results in requested format
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


    def train(self, batch_size: int = 1000, epochs = None, lr: float = 3e-4, verbose: bool = False) -> Tuple[Flow, float, List[float]]:
        if not epochs:
            epochs = settings.TRAINING_EPOCHS
        dataset = Dataset.PhaseSpaceDataset(self.phase_space_points, self.cross_sections, device=self.device)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        flow = Flow(dims_in=self.n_dims, uniform_latent=True).to(self.device)
        flow_best = Flow(dims_in=self.n_dims, uniform_latent=True).to(self.device)
        
        best_loss = float('inf')
        tot_losses = []
        
        flow.eval()
        with torch.no_grad():
            untrained_losses = []
            for phase_space, weight in loader:
                log_prob = flow.log_prob(phase_space)
                weighted_loss = -(log_prob * weight).mean()
                untrained_losses.append(weighted_loss.item())
            untrained_loss = sum(untrained_losses) / len(untrained_losses)
            tot_losses.append(untrained_loss)
            if verbose:
                print(f"Untrained model loss: {untrained_loss}")

        optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
        progress_bar = tqdm(range(epochs), desc="Training", unit="epoch", disable=not verbose)

        self.model = flow_best

        if settings.COLLECT_TRAINING_INTEGRATION_METRICS:
            self.integ_metrics = {
                "effective_sample_sizes": [self.integrate(1000)["effective_sample_size"]],
                "unweighting_efficiencies": [self.integrate(1000)["unweighting_efficiency"]],
                "variance_50_samples": [self.repeat_integrate(50)["error"]],
                "variance_100_samples": [self.repeat_integrate(100)["error"]],
                "variance_1000_samples": [self.integrate(1000)["error"]],
                "zero_count" : [self.integrate(1000)["zero_count"]]
            }
            self.plot_integral(close_plot=False, label="Untrained model 1", save=False)
            self.plot_integral(close_plot=False, label="Untrained model 2", save=False)
            self.plot_integral(close_plot=False, label="Untrained model 2", save=False)
            self.plot_integral(close_plot=False, label="Untrained model 2", save=False)
            self.plot_integral(close_plot=False, label="Untrained model 2", save=False)
            self.plot_integral(close_plot=False, label="Untrained model 2", save=False)
            self.plot_integral(close_plot=True, label="Untrained model 2", file_name="integral_convergence_before_training.png", save=True)

        for epoch in progress_bar:
            flow.train()
            epoch_losses = []
            for phase_space, weight in loader:
                optimizer.zero_grad()
                log_prob = flow.log_prob(phase_space)
                weighted_loss = -(log_prob * weight).mean()
                weighted_loss.backward()
                optimizer.step()
                epoch_losses.append(weighted_loss.item())
            
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
            
            # if epoch % 5 == 0:
            #     self.model = flow_best
            #     self.plot_integral(file_name=f"integral_convergence_epoch_{epoch}.png")
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
        
        self.plot_dims()
        

        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title(f"Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plot_path = os.path.join(self.basepath, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()

    def _integrate(self, func_vals, prob, sample_size):
        assert sample_size != 0, "Sample size cannot be zero"
        assert np.sum(prob) != 0, "Summed probability cannot be zero"
        zero_mask = func_vals == 0
        weights = func_vals / prob
        weights = np.where(prob == 0, 0, func_vals / prob)
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
    
    def load(self, path=None):
        self.model = Flow(dims_in=self.n_dims, uniform_latent=True).to(self.device)
        if path is None:
            path = os.path.join(self.basepath, "best_model.pth")
        self.model.load_state_dict(torch.load(path))


