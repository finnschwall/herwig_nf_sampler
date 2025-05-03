import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from madnis.nn import Flow
from Dataset import PhaseSpaceDataset
from AbstractSampler import AbstractSampler

class MadnisFlow(AbstractSampler):

        
    def train(
        self,
        batch_size: int = 1000,
        epochs: int = 20,
        lr: float = 3e-4,
        verbose: bool = False
    ) -> Tuple[Flow, float, List[float]]:
        """
        Train a single normalizing flow for one channel.
        
        Args:
            phase_space_points: Array of phase space points
            cross_sections: Array of cross section values
            batch_size: Batch size for training
            epochs: Number of epochs to train
            lr: Learning rate
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best flow model, best loss value, list of losses)
        """

        # Create dataset and loader
        dataset = PhaseSpaceDataset(self.sampled_points, self.sampled_cross_sections, device=self.device)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize flow models
        flow = Flow(dims_in=self.n_dims, uniform_latent=True).to(self.device)
        flow_best = Flow(dims_in=self.n_dims, uniform_latent=True).to(self.device)
        
        # Evaluate untrained model
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
        
        # Train the flow
        optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
        progress_bar = tqdm(range(epochs), desc="Training", unit="epoch", disable=not verbose)
        
        self.effective_sample_sizes = []
        self.unweighting_efficiencies = []

        self.model = flow_best
        self.model.to("cpu")
        self.plot_integral(file_name=f"integral_convergence_before_training.png")

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
                progress_bar.set_postfix(loss=f"{epoch_loss:.6f}")
            
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                flow_best.load_state_dict(flow.state_dict())
            if epoch % 5 == 0:
                self.model = flow_best
                self.model.to("cpu")
                self.plot_integral(file_name=f"integral_convergence_epoch_{epoch}.png")
        self.model = flow_best
        self.model.to("cpu")
        self.plot_integral()
        self.losses = tot_losses
        # self.save_model(flow_best)
        self.plot_dims()
        self.plot_integral()
        
        fig, ax1 = plt.subplots()

        
        ax1.plot(self.effective_sample_sizes, label="Effective Sample Size", marker="o", color='b')
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Effective Sample Size', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(self.unweighting_efficiencies, label="Unweighting Efficiency", marker="o", color='r')
        ax2.set_ylabel('Unweighting Efficiency', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        fig.tight_layout()  # Adjust layout to prevent overlap
        fig.legend(loc="upper left", bbox_to_anchor=(0,1), bbox_transform=ax1.transAxes)
        plt.savefig(os.path.join(self.basepath, "integration_metrics.png"))

        return flow_best, best_loss, tot_losses
    
    def matrix_callback(self, x, channel=None):
        if channel is None:
            # channel = torch.rand((x.shape[0],), device=self.device)
            channel = torch.zeros((x.shape[0],))
        else:
            if isinstance(channel, torch.Tensor):
                channel = channel.float()/ self.total_channel_count
            elif isinstance(channel, int):
                channel_tensor_size = (x.shape[0],)
                channel = torch.full(channel_tensor_size, channel / self.total_channel_count)
            else:
                raise ValueError("Channel must be a tensor or an integer.")
        # x = torch.cat((channel.unsqueeze(1), x), dim=1)

        x = torch.cat((x[:, :self.channel_selection_dim], channel.unsqueeze(1), x[:, self.channel_selection_dim:]), dim=1)
        matrix_list = x.tolist()
        result = self.integrand(matrix_list)
        result_tensor = torch.tensor(result)
        # self.sampled_points.extend(matrix_list)
        # self.sampled_cross_sections.extend(result)
        return result_tensor#.to(self.device)
    
    
    def _integrate(self, func_vals, prob, sample_size):
        weights = func_vals / prob
        integral = torch.sum(weights) / sample_size
        error = torch.sqrt(torch.var(weights) / sample_size)
        normalized_weights = weights / weights.sum()
        ess = 1.0 / (normalized_weights ** 2).sum()
        unweighting_efficiency = weights.mean()/weights.max()
        normalized_weights = weights / weights.sum()
        variance = ((normalized_weights - 1/len(weights)) ** 2).sum() * len(weights)
        vrf = 1.0 / (1.0 + variance)
        return {
            "integral": integral.item(),
            "error": error.item(),
            "ess": ess.item(),
            "effective_sample_size": ess.item(),
            "unweighting_efficiency": unweighting_efficiency.item(),
            "vrf": vrf.item(),
        }

    
    def integrate(self, sample_size):
        with torch.no_grad():
            x, prob = self.model.sample(
                    sample_size,
                    # channel=channels_remapped,
                    return_prob=True,
                    device=self.device
                    # dtype=self.dummy.dtype,
                )
        func_vals = self.matrix_callback(x, self.channel_number)
        return self._integrate(func_vals, prob, sample_size)
        
    def plot_integral(self, sample_size=1000, plot_points=100, file_name="integral_convergence.png"):
        with torch.no_grad():
            # prior = torch.randn((sample_size, self.n_dims))
            # x, jac = self.model.transform(prior)
            x, prob = self.model.sample(
                    sample_size,
                    # channel=channels_remapped,
                    return_prob=True,
                    # device=self.device
                    # dtype=self.dummy.dtype,
                )

        func_vals = self.matrix_callback(x, self.channel_number)
        means = []
        errors = []
        len_per_iter = sample_size // plot_points
        for i in range(plot_points):
            end = (i + 1) * len_per_iter
            result = self._integrate(func_vals[0:end], prob[0:end], len_per_iter*i)
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
        plt.plot(x_values, means, 'o-', label="Integral Means")
        # plt.errorbar(x_values, means, yerr=errors, fmt='o', capsize=5, label="Integral Means")
        plt.fill_between(x_values, means - errors, means + errors, color='blue', alpha=0.1)

        plt.title("Integral Means with Error Bars")
        plt.xlabel(f"i * {len_per_iter} points")
        plt.ylabel("Mean Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.basepath, file_name))
        plt.close()
        self.effective_sample_sizes.append(result["ess"])
        self.unweighting_efficiencies.append(result["unweighting_efficiency"])

        # n_rows = (self.n_dims + 3) // 4
        # fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(20, 5 * n_rows))
        # axes = axes.flatten()
        # for dim in range(self.n_dims):
        #     ax = axes[dim]
        #     bins = np.linspace(-5, 4, 50)
        #     ax.hist(x[:,dim],  histtype="step", label="generated", bins=50, density=True)
        #     ax.set_title(f'Dimension {dim}')
        #     ax.legend()
        # for j in range(self.n_dims, len(axes)):
        #     fig.delaxes(axes[j])
        # plt.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.show()

    
    def sample(self, n_samples):
        """
        Sample points from the trained flow model.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(n_samples)
        return samples.cpu().numpy()
    
    # def plot_dims(self, path, flow_best, sampled_points, cross_sections, n_dims, n_points):
    #     n_rows = (n_dims + 3) // 4
    #     fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(20, 5 * n_rows))
        
    #     axes = axes.flatten()
    #     flow_best.to("cpu")
    #     with torch.no_grad():
    #         samples = flow_best.sample(n_points)
    #     # flow_best.to(device)
    #     for dim in range(n_dims):
    #         ax = axes[dim]
    #         bins = np.linspace(-5, 4, 50)
    #         ax.hist(sampled_points[:,dim],weights=cross_sections, histtype="step", label="training data", bins=50, density=True)
    #         # ax.hist(x[:,dim],weights=y, histtype="step", label="training data", bins=50, density=True)
    #         ax.hist(samples[:,dim],  histtype="step", label="generated", bins=50, density=True)
    #         ax.set_title(f'Dimension {dim}')
    #         ax.legend()

    #     for j in range(n_dims, len(axes)):
    #         fig.delaxes(axes[j])
        
    #     # fig.suptitle(f'Histograms for Each dim for channel {channel}', fontsize=18)
    #     plt.tight_layout(rect=[0, 0, 1, 0.95])
    #     plt.savefig(path)

    
    def save_model(self, flow):
        # os.makedirs(dir_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(self.basepath, "best_model.pth")
        torch.save(flow.state_dict(), model_path)
        
        # Save loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses)
        plt.title(f"Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plot_path = os.path.join(self.basepath, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()