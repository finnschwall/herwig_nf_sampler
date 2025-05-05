from abc import abstractmethod
import matplotlib.pyplot as plt
import os
import numpy as np
import settings
import torch
from typing import List, Tuple, Optional

class AbstractSampler:
    """
    """

    def __init__(self, sampled_points, sampled_cross_sections, integrand, basepath, n_dims,
                  channel_number, total_channel_count,single_channel=True):
        self.basepath = basepath
        self.n_dims = n_dims
        self.sampled_points = sampled_points
        self.sampled_cross_sections = sampled_cross_sections
        self.single_channel = single_channel
        self.channel_selection_dim = settings.CHANNEL_SELECTION_DIM
        self.total_channel_count = total_channel_count
        self.integrand = integrand
        self.model = None
        self.channel_number = channel_number
        self.device = "cuda" if settings.USE_CUDA else "cpu"

        if self.device == "cuda":
            "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.basepath, exist_ok=True)

    @abstractmethod
    def sample(self, n_samples):
        pass

    @abstractmethod
    def integrate(self, sample_size):
        pass


    def plot_dims(self, n_points=None):
        n_rows = (self.n_dims + 3) // 4
        fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(20, 5 * n_rows))
        
        axes = axes.flatten()
        if n_points is None:
            n_points = len(self.sampled_points)
        samples = self.sample(n_points)

        for dim in range(self.n_dims):
            ax = axes[dim]
            bins = np.linspace(-5, 4, 50)
            ax.hist(self.sampled_points[:,dim],weights=self.sampled_cross_sections, histtype="step", label="training data", bins=50, density=True)
            # ax.hist(x[:,dim],weights=y, histtype="step", label="training data", bins=50, density=True)
            ax.hist(samples[:,dim],  histtype="step", label="generated", bins=50, density=True)
            ax.set_title(f'Dimension {dim}')
            ax.legend()

        for j in range(self.n_dims, len(axes)):
            fig.delaxes(axes[j])
        
        # fig.suptitle(f'Histograms for Each dim for channel {channel}', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(self.basepath+"/phase_space_distrib.png")
        plt.close()

    
    def _save_model(self, flow, loss: float, losses: List[float], dir_path="ERROR"):
        os.makedirs(dir_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(dir_path, "best_model.pth")
        torch.save(flow.state_dict(), model_path)
        
        # Save loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f"Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plot_path = os.path.join(dir_path, "loss_plot.png")
        plt.savefig(plot_path)
        plt.close()