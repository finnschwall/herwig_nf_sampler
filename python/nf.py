import herwig_python

import sys
import os
sys.path.insert(0, '/home/finn/.pyenv/versions/3.10.11/envs/herwig/lib/python3.10/site-packages')
sys.path.append('/mnt/data-slow/herwig/python/madnis')
import madnis
import torch
from madnis.integrator import Integrator, Integrand
import matplotlib.pyplot as plt
import numpy as np


import base

class FlowSampler(base.BaseSampler):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def matrix_callback(self, x, channel=None):
        if channel is None:
            # channel = torch.rand((x.shape[0],), device=self.device)
            channel = torch.zeros((x.shape[0],), device=self.device)
        else:
            channel =channel.float()/ self.diagram_dimension
        # x = torch.cat((channel.unsqueeze(1), x), dim=1)
        x = torch.cat((x[:, :self.channel_selection_dim], channel.unsqueeze(1), x[:, self.channel_selection_dim:]), dim=1)
        matrix_list = x.tolist()
        result = self.cpp_matrix_element_callback(matrix_list)
        result_tensor = torch.tensor(result)
        self.sampled_points.extend(matrix_list)
        self.sampled_cross_sections.extend(result)
        return result_tensor.to(self.device)
    


    def train(self):
        # self.device ="cpu"
        print(f"Using device {self.device}")
        integrand = Integrand(self.matrix_callback, input_dim=self.dims, channel_count=self.diagram_dimension)
        self.integrator = Integrator(integrand, device=self.device, group_channels_in_loss=True, batch_size=40)
        # integrand = Integrand(self.matrix_callback, input_dim=self.dims)
        # self.integrator = Integrator(integrand, device=self.device, flow_kwargs={"uniform_latent":True}, batch_size=8192)
        results = []
        errors = []
        losses = []
        batches = []


        result, error = self.integrator.integrate(10000)
        print(f"Untrained result={result:.5f} +- {error:.5f}, rel. error={error/result*100:.2f}%")
        def progress_callback(status):
            if (status.step + 1) % 1 == 0:
                result, error = self.integrator.integrate(10000)
                print(f"Batch {status.step + 1}: loss={status.loss:.5f}, result={result:.5f} +- {error:.5f},l Rel. Error={error/result*100:.2f}%")
                results.append(result)
                errors.append(error)
                losses.append(status.loss)
                batches.append(status.step + 1)

        batch_count = 10
        
        self.integrator.train(batch_count, progress_callback)
        from pprint import pp
        pp(self.integrator.integration_metrics(100000))

        # self.plot_convergence(num_samples=1000)
        psp = np.array(self.sampled_points)
        weights = np.array(self.sampled_cross_sections)[:5000]
        print(weights)
        psp = psp[:5000]
        psp = psp.T
        mask = weights != 0
        filtered_psp = psp[:, mask]
        filtered_weights = weights[mask]

        # plt.scatter(filtered_psp[0], filtered_psp[1], c=filtered_weights, cmap='viridis')
        # plt.colorbar(label='Weight')
        # plt.show()

        # plt.scatter(psp[2], psp[0], c=weights, cmap='viridis')
        # plt.colorbar(label='Weight')
        # plt.show()

        
        samples = self.integrator.sample(100000)
        bins = np.linspace(0, 1, 30)
        plt.hist(samples.x.cpu()[:,1].numpy(), bins, histtype="step", label="learned", density=True)
        plt.hist(
            samples.x.cpu()[:,1].numpy(),
            bins,
            weights=samples.weights.cpu().numpy(),
            histtype="step",
            label="reweighted",
            density=True
        )
        plt.xlabel("$x_1$")
        plt.xlim(0, 1)
        plt.legend()
        plt.show()


        return 0
    
    def integrate(self, sample_size):
        return self.integrator.integrate(sample_size)

    def plot_convergence(self, num_samples = 20, points_per_sample = 100):
        results = []
        averaged_results = []
        errors = []
        samples = []
        # for i in range(num_samples):
        #     samples.append((i+1) * points_per_sample)
        #     result, error = self.integrate(points_per_sample)
        #     results.append(result)
        #     averaged_results.append(np.mean(results))
            
        #     errors.append(error)
        for i in range(num_samples):
            samples.append((i+1) * points_per_sample)
            result, single_run_error = self.integrate(points_per_sample)
            results.append(result)
            
            # Calculate mean of all results so far
            current_mean = np.mean(results)
            averaged_results.append(current_mean)
            
            # For proper error propagation:
            if len(results) > 1:
                # Statistical error of the mean (more accurate as sample size increases)
                propagated_error = np.std(results, ddof=1) / np.sqrt(len(results))
            else:
                propagated_error = single_run_error
                
            errors.append(propagated_error)
        # plt.errorbar(samples, averaged_results, yerr=errors, fmt='-o')
        # plt.xlabel('Samples')
        # plt.ylabel('Integration Result')
        # plt.show()
        plt.figure(figsize=(10, 6))

        # Plot the main line
        line, = plt.plot(samples, results, 'b-', linewidth=2)

        # Create the shaded error band
        # plt.fill_between(samples, 
        #                 np.array(averaged_results) - np.array(errors),
        #                 np.array(averaged_results) + np.array(errors),
        #                 alpha=0.3, color=line.get_color())

        # Add labels and title
        plt.xlabel('Number of Points', fontsize=12)
        plt.ylabel('Integration Result', fontsize=12)
        plt.title('Monte Carlo Integration Convergence', fontsize=14)

        true_value =  38.0975
        true_error = 1.53624
        # Add the true value as a horizontal line
        plt.axhline(y=true_value, color='r', linestyle='-', linewidth=2, label='True Value')

        # Add the true value error band
        plt.axhspan(true_value - true_error, true_value + true_error, 
                    alpha=0.2, color='r', label='True Value Error')

        # Optional: Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)

        # Optional: Improve overall appearance
        plt.tight_layout()
        plt.show()


    def plot_results(self, batches, results, errors, losses):
        fig, ax1 = plt.subplots()
        ax1.errorbar(batches, results, yerr=errors, fmt='-o', label='Integration Result', color='b')
        ax1.set_xlabel('Batches')
        ax1.set_ylabel('Integration Result', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax2 = ax1.twinx()
        ax2.plot(batches, losses, label='Loss', color='r')
        ax2.set_ylabel('Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        fig.tight_layout()
        plt.savefig('convergence.png')

    def save_flow(self, filepath):
        if self.integrator is not None:
            torch.save(self.integrator.state_dict(), filepath)
            print(f"Flow saved to {filepath}")
        else:
            print("No integrator to save")

    def load_flow(self, python_sampler, n_dims):
        filepath = "flow.pt"
        print(f"Loading flow from {filepath}")
        self.dims = n_dims
        self.cpp_matrix_element_callback = python_sampler.dSigDRMatrix
        self.integrator = Integrator(self.matrix_callback, dims=self.dims)
        self.integrator.load_state_dict(torch.load(filepath))
        print(f"Flow loaded from {filepath}")

    def generate(self, n_samples):
        samples = self.integrator.sample(n_samples)
        ret_tuple = (samples.x.tolist()[0], samples.func_vals[0], samples.weights[0])
        return ret_tuple