import numpy as np
import matplotlib.pyplot as plt


class BaseSampler:
    def __init__(self):
        self.cpp_matrix_element_callback = None
        self.integrator = None
        self.dims = None
        self.sampled_points = []
        self.sampled_cross_sections = []
        self.channel_selection_dim = 1

    def setup_base(self, python_sampler, n_dims, diagram_dimension):
        n_dims = n_dims - 1 #first random number chooses the channel
        print(f"Training in {n_dims} dimensions in {diagram_dimension} channels")
        self.dims = n_dims
        self.cpp_matrix_element_callback = python_sampler.dSigDRMatrix
        self.python_sampler = python_sampler
        self.diagram_dimension = diagram_dimension

    def matrix_callback(self, x, channel=None):
        pass

    def train(self):
        pass

    def plot_phase_space(self, d1=0, d2=1):
        psp = np.array(self.sampled_points[1:])
        weights = np.array(self.sampled_cross_sections[1:])[:5000]
        psp = psp[:5000]
        psp = psp.T
        mask = weights != 0
        filtered_psp = psp[:, mask]
        filtered_weights = weights[mask]

        plt.scatter(filtered_psp[d1], filtered_psp[d2], c=filtered_weights, cmap='viridis')
        plt.colorbar(label='Weight')
        plt.show()

        # plt.scatter(psp[2], psp[0], c=weights, cmap='viridis')
        # plt.colorbar(label='Weight')
        # plt.show()

    def weighted_hist(self, d=0):
        psp = np.array(self.sampled_points[1:])
        weights = np.array(self.sampled_cross_sections[1:])
        psp = psp.T
        plt.hist(psp[d], weights=weights, bins=100)

        # Display the plot
        plt.xlabel('Points')
        plt.ylabel('Weighted Frequency')
        plt.title('Weighted Histogram')
        plt.show()
    
    def integrate(self, sample_size):
        if not self.flat_sampling:
            return self.integrator.integrate(sample_size)
        else:
            vec = np.random.rand(sample_size, self.dims)
            result = self.cpp_matrix_element_callback(vec)
            error = np.sqrt(np.std(result))
            return np.mean(result), error

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

    def save(self, filepath):
        pass

    def load(self, python_sampler, n_dims):
        pass

    def generate(self, n_samples):
        samples = self.integrator.sample(n_samples)
        ret_tuple = (samples.x.tolist()[0], samples.func_vals[0], samples.weights[0])
        return ret_tuple