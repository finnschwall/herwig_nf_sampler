

class Base:
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
    
    def integrate(self, sample_size):
        if not self.flat_sampling:
            return self.integrator.integrate(sample_size)
        else:
            vec = np.random.rand(sample_size, self.dims)
            result = self.cpp_matrix_element_callback(vec)
            error = np.sqrt(np.std(result))
            return np.mean(result), error



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