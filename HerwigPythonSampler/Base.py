

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
        self.dims = n_dims
        self.cpp_matrix_element_callback = python_sampler.dSigDRMatrix
        self.python_sampler = python_sampler
        self.diagram_dimension = diagram_dimension

    def matrix_callback(self, x, channel=None):
        pass
