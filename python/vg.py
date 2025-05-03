import numpy as np
import base
import matplotlib.pyplot as plt
import vegas


class VegasSampler(base.BaseSampler):
    @vegas.rbatchintegrand
    def matrix_callback(self, x, channel=None):
        self.sampled_points.extend(list(x.T))
        # for i in x.T:
        #     self.sampled_points.append(list(i))
        matrix_list = x.T
        result = self.cpp_matrix_element_callback(matrix_list)
        
        self.sampled_cross_sections.extend(result)
        result = np.array(result)
        return result

    def train(self):
        integ = vegas.Integrator([[0, 1]*self.dims])
        result = integ(self.matrix_callback, nitn=10, neval=1000)
        # print(self.sampled_points[:20])
        # print(self.sampled_cross_sections)
        # integ.map.show_grid(30)
        self.plot_phase_space()
        return 0
    
    def integrate(self, sample_size):
        vec = np.random.rand(sample_size, self.dims)
        result = self.cpp_matrix_element_callback(vec)
        error = np.sqrt(np.std(result))
        return np.mean(result), error, vec
