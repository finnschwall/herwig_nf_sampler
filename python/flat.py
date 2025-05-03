import numpy as np
import base
import matplotlib.pyplot as plt
import time

class FlatSampler(base.BaseSampler):

    def statistics(self, time, result, x):
        zero_threshold = 100

        avg_cross_section = np.mean(result)
        zero_cross_sections = np.sum(result < avg_cross_section/zero_threshold)

        



        channels = x[:,1]
        channels = channels*self.dims
        channels = channels.astype(int)

        cross_sections_subarrays = [[] for _ in range(self.dims)]
        channel_subarrays = [[] for _ in range(self.dims)]
        for i, point in enumerate(x):
            channel = channels[i]
            channel_subarrays[channel].append(point[1:])
            cross_sections_subarrays[channel].append(result[i])
        channel_subarrays = [np.array(subarray) for subarray in channel_subarrays]
        cross_sections_subarrays = [np.array(subarray) for subarray in cross_sections_subarrays]

        zero_percentage = []
        cross_sec_percentage = []
        tot_cross_section = np.sum(result)
        for i in range(self.dims):
            # print(f"Channel {i}: {len(channel_subarrays[i])} points, {np.sum(cross_sections_subarrays[i]):.2e} tot. cross section")
            if len(cross_sections_subarrays[i]) > 0:
                avg_cross_section = np.mean(cross_sections_subarrays[i])
                zero_cross_sections = np.sum(cross_sections_subarrays[i] < avg_cross_section/zero_threshold)
                # print(f"\tMax: {int(np.max(cross_sections_subarrays[i]))}, Avg: {avg_cross_section:.2f}")
                # print(f"\tZero: {zero_cross_sections/len(cross_sections_subarrays[i])*100:.2f}%, {zero_cross_sections}/{len(cross_sections_subarrays[i])}")
                # print(f"\tTotal: {np.sum(cross_sections_subarrays[i])/tot_cross_section*100:.2f}%")
                cross_sec_percentage.append(np.sum(cross_sections_subarrays[i])/tot_cross_section*100)
                zero_percentage.append(zero_cross_sections/len(cross_sections_subarrays[i])*100)
        
        cross_sec_percentage = np.array(cross_sec_percentage)
        
        count_cross_per_1_per = np.sum(cross_sec_percentage > 1)
        expected_per = 100/self.dims
        count_exp = np.sum(cross_sec_percentage > expected_per)
        count_exp_half = np.sum(cross_sec_percentage > expected_per/2)
        count_exp_quarter = np.sum(cross_sec_percentage > expected_per/4)
        count_exp_tenth = np.sum(cross_sec_percentage > expected_per/10)
        # print(f"Count > 1%: {count_cross_per_1_per}/{self.dims}, {count_cross_per_1_per/self.dims*100:.2f}%")
        print(f"Count > exp ({expected_per:.2f})%: {count_exp}/{self.dims}, {count_exp/self.dims*100:.2f}%")
        print(f"Count > exp/2 ({expected_per/2:.2f})%: {count_exp_half}/{self.dims}, {count_exp_half/self.dims*100:.2f}%")
        print(f"Count > exp/4 ({expected_per/4:.2f})%: {count_exp_quarter}/{self.dims}, {count_exp_quarter/self.dims*100:.2f}%")
        print(f"Count > exp/10 ({expected_per/10:.2f})%: {count_exp_tenth}/{self.dims}, {count_exp_tenth/self.dims*100:.2f}%")

        print()
        print(f"Max cross section: {np.max(result):.2e}, Avg cross section: {avg_cross_section:.2e}, Std. Err.: {np.var(result)**0.5:.2e}")
        print(f"Zero cross sections: {zero_cross_sections/len(result)*100:.2f}%, {zero_cross_sections}/{len(result)}, ")
        
        print(f"Time for {len(x)} points: {time:.3f} seconds\nTime per point: {time/(len(x)/1000):.8f} ms")
        

        try:
            with open("timings.csv", "r") as f:
                pass
        except FileNotFoundError:
            with open("timings.csv", "w") as f:
                f.write("n_points,time,dims,count_exp,count_exp_half,count_exp_quarter,count_exp_tenth,count_zero,avg_cross_section,std_cross_section\n")

        with open("timings.csv", "a") as f:
            f.write(f"{len(x)},{time:.3f},{self.dims},{count_exp},{count_exp_half},{count_exp_quarter},{count_exp_tenth},{zero_cross_sections/len(result)*100:.2f},{avg_cross_section:.2f},{np.var(result)**0.5:.2e}\n")

        ps_points = x
        cross_section = result

        n_dims = ps_points.shape[1]  # Number of dimensions

        # Calculate the number of rows needed
        n_rows = (n_dims + 3) // 4  # Integer division to determine the number of rows

        # Create a grid of histograms
        fig, axes = plt.subplots(nrows=n_rows, ncols=4, figsize=(20, 5 * n_rows))

        # Flatten the axes array for easy iteration
        axes = axes.flatten()

        for dim in range(n_dims):
            ax = axes[dim]
            ax.hist(ps_points[:, dim], weights=cross_section, bins=200)
            ax.set_title(f'Dimension {dim}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')

        # Hide any empty subplots
        for j in range(n_dims, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f'Histograms for Each dim', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


    def matrix_callback(self, x, channel=None):
        start_time = time.time()
        matrix_list = x
        result = self.cpp_matrix_element_callback(matrix_list)
        self.sampled_points.extend(matrix_list)
        self.sampled_cross_sections.extend(result)
        tot_time = time.time() - start_time

        # self.statistics(tot_time, result, matrix_list)
        return result

    def train(self):
        elements_per_batch = 1000
        x = np.random.rand(elements_per_batch, self.dims+1)
        # x[:, 0] = 0.3
        self.matrix_callback(x)
        # self.plot_phase_space(2,1)
        # print(self.sampled_points)
        #save the sampled points and cross sections

        # np.save("sampled_points.npy", self.sampled_points)
        # np.save("sampled_cross_sections.npy", self.sampled_cross_sections)
    
        # self.weighted_hist(2)
        return 0
    
    def integrate(self, sample_size):
        vec = np.random.rand(sample_size, self.dims)
        result = self.cpp_matrix_element_callback(vec)
        error = np.sqrt(np.std(result))
        return np.mean(result), error, vec
