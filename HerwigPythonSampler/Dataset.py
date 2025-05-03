import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from madnis.nn import Flow


class PhaseSpaceDataset(Dataset):
    """Dataset for phase space points and their cross sections."""
    
    def __init__(self, phase_space_points, cross_sections, device="cuda"):
        """
        Initialize the dataset with phase space points and cross sections.
        
        Args:
            phase_space_points (np.ndarray): Array of phase space points
            cross_sections (np.ndarray): Array of cross section values corresponding to each point
            device (str): Device to store tensors on ("cuda" or "cpu")
        """
        self.device = device
        self.phase_space = torch.tensor(phase_space_points, dtype=torch.float32).to(device)
        self.cross_sections = torch.tensor(cross_sections, dtype=torch.float32).to(device)
        self.weights = self.cross_sections / self.cross_sections.sum()
        self.total_cross_section = self.cross_sections.sum()
        
    def __len__(self):
        return len(self.phase_space)
    
    def __getitem__(self, idx):
        return self.phase_space[idx], self.weights[idx]


class ChannelDataPreprocessor:
    """Preprocessor for splitting data into channels."""
    
    def __init__(self, n_channels):
        """
        Initialize the preprocessor with the number of channels.
        
        Args:
            n_channels (int): Number of channels to split the data into
        """
        self.n_channels = n_channels
    
    def split_by_channel(self, phase_space_points, cross_sections, channel_selection_dim=1):
        """
        Split data into separate channels based on a column in the phase space points.
        
        Args:
            phase_space_points (np.ndarray): Array of phase space points
            cross_sections (np.ndarray): Array of cross section values
            channel_selection_dim (int): Column index that determines the channel
            
        Returns:
            tuple: Lists of phase space points and cross sections for each channel
        """
        # Extract the channel information
        channels = phase_space_points[:, channel_selection_dim]
        # Remove the channel column from phase space points
        phase_space_points = np.delete(phase_space_points, channel_selection_dim, axis=1)
        
        # Scale and convert to integer channel indices
        channels = channels * self.n_channels
        channels = channels.astype(int)
        
        # Initialize lists to hold data for each channel
        channel_phase_space = [[] for _ in range(self.n_channels)]
        channel_cross_sections = [[] for _ in range(self.n_channels)]
        
        # Split data by channel
        for i, point in enumerate(phase_space_points):
            channel = channels[i]
            if 0 <= channel < self.n_channels:  # Ensure valid channel index
                channel_phase_space[channel].append(point)
                channel_cross_sections[channel].append(cross_sections[i])
        
        # Convert lists to numpy arrays
        channel_phase_space = [np.array(points) for points in channel_phase_space]
        channel_cross_sections = [np.array(xs) for xs in channel_cross_sections]
        return channel_phase_space, channel_cross_sections
    
    def get_datasets(self, phase_space_points, cross_sections, channel_selection_dim=1):
        """
        Get the dataset for each channel.
        
        Args:
            phase_space_points (np.ndarray): Array of phase space points
            cross_sections (np.ndarray): Array of cross section values
            channel_selection_dim (int): Column index that determines the channel
            
        Returns:
            list: List of datasets for each channel
        """
        channel_phase_space, channel_cross_sections = self.split_by_channel(
            phase_space_points, cross_sections, channel_selection_dim
        )
        
        datasets = []
        for i in range(self.n_channels):
            dataset = PhaseSpaceDataset(channel_phase_space[i], channel_cross_sections[i])
            datasets.append(dataset)
        
        return datasets