
from abc import abstractmethod
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import scipy.io
from typing import List, Tuple, Literal, Optional

class TorchbciDataset(Dataset):
    """Base class for datasets in torchbci.
    
    Args:
        data_dir (str): Directory containing the dataset.
        label_dir (str, optional): Directory containing the labels. Defaults to None.
    """
    def __init__(self, data_dir: str, label_dir: Optional[str] = None):      
        super().__init__()
        self.data_dir = data_dir
        self.label_dir = label_dir

    def _from_numpy(self, npy_file: str) -> np.ndarray:
        """Load dataset from a NumPy file.

        Args:
            npy_file (str): Path to the NumPy file.
        """
        data = np.load(npy_file)
        return data

    def _from_numpy_memmap(
        self,
        npy_file: str,
        shape: Tuple[int],
        dtype: np.dtype,
        order: 'Literal["C", "F", "A", "K"]' = 'C'
    ) -> np.memmap:
        """Load dataset from a NumPy memmap file.

        Args:
            npy_file (str): Path to the NumPy memmap file.
            shape (Tuple[int]): Shape of the memmap array.
            dtype (np.dtype, optional): Data type of the memmap array. Defaults to np.float32.
            order (Literal["C", "F", "A", "K"], optional): Memory layout order. Defaults to 'C'.
        """
        data = np.memmap(npy_file, mode='r', shape=shape, dtype=dtype, order=order)
        return data
    
    def _from_torch(self, tensor_file: str) -> torch.Tensor:
        """Load dataset from a PyTorch tensor file.

        Args:
            tensor_file (str): Path to the PyTorch tensor file.
        """
        data = torch.load(tensor_file)
        return data
    
    def _from_matlab(self, mat_file: str, variable_name: str) -> np.ndarray:
        """Load dataset from a MATLAB .mat file.

        Args:
            mat_file (str): Path to the MATLAB .mat file.
            variable_name (str): Name of the variable to extract from the .mat file.
        """
        mat = scipy.io.loadmat(mat_file)
        data = mat[variable_name]
        return data
    
    @abstractmethod
    def from_custom(self):
        """API for loading dataset from a custom format. This method should be overridden by subclasses.
        """
        pass

    @abstractmethod
    def visualize(self):
        """API for visualizing the dataset. This method should be overridden by subclasses.
        """        
        pass

    @abstractmethod
    def preprocess(self):
        """API for preprocessing the data. This method should be overridden by subclasses.
        """
        pass

class DataStreamer(DataLoader):
    """Base class for data streaming in torchbci.

    Args:
        DataLoader (torch.utils.data.DataLoader): Inherits from PyTorch's DataLoader class.
    """
    def __init__(self, dataset: Dataset):
        """Initialize the DataStreamer.

        Args:
            dataset (Dataset): The dataset to stream data from.
        """        
        self.dataset = dataset
        super().__init__(dataset=self.dataset)

    @abstractmethod
    def stream(self):
        """API for streaming data. This method should be overridden by subclasses.
        """
        pass
