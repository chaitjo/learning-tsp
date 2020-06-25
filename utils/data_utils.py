import os
import pickle
import numpy as np

from torch.utils.data import Sampler


def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename


def save_dataset(dataset, filename):

    filedir = os.path.split(filename)[0]

    if not os.path.isdir(filedir):
        os.makedirs(filedir)

    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


def load_dataset(filename):

    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)
    
    
class BatchedRandomSampler(Sampler):
    """Samples elements randomly, while maintaining sequential order within a batch size

    Arguments:
        data_source (Dataset): dataset to sample from
        batch_size (int): batch size 
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size
        
        assert len(data_source) % batch_size == 0, "Number of samples must be divisible by batch size"

    def __iter__(self):
        batch_nums = np.arange(len(self.data_source) // self.batch_size)
        np.random.shuffle(batch_nums)
        idx = []
        for b_idx in batch_nums:
            idx += [self.batch_size*b_idx + s_idx for s_idx in range(self.batch_size)]
        return iter(idx)

    def __len__(self):
        return len(self.data_source)
