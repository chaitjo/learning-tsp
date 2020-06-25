from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist, squareform

from problems.tsp.state_tsp import StateTSP
from utils.beam_search import beam_search


def nearest_neighbor_graph(nodes, neighbors, knn_strat):
    """Returns k-Nearest Neighbor graph as a **NEGATIVE** adjacency matrix
    """
    num_nodes = len(nodes)
    # If `neighbors` is a percentage, convert to int
    if knn_strat == 'percentage':
        neighbors = int(num_nodes * neighbors)
    
    if neighbors >= num_nodes-1 or neighbors == -1:
        W = np.zeros((num_nodes, num_nodes))
    else:
        # Compute distance matrix
        W_val = squareform(pdist(nodes, metric='euclidean'))
        W = np.ones((num_nodes, num_nodes))
        
        # Determine k-nearest neighbors for each node
        knns = np.argpartition(W_val, kth=neighbors, axis=-1)[:, neighbors::-1]
        # Make connections
        for idx in range(num_nodes):
            W[idx][knns[idx]] = 0
    
    # Remove self-connections
    np.fill_diagonal(W, 1)
    return W


def tour_nodes_to_W(tour_nodes):
    """Computes edge adjacency matrix representation of tour
    """
    num_nodes = len(tour_nodes)
    tour_edges = np.zeros((num_nodes, num_nodes))
    for idx in range(len(tour_nodes) - 1):
        i = tour_nodes[idx]
        j = tour_nodes[idx + 1]
        tour_edges[i][j] = 1
        tour_edges[j][i] = 1
    # Add final connection
    tour_edges[j][tour_nodes[0]] = 1
    tour_edges[tour_nodes[0]][j] = 1
    return tour_edges


class TSP(object):
    """Class representing the Travelling Salesman Problem
    """

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        """Returns TSP tour length for given graph nodes and tour permutations

        Args:
            dataset: graph nodes (torch.Tensor)
            pi: node permutations representing tours (torch.Tensor)

        Returns:
            TSP tour length, None
        """
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour:\n{}\n{}".format(dataset, pi)

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(nodes, graph, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        """Method to call beam search, given TSP samples and a model
        """

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(nodes, graph)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            nodes, graph, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)
    
    
class TSPSL(TSP):
    """Class representing the Travelling Salesman Problem, trained with Supervised Learning
    """

    NAME = 'tspsl'


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, min_size=20, max_size=50, batch_size=128,
                 num_samples=128000, offset=0, distribution=None, neighbors=20, 
                 knn_strat=None, supervised=False, nar=False):
        """Class representing a PyTorch dataset of TSP instances, which is fed to a dataloader

        Args:
            filename: File path to read from (for SL)
            min_size: Minimum TSP size to generate (for RL)
            max_size: Maximum TSP size to generate (for RL)
            batch_size: Batch size for data loading/batching
            num_samples: Total number of samples in dataset
            offset: Offset for loading from file
            distribution: Data distribution for generation (unused)
            neighbors: Number of neighbors for k-NN graph computation
            knn_strat: Strategy for computing k-NN graphs ('percentage'/'standard')
            supervised: Flag to enable supervised learning
            nar: Flag to indicate Non-autoregressive decoding scheme, which uses edge-level groundtruth

        Notes:
            `batch_size` is important to fix across dataset and dataloader,
            as we are dealing with TSP graphs of variable sizes. To enable
            efficient training without DGL/PyG style sparse graph libraries,
            we ensure that each batch contains dense graphs of the same size.
        """
        super(TSPDataset, self).__init__()

        self.filename = filename
        self.min_size = min_size
        self.max_size = max_size
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.offset = offset
        self.distribution = distribution
        self.neighbors = neighbors
        self.knn_strat = knn_strat
        self.supervised = supervised
        self.nar = nar

        # Loading from file (usually used for Supervised Learning or evaluation)
        if filename is not None:
            self.nodes_coords = []
            self.tour_nodes = []

            print('\nLoading from {}...'.format(filename))
            for line in tqdm(open(filename, "r").readlines()[offset:offset+num_samples], ascii=True):
                line = line.split(" ")
                num_nodes = int(line.index('output')//2)
                self.nodes_coords.append(
                    [[float(line[idx]), float(line[idx + 1])] for idx in range(0, 2 * num_nodes, 2)]
                )

                if self.supervised:
                    # Convert tour nodes to required format
                    # Don't add final connection for tour/cycle
                    tour_nodes = [int(node) - 1 for node in line[line.index('output') + 1:-1]][:-1]
                    self.tour_nodes.append(tour_nodes)

        # Generating random TSP samples (usually used for Reinforcement Learning)
        else:
            # Sample points randomly in [0, 1] square
            self.nodes_coords = []

            print('\nGenerating {} samples of TSP{}-{}...'.format(num_samples, min_size, max_size))
            for _ in tqdm(range(num_samples//batch_size), ascii=True):
                # Each mini-batch contains graphs of the same size
                # Graph size is sampled randomly between min and max size
                num_nodes = np.random.randint(low=min_size, high=max_size+1)
                self.nodes_coords += list(np.random.random([batch_size, num_nodes, 2]))
        
        self.size = len(self.nodes_coords)
        assert self.size % batch_size == 0, \
            "Number of samples ({}) must be divisible by batch size ({})".format(self.size, batch_size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        nodes = self.nodes_coords[idx]
        item = {
            'nodes': torch.FloatTensor(nodes),
            'graph': torch.ByteTensor(nearest_neighbor_graph(nodes, self.neighbors, self.knn_strat))
        }
        if self.supervised:
            # Add groundtruth labels in case of SL
            tour_nodes = self.tour_nodes[idx]
            item['tour_nodes'] = torch.LongTensor(tour_nodes)
            if self.nar:
                # Groundtruth for NAR decoders is the TSP tour in adjacency matrix format
                item['tour_edges'] = torch.LongTensor(tour_nodes_to_W(tour_nodes))

        return item
