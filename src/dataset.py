import os.path as osp
from pathlib import Path
import pickle

import torch
from torch_scatter import scatter_add


class BaseDataset(object):
    def __init__(self, type: str, name: str, device: str = "cpu", data_dir=None):
        self.type = type
        self.name = name
        self.device = device
        data_dir = (
            Path(data_dir)
            if data_dir is not None
            else Path(__file__).resolve().parents[1] / "data" / "dataset"
        )
        if self.type in ["cocitation", "coauthorship"]:
            self.dataset_dir = osp.join(data_dir, self.type, self.name)
        else:
            self.dataset_dir = osp.join(data_dir, self.name)
        self.split_dir = osp.join(self.dataset_dir, "splits")

        self.load_dataset()
        self.preprocess_dataset()

    def load_dataset(self):
        with open(osp.join(self.dataset_dir, "features.pickle"), "rb") as f:
            self.features = pickle.load(f)
        with open(osp.join(self.dataset_dir, "hypergraph.pickle"), "rb") as f:
            self.hypergraph = pickle.load(f)
        with open(osp.join(self.dataset_dir, "labels.pickle"), "rb") as f:
            self.labels = pickle.load(f)

    def preprocess_dataset(self):
        edge_set = set(self.hypergraph.keys())
        edge_to_num = {}
        num_to_edge = {}
        num = 0
        for edge in edge_set:
            edge_to_num[edge] = num
            num_to_edge[num] = edge
            num += 1

        incidence_matrix = []
        processed_hypergraph = {}
        for edge in edge_set:
            nodes = self.hypergraph[edge]
            processed_hypergraph[edge_to_num[edge]] = nodes
            for node in nodes:
                incidence_matrix.append([node, edge_to_num[edge]])

        self.processed_hypergraph = processed_hypergraph
        self.features = torch.as_tensor(self.features.toarray())
        self.hyperedge_index = torch.LongTensor(incidence_matrix).T.contiguous()
        self.labels = torch.LongTensor(self.labels)
        self.num_nodes = int(self.hyperedge_index[0].max()) + 1
        self.num_edges = int(self.hyperedge_index[1].max()) + 1
        self.edge_to_num = edge_to_num
        self.num_to_edge = num_to_edge

        weight = torch.ones(self.num_edges)
        Dn = scatter_add(
            weight[self.hyperedge_index[1]],
            self.hyperedge_index[0],
            dim=0,
            dim_size=self.num_nodes,
        )
        De = scatter_add(
            torch.ones(self.hyperedge_index.shape[1]),
            self.hyperedge_index[1],
            dim=0,
            dim_size=self.num_edges,
        )

        self.to(self.device)

    def to(self, device: str):
        self.features = self.features.to(device)
        self.hyperedge_index = self.hyperedge_index.to(device)
        self.labels = self.labels.to(device)
        self.device = device
        return self


class CoraCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("cocitation", "cora", **kwargs)


class CiteseerCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("cocitation", "citeseer", **kwargs)


class PubmedCocitationDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("cocitation", "pubmed", **kwargs)


class CoraCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("coauthorship", "cora", **kwargs)


class DBLPCoauthorshipDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("coauthorship", "dblp", **kwargs)


class ZooDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("etc", "zoo", **kwargs)


class NewsDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("etc", "20newsW100", **kwargs)


class MushroomDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("etc", "Mushroom", **kwargs)


class NTU2012Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("cv", "NTU2012", **kwargs)


class ModelNet40Dataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("cv", "ModelNet40", **kwargs)


class yelpRestaurantDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__("cv", "yelpRestaurant", **kwargs)
