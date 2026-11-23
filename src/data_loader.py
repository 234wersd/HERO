from .dataset import (
    CoraCocitationDataset,
    CiteseerCocitationDataset,
    PubmedCocitationDataset,
    CoraCoauthorshipDataset,
    DBLPCoauthorshipDataset,
    ZooDataset,
    NewsDataset,
    MushroomDataset,
    NTU2012Dataset,
    ModelNet40Dataset,
    yelpRestaurantDataset,
)


class DatasetLoader(object):
    def __init__(self):
        pass

    def load(self, dataset_name: str = "cora", data_dir=None):
        if dataset_name == "cora":
            return CoraCocitationDataset(data_dir=data_dir)
        elif dataset_name == "citeseer":
            return CiteseerCocitationDataset(data_dir=data_dir)
        elif dataset_name == "pubmed":
            return PubmedCocitationDataset(data_dir=data_dir)
        elif dataset_name == "cora_coauthor":
            return CoraCoauthorshipDataset(data_dir=data_dir)
        elif dataset_name == "dblp_coauthor":
            return DBLPCoauthorshipDataset(data_dir=data_dir)
        elif dataset_name == "zoo":
            return ZooDataset(data_dir=data_dir)
        elif dataset_name == "20newsW100":
            return NewsDataset(data_dir=data_dir)
        elif dataset_name == "Mushroom":
            return MushroomDataset(data_dir=data_dir)
        elif dataset_name == "NTU2012":
            return NTU2012Dataset(data_dir=data_dir)
        elif dataset_name == "ModelNet40":
            return ModelNet40Dataset(data_dir=data_dir)
        elif dataset_name == "yelpRestaurant":
            return yelpRestaurantDataset(data_dir=data_dir)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
