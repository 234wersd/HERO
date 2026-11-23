# HERO

HERO: Hyperedge-Aware Expert Routing with Confidence-Guided
Optimization for Attributed Hypergraph Clustering

## Requirements

Python 3.10.

```text
torch==2.4.1
torch-geometric==2.7.0
torch-scatter==2.1.2
torch-sparse==0.6.18
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.7.1
```

## Parameter Settings

Set parameters manually through command-line arguments, for example:

```bash
python train.py --dataset cora-c --emb_dim 512 --lr 5e-4 --seeds 0 1 2 3 4
```
