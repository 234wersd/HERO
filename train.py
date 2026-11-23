from src.config import parse_args


DEFAULT_PARAMETERS = {
    "dataset": "cora",
    "device": 0,
    "seeds": [0],
    "emb_dim": 512,
    "lr": 5e-4,
    "finetune_lr_scale": 1.0,
    "weight_decay": 5e-4,
    "pretrain_epochs": 180,
    "finetune_epochs": 120,
    "feat_mask_rate": 0.4,
    "edge_drop_rate": 0.4,
}


def main():
    args = parse_args(defaults=DEFAULT_PARAMETERS)
    from src.training import main as train

    train(parsed_args=args)


if __name__ == "__main__":
    main()
