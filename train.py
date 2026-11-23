from src.config import parse_args


def main():
    args = parse_args()
    from src.training import main as train

    train(parsed_args=args)


if __name__ == "__main__":
    main()
