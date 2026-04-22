"""CLI entry for `python -m coart.vae` / `torchrun ... -m coart.vae`."""
from .config import parse_args
from .train import train


def main() -> None:
    cfg = parse_args()
    train(cfg)


if __name__ == "__main__":
    main()
