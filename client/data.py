import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from client import config

_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def _client_index() -> int:
    """Numeric shard index derived from CLIENT_ID (expects "0", "1", ... or "client-N")."""
    digits = "".join(ch for ch in config.CLIENT_ID if ch.isdigit())
    if not digits:
        raise ValueError(f"CLIENT_ID must contain a shard index, got {config.CLIENT_ID!r}")
    return int(digits)


def load_full_train_set() -> datasets.MNIST:
    return datasets.MNIST(config.DATA_DIR, train=True, download=True, transform=_TRANSFORM)


def load_test_set() -> datasets.MNIST:
    return datasets.MNIST(config.DATA_DIR, train=False, download=True, transform=_TRANSFORM)


def client_train_loader() -> DataLoader:
    """This client's disjoint shard of MNIST train — deterministic given SEED/NUM_CLIENTS."""
    full = load_full_train_set()
    generator = torch.Generator().manual_seed(config.SEED)
    order = torch.randperm(len(full), generator=generator).tolist()

    shard_size = len(full) // config.NUM_CLIENTS
    idx = _client_index()
    start = idx * shard_size
    # last shard absorbs the remainder so every sample is used exactly once
    end = len(full) if idx == config.NUM_CLIENTS - 1 else start + shard_size
    shard = Subset(full, order[start:end])

    return DataLoader(shard, batch_size=config.BATCH_SIZE, shuffle=True)
