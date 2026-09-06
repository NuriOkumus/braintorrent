"""Pulls current weights from every running client and reports the 'aggregated model'
accuracy — the same quantity the paper's Tables 1/2/4 report in the 'Aggregated Model'
column: a sample-weighted average of all clients' models, evaluated on the held-out test set.
"""

import argparse
import io
import sys

import requests
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from client.model import MnistCNN

TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


def fetch_client(host: str):
    r = requests.get(f"http://{host}/weights", timeout=10)
    r.raise_for_status()
    state_dict = torch.load(io.BytesIO(r.content), map_location="cpu")
    n_samples = int(r.headers["X-Num-Samples"])
    version = int(r.headers["X-Version"])
    return state_dict, n_samples, version


def aggregate(clients: list[tuple[dict, int, int]]) -> dict:
    total = sum(n for _, n, _ in clients)
    merged = None
    for state_dict, n_samples, _ in clients:
        weight = n_samples / total
        if merged is None:
            merged = {k: v.clone() * weight for k, v in state_dict.items()}
        else:
            for k in merged:
                merged[k] += state_dict[k] * weight
    return merged


def evaluate(model: MnistCNN) -> float:
    # /data already has MNIST baked in at image-build time (see Dockerfile) — reuse it so
    # this doesn't re-download on every eval run.
    test_set = datasets.MNIST("/data", train=False, download=True, transform=TRANSFORM)
    loader = DataLoader(test_set, batch_size=256)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hosts",
        default="localhost:5000,localhost:5001,localhost:5002,localhost:5003,localhost:5004",
        help="comma-separated host:port list of running clients",
    )
    args = parser.parse_args()
    hosts = args.hosts.split(",")

    reachable = []
    for host in hosts:
        try:
            reachable.append((host, fetch_client(host)))
        except requests.RequestException as exc:
            print(f"skipping {host}: {exc}", file=sys.stderr)

    if not reachable:
        print("no reachable clients", file=sys.stderr)
        sys.exit(1)

    for host, (_, n_samples, version) in reachable:
        print(f"{host}: version={version} n_samples={n_samples}")

    merged_state = aggregate([result for _, result in reachable])
    model = MnistCNN()
    model.load_state_dict(merged_state)
    acc = evaluate(model)
    print(f"\naggregated model test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
