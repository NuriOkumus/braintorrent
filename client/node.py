import io
import itertools
import logging
import threading
import time
from collections import deque

import requests
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from client import config, data
from client.model import MnistCNN

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("braintorrent")


class BrainTorrentNode:
    """Implements Algorithm 1 from the BrainTorrent paper for a single client."""

    def __init__(self) -> None:
        self.id = config.CLIENT_ID
        self.model = MnistCNN()
        self.lock = threading.RLock()

        # v^i: this client's own version, incremented every time it fine-tunes.
        self.version = 0
        # For every peer j, the last version of C_j's model this client has already merged.
        self.peer_last_used_version = {peer: 0 for peer in config.PEERS}

        self.train_loader = data.client_train_loader()
        self.n_samples = len(self.train_loader.dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        eval_set = data.load_test_set()
        eval_subset = Subset(eval_set, list(range(200)))
        self._eval_loader = DataLoader(eval_subset, batch_size=100)

        self.round_count = 0
        self.last_acc = None

        # Observability for the dashboard: a bounded, sequence-numbered event log
        # (so a poller can ask for "everything after seq N") and a round-by-round
        # history for charting version/accuracy over time.
        self._seq = itertools.count()
        self.events = deque(maxlen=500)
        self.history = deque(maxlen=200)

    def _log(self, event_type: str, **fields) -> None:
        with self.lock:
            self.events.append({"seq": next(self._seq), "ts": time.time(), "type": event_type, **fields})

    def snapshot(self) -> tuple[bytes, int, int]:
        """Thread-safe consistent (state_dict bytes, version, n_samples), for GET /weights."""
        with self.lock:
            buf = io.BytesIO()
            torch.save(self.model.state_dict(), buf)
            return buf.getvalue(), self.version, self.n_samples

    def status(self) -> dict:
        """Snapshot for GET /status — everything the dashboard needs about this node."""
        with self.lock:
            return {
                "id": self.id,
                "version": self.version,
                "n_samples": self.n_samples,
                "round_count": self.round_count,
                "last_acc": self.last_acc,
                "peers": list(config.PEERS),
                "peer_last_used_version": dict(self.peer_last_used_version),
                "history": list(self.history),
            }

    def _ping(self, peer: str) -> int | None:
        try:
            r = requests.get(f"http://{peer}/ping", timeout=config.PEER_TIMEOUT_SECONDS)
            r.raise_for_status()
            version = int(r.json()["version"])
            self._log("ping", peer=peer, ok=True, version=version)
            return version
        except requests.RequestException:
            log.info("peer %s unreachable, skipping this round", peer)
            self._log("ping", peer=peer, ok=False)
            return None

    def _pull_weights(self, peer: str):
        try:
            r = requests.get(f"http://{peer}/weights", timeout=config.PEER_TIMEOUT_SECONDS)
            r.raise_for_status()
            state_dict = torch.load(io.BytesIO(r.content), map_location="cpu")
            version = int(r.headers["X-Version"])
            n_samples = int(r.headers["X-Num-Samples"])
            self._log("pull_weights", peer=peer, version=version, n_samples=n_samples)
            return state_dict, version, n_samples
        except (requests.RequestException, KeyError, RuntimeError) as exc:
            log.warning("failed to pull weights from %s: %s", peer, exc)
            self._log("pull_weights_failed", peer=peer, error=str(exc))
            return None

    def run_round(self) -> None:
        """One BrainTorrent round with this node as the initiating client C_i."""
        self.round_count += 1
        self._log("round_start", round=self.round_count)

        # Step 1 (ping_request): ask every peer for its current version.
        current_versions = {}
        for peer in config.PEERS:
            v = self._ping(peer)
            if v is not None:
                current_versions[peer] = v

        # Step 2: only pull weights from peers whose version is newer than what we last merged.
        contributions = []  # list of (state_dict, n_samples)
        newly_merged_versions = {}
        for peer, v_new in current_versions.items():
            if v_new > self.peer_last_used_version.get(peer, 0):
                pulled = self._pull_weights(peer)
                if pulled is None:
                    continue
                state_dict, version, n_samples = pulled
                contributions.append((state_dict, n_samples))
                newly_merged_versions[peer] = version

        with self.lock:
            # Step 3: weighted-average merge over the participating subset only
            # (own weights + only the peers that actually responded this round),
            # so the coefficients sum to exactly 1.
            own_state = self.model.state_dict()
            total_samples = self.n_samples + sum(n for _, n in contributions)
            merged = {k: v.clone() * (self.n_samples / total_samples) for k, v in own_state.items()}
            for state_dict, n_samples in contributions:
                weight = n_samples / total_samples
                for k in merged:
                    merged[k] += state_dict[k] * weight
            self.model.load_state_dict(merged)
            self._log(
                "merge",
                peers=list(newly_merged_versions.keys()),
                total_samples=total_samples,
                own_weight=self.n_samples / total_samples,
            )

            # Step 4: fine-tune the merged model on this client's local data.
            self._local_train()

            # Record which peer versions we've now incorporated, then bump our own version.
            self.peer_last_used_version.update(newly_merged_versions)
            self.version += 1

            acc = self._quick_eval()
            self.last_acc = acc
            self.history.append(
                {"round": self.round_count, "version": self.version, "acc": acc, "ts": time.time()}
            )

        self._log(
            "round_complete",
            round=self.round_count,
            version=self.version,
            acc=acc,
            peers=list(newly_merged_versions.keys()),
        )
        log.info(
            "round=%d merged_peers=%s own_version=%d eval_acc=%.4f",
            self.round_count,
            list(newly_merged_versions),
            self.version,
            acc,
        )

    def _local_train(self) -> None:
        self.model.train()
        for _ in range(config.LOCAL_EPOCHS):
            for images, labels in self.train_loader:
                self.optimizer.zero_grad()
                loss = F.cross_entropy(self.model(images), labels)
                loss.backward()
                self.optimizer.step()

    def _quick_eval(self) -> float:
        """Accuracy on a fixed 200-image test subset — cheap signal for round-by-round logs."""
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self._eval_loader:
                preds = self.model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total
