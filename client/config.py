import os


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


CLIENT_ID = os.environ["CLIENT_ID"]

# Comma-separated list of "host:port" for every other client in the environment,
# e.g. "client-2:5000,client-3:5000". Compose service names act as hostnames.
PEERS = [p for p in os.environ.get("PEERS", "").split(",") if p]

PORT = _int_env("PORT", 5000)

# Total number of clients in the environment (this one + PEERS), used to shard MNIST.
NUM_CLIENTS = _int_env("NUM_CLIENTS", len(PEERS) + 1)

# Local fine-tuning: fixed epoch count per round (paper fixes this to avoid overfitting).
LOCAL_EPOCHS = _int_env("LOCAL_EPOCHS", 1)
BATCH_SIZE = _int_env("BATCH_SIZE", 32)
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.001"))

# Random delay range (seconds) between rounds a client initiates on its own.
ROUND_INTERVAL_MIN = _int_env("ROUND_INTERVAL_MIN", 5)
ROUND_INTERVAL_MAX = _int_env("ROUND_INTERVAL_MAX", 15)

# HTTP timeout when pinging/pulling weights from a peer; a down peer is skipped, not fatal.
PEER_TIMEOUT_SECONDS = float(os.environ.get("PEER_TIMEOUT_SECONDS", "3.0"))

DATA_DIR = os.environ.get("DATA_DIR", "/data")
SEED = _int_env("SEED", 42)
