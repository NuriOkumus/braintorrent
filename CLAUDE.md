# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An implementation of *BrainTorrent* (Roy et al., "BrainTorrent: A Peer-to-Peer Environment for
Decentralized Federated Learning", arXiv:1905.06731 — the PDF in the repo root is the source paper),
demonstrated on MNIST digit classification instead of the paper's whole-brain MRI segmentation task.

BrainTorrent is federated learning **without a central server**. In classic federated learning, every
client sends weights to a server which averages and redistributes them. Here, clients talk directly to
each other: each round, one client pings its peers for their current model version, pulls weights only
from peers that are newer than what it last merged, combines those (sample-count-weighted average) with
its own model, fine-tunes on its local shard, and bumps its own version. There is no coordinator process
and no single point of failure — this is Algorithm 1 in the paper, implemented as-is except the model/
dataset are swapped for a small CNN on MNIST.

Each client runs as its own Docker container and is both an HTTP server (exposing its model to peers) and
a background loop (autonomously initiating rounds against its peers). "Client" in the code means a node in
the P2P mesh, not an HTTP client library.

## Commands

```bash
# Build and start all 5 clients (client-0 .. client-4) plus the dashboard, each its own container:
docker compose up --build

# Open the live monitoring dashboard (network diagram, per-client version/version-vector/accuracy
# cards, live event feed of ping/pull_weights/merge/round_complete):
open http://localhost:8080

# Watch a specific client's round-by-round log (version, merged peers, quick eval accuracy):
docker compose logs -f client-0

# Evaluate the current "aggregated model" (sample-weighted average of all clients' weights,
# same quantity as the paper's "Aggregated Model" column in Tables 1/2/4), on the MNIST test set.
# Must run inside a container on the compose network — the host has no torch installed:
docker run --rm --network braintorrent_default -v "$(pwd)/eval.py:/app/eval.py" \
    braintorrent-client-0 python eval.py \
    --hosts client-0:5000,client-1:5000,client-2:5000,client-3:5000,client-4:5000

# Manually trigger one round on a single client (bypasses the autonomous random-interval loop):
curl -X POST localhost:5000/round

# Check a client's version without triggering a round:
curl localhost:5000/ping

docker compose down
```

There is no test suite, linter, or CI config in this repo — verify changes by running the stack and
reading the round logs / eval.py output as above.

## Architecture

- `client/model.py` — `MnistCNN`, the model every node trains. No config/env dependency; also imported
  standalone by `eval.py`.
- `client/data.py` — deterministically shards MNIST train data across `NUM_CLIENTS` by seeded shuffle +
  contiguous split, keyed off the numeric part of `CLIENT_ID`. Shards are disjoint and reproducible; the
  last shard absorbs the remainder so no samples are dropped.
- `client/node.py` — `BrainTorrentNode`, Algorithm 1 itself:
  - `self.version` is this node's own version (`v^i` in the paper), incremented once per completed round.
  - `self.peer_last_used_version` is this node's copy of the version vector for *other* clients — the
    last version of each peer it has already merged in. A peer only gets pulled when its current version
    (from `/ping`) is greater than the stored entry.
  - `run_round()`: ping every peer → pull `/weights` only from those strictly newer → weighted-average
    merge over **the participating subset only** (own weights + whichever peers actually responded this
    round) so the mix coefficients sum to exactly 1 — normalizing by the full `NUM_CLIENTS` instead would
    silently decay the model over time when peers are unreachable → fine-tune the merged model on the
    local shard → update `peer_last_used_version` for merged peers → increment own version.
  - All of the above runs under `self.lock` (a `threading.RLock`) because `run_round()` executes on a
    background thread while Flask serves `/weights` concurrently; `snapshot()` takes the lock too, so
    peers always see a consistent `(state_dict, version, n_samples)` triple, never a half-merged model.
  - A down/unreachable peer is skipped for that round (logged, not raised) — matches the paper's framing
    of a dynamic P2P environment where any client can be unavailable without stopping the others.
- `client/server.py` — Flask app (`/ping`, `/weights`, `POST /round`) plus the autonomous background
  thread that calls `run_round()` at random intervals (`ROUND_INTERVAL_MIN`/`MAX`). This randomness is
  intentional, not a placeholder — it's what makes the environment "highly dynamic" per the paper rather
  than a fixed round-robin schedule. There is deliberately no coordinator container; each client decides
  for itself when to initiate, same as the paper's server-less framing.
- `client/config.py` — all tunables read from env vars, set per-service in `docker-compose.yml`
  (`CLIENT_ID`, `PEERS` as a comma-separated `host:port` list using compose service names as hostnames,
  `NUM_CLIENTS`, `LOCAL_EPOCHS`, learning rate, round interval bounds, etc).
- `eval.py` — standalone script (only depends on `client/model.py`, not `client/config.py`, so it has no
  `CLIENT_ID` requirement and can run outside the P2P mesh). Pulls `/weights` from a set of hosts,
  reproduces the same sample-weighted merge as a node's own `run_round()`, and reports test accuracy —
  this is how to check convergence from outside without SSHing into a container's logs.
- `requirements.txt` — reference only, for a host-side venv (e.g. to run `eval.py` without Docker); the
  Dockerfile installs pinned versions directly and does not read this file, so keep the two in sync by hand
  if you bump a dependency.
- `Dockerfile` — CPU-only torch/torchvision wheels (`--index-url https://download.pytorch.org/whl/cpu`;
  the default CUDA wheels would bloat the image well past 2GB for no benefit here). MNIST is downloaded
  once at image-build time and baked into the layer specifically so 5 containers don't race each other
  downloading it at startup.
- `docker-compose.yml` — 5 client services (`client-0..4`), same image, differing only in `CLIENT_ID` and
  `PEERS`, plus a `dashboard` service (plain `nginx:alpine` serving `dashboard/` as static files on
  `:8080`). Adding a 6th client means adding a service block and adding its address to every other
  service's `PEERS` list (and bumping `NUM_CLIENTS`, since the MNIST shard split depends on it) — and to
  the dashboard's default host list (see below) if you want it to show up there too.
- `dashboard/index.html` — single-file vanilla-JS/CSS monitoring UI, no build step, no external
  dependencies (loads nothing but itself, so it works offline). Polls every client's `GET /status` and
  `GET /events?after=<seq>` directly **from the browser**, once a second — the `dashboard` container's
  only job is serving this one file; it never talks to the clients itself, which is why it doesn't need
  to be on the compose network. Client host list defaults to `localhost:5000..5004` and is overridable via
  `?hosts=host1:port,host2:port,...` in the URL. Polling is a `setTimeout`-chained loop, not
  `setInterval`, so a slow round-trip never overlaps the next poll — with `setInterval` two overlapping
  polls would read the same `lastSeq` and double-insert the same events into the feed.
  - `client/node.py`'s event log (`self.events`, a `seq`-numbered bounded deque) and `self.history`
    (per-round version/accuracy) exist specifically to feed this dashboard — `seq` lets the dashboard ask
    "what's new since I last asked" instead of re-fetching and de-duplicating everything each poll.
  - The "merge" event's `peers` list drives the network diagram: when client X merges peer Y's weights,
    the dashboard highlights the X–Y edge for a few seconds, so which clients just exchanged weights is
    visible without reading the event feed text.
  - `client/server.py` adds a wildcard `Access-Control-Allow-Origin` header on every response — required
    because the dashboard (port 8080) and each client (port 500X) are different origins from the
    browser's point of view. Fine for a local demo; would need tightening for anything internet-facing.

## Known scope boundaries (intentional, not gaps)

- Single-machine simulation via Docker containers, not a real multi-host deployment.
- MNIST + a small CNN, not the paper's QuickNAT whole-brain MRI segmentation — the P2P/versioning logic
  is what's being demonstrated, not the medical imaging task.
- No non-IID data distribution experiments (the paper's Experiment 2) — shards are uniform IID splits.
