# BrainTorrent

A working implementation of *[BrainTorrent: A Peer-to-Peer Environment for Decentralized Federated
Learning](https://arxiv.org/abs/1905.06731)* (Roy et al., 2019) — demonstrated on MNIST digit
classification instead of the paper's whole-brain MRI segmentation task, with a live web dashboard for
watching the protocol run.

## What is BrainTorrent

Federated learning normally works through a central server: every client trains locally, sends its
weights to the server, the server averages them, and sends the result back. BrainTorrent removes the
server entirely. Clients talk directly to each other:

1. A client pings its peers for their current model version.
2. It pulls weights only from peers whose version is newer than what it last merged.
3. It combines those weights with its own model — a sample-count-weighted average.
4. It fine-tunes the merged model on its own local data.
5. It bumps its own version and repeats, on its own schedule.

There's no coordinator, no single point of failure, and no fixed round-robin — every client decides for
itself when to start a round. This repo implements exactly that (Algorithm 1 in the paper), swapping the
paper's QuickNAT/MRI segmentation setup for a small CNN on MNIST so it runs in seconds on a laptop.

## Quick start

Requires Docker and Docker Compose.

```bash
docker compose up --build
```

This starts 5 client containers (`client-0` .. `client-4`), each with its own disjoint shard of MNIST,
each running the P2P protocol autonomously against the other 4.

Open the live dashboard:

```
http://localhost:8080
```

You'll see a network diagram (edges light up when two clients merge), per-client cards (version, round
count, accuracy, and each client's local copy of the version vector), and a live feed of every
ping/pull/merge/round event as it happens.

Try killing a client mid-run to see the protocol's fault tolerance:

```bash
docker compose stop client-3   # watch it go red on the dashboard; the rest keep merging
docker compose start client-3  # bring it back
```

Check convergence from the outside (pulls every client's current weights, merges them the same way a
node would, and reports test accuracy — the same "aggregated model" metric the paper reports):

```bash
docker run --rm --network braintorrent_default -v "$(pwd)/eval.py:/app/eval.py" \
    braintorrent-client-0 python eval.py \
    --hosts client-0:5000,client-1:5000,client-2:5000,client-3:5000,client-4:5000
```

Stop everything:

```bash
docker compose down
```

## Layout

| Path | What it is |
|---|---|
| `client/` | The node implementation: model, data sharding, Algorithm 1, Flask server |
| `dashboard/` | Single-file vanilla-JS monitoring UI, no build step |
| `eval.py` | Standalone script to check aggregated-model accuracy from outside the mesh |
| `docker-compose.yml` | 5 client services + the dashboard's static file server |
| `Dockerfile` | CPU-only PyTorch image with MNIST baked in at build time |

See [`CLAUDE.md`](CLAUDE.md) for a detailed architecture walkthrough (version vector semantics, merge
normalization, concurrency handling) and the scope boundaries of this demo versus the original paper.

## Reference

Roy, A. G., Siddiqui, S., Pölsterl, S., Navab, N., & Wachinger, C. (2019). *BrainTorrent: A Peer-to-Peer
Environment for Decentralized Federated Learning.* [arXiv:1905.06731](https://arxiv.org/abs/1905.06731)
