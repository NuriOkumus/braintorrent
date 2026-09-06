import random
import threading
import time

from flask import Flask, Response, jsonify, request

from client import config
from client.node import BrainTorrentNode, log

app = Flask(__name__)
node = BrainTorrentNode()


@app.after_request
def add_cors_headers(response):
    # The dashboard (a static page on its own port) fetches these endpoints directly from
    # the browser, so every client origin needs to be readable cross-origin. This is a local
    # demo with no auth/sensitive data, so a wildcard is fine.
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.get("/ping")
def ping():
    """ping_request target: lets any peer check this client's current version."""
    return jsonify({"id": node.id, "version": node.version})


@app.get("/status")
def status():
    """Everything the dashboard needs about this node in one call."""
    return jsonify(node.status())


@app.get("/events")
def events():
    """Event log for the dashboard's live feed. ?after=<seq> returns only newer events."""
    after = int(request.args.get("after", -1))
    with node.lock:
        return jsonify([e for e in node.events if e["seq"] > after])


@app.get("/weights")
def weights():
    """Returns a consistent snapshot of this client's model weights."""
    state_bytes, version, n_samples = node.snapshot()
    return Response(
        state_bytes,
        mimetype="application/octet-stream",
        headers={"X-Version": str(version), "X-Num-Samples": str(n_samples), "X-Client-Id": node.id},
    )


@app.post("/round")
def trigger_round():
    """Manual trigger, mainly useful for tests/debugging outside the autonomous loop."""
    node.run_round()
    return jsonify({"id": node.id, "version": node.version})


def _autonomous_loop() -> None:
    # Stagger the very first round so all containers aren't hammering /ping at t=0.
    time.sleep(random.uniform(1, config.ROUND_INTERVAL_MIN))
    while True:
        try:
            node.run_round()
        except Exception:
            log.exception("round failed, will retry next interval")
        time.sleep(random.uniform(config.ROUND_INTERVAL_MIN, config.ROUND_INTERVAL_MAX))


def main() -> None:
    thread = threading.Thread(target=_autonomous_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=config.PORT, threaded=True)


if __name__ == "__main__":
    main()
