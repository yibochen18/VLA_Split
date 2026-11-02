"""Model 1 side of the socket: sends interface tensors to the Model 2 server and returns actions."""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

from protocol import actions_from_bytes, encode_request, encode_stats_request, read_response

SPLIT_DIR = Path(__file__).resolve().parent


def start_server(bundle, socket_path, log_path):
    """Launch model2_server.py as a separate process and wait until it is listening."""
    log = open(log_path, "w")
    proc = subprocess.Popen(
        [sys.executable, str(SPLIT_DIR / "model2_server.py"), "--bundle", str(bundle), "--socket", socket_path],
        stdout=subprocess.PIPE, stderr=log, text=True,
    )
    line = proc.stdout.readline()
    if not line.startswith("READY"):
        proc.kill()
        raise RuntimeError(f"Model 2 server failed to start; see {log_path}")
    return proc


class SplitClient:
    def __init__(self, socket_path):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(socket_path)
        self.next_id = 0

    def query(self, hidden):
        """Returns (actions (8, 7) float64, timings in seconds)."""
        t0 = time.perf_counter()
        request = encode_request(self.next_id, hidden)
        t1 = time.perf_counter()
        self.sock.sendall(request)
        request_id, status, payload, server_seconds, _ = read_response(self.sock)
        t2 = time.perf_counter()
        assert request_id == self.next_id and status == 0, (request_id, self.next_id, status)
        self.next_id += 1
        timings = {
            "serialize": t1 - t0,
            "round_trip": t2 - t1,
            "server": server_seconds,
            "transfer": (t2 - t1) - server_seconds,
            "request_bytes": len(request),
        }
        return actions_from_bytes(payload), timings

    def stats(self):
        self.sock.sendall(encode_stats_request(self.next_id))
        _, _, payload, _, _ = read_response(self.sock)
        return json.loads(payload)

    def close(self):
        self.sock.close()
