"""Unix-socket server that runs a Model 2 bundle."""

import argparse
import json
import os
import socket
import sys
import time

import torch

from model2 import Model2
from protocol import MAGIC_STATS, actions_to_bytes, encode_response, read_request


class Server:
    def __init__(self, model2):
        self.model2 = model2
        self.bytes_in = 0
        self.bytes_out = 0
        self.times = {"decode": [], "compute": [], "encode": []}

    def stats(self):
        head = self.model2.head
        return {
            "num_queries": self.model2.num_queries,
            "bytes_in": self.bytes_in,
            "bytes_out": self.bytes_out,
            "mean_ms": {k: 1e3 * sum(v) / len(v) for k, v in self.times.items() if v},
            "model2_params": sum(p.numel() for p in head.parameters()),
            "model2_param_bytes": sum(p.numel() * p.element_size() for p in head.parameters()),
            "cuda_max_memory_allocated": torch.cuda.max_memory_allocated(),
            "pid": os.getpid(),
            "loaded_modules_check": sorted(m for m in sys.modules if m.split(".")[0] in ("prismatic", "transformers", "experiments")),
        }

    def handle(self, conn):
        while True:
            try:
                magic, request_id, hidden, n_in = read_request(conn)
            except ConnectionError:
                return
            t0 = time.perf_counter()
            self.bytes_in += n_in
            if magic == MAGIC_STATS:
                payload = json.dumps(self.stats()).encode()
            else:
                t1 = time.perf_counter()
                hidden = hidden.to(self.model2.device)
                actions = self.model2(hidden)
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                payload = actions_to_bytes(actions)
                t3 = time.perf_counter()
                self.times["decode"].append(t1 - t0)
                self.times["compute"].append(t2 - t1)
                self.times["encode"].append(t3 - t2)
            reply = encode_response(request_id, payload, time.perf_counter() - t0)
            conn.sendall(reply)
            self.bytes_out += len(reply)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--socket", default="/tmp/vla_split.sock")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    server = Server(Model2.load(args.bundle, args.device))
    if os.path.exists(args.socket):
        os.unlink(args.socket)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(args.socket)
    listener.listen(1)
    print(f"READY {args.socket}", flush=True)
    try:
        while True:
            conn, _ = listener.accept()
            with conn:
                server.handle(conn)
    finally:
        os.unlink(args.socket)


if __name__ == "__main__":
    main()
