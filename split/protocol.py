"""Wire format between Model 1 and Model 2."""

import struct

import numpy as np
import torch

REQUEST_HEADER = struct.Struct("<4sQB3I7x")
RESPONSE_HEADER = struct.Struct("<QIId")
MAGIC_QUERY = b"VLA1"
MAGIC_STATS = b"STAT"
DTYPE_BF16 = 0
STATUS_OK = 0
ACTION_SHAPE = (8, 7)


def recv_exact(sock, n):
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("socket closed")
        got += k
    return buf


def encode_request(request_id, hidden):
    assert hidden.dtype == torch.bfloat16 and hidden.dim() == 3
    payload = hidden.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()
    return REQUEST_HEADER.pack(MAGIC_QUERY, request_id, DTYPE_BF16, *hidden.shape) + payload


def encode_stats_request(request_id):
    return REQUEST_HEADER.pack(MAGIC_STATS, request_id, DTYPE_BF16, 0, 0, 0)


def read_request(sock):
    """Returns (magic, request_id, tensor or None, bytes_received)."""
    magic, request_id, dtype, *shape = REQUEST_HEADER.unpack(recv_exact(sock, REQUEST_HEADER.size))
    if magic == MAGIC_STATS:
        return magic, request_id, None, REQUEST_HEADER.size
    assert magic == MAGIC_QUERY and dtype == DTYPE_BF16, (magic, dtype)
    nbytes = int(np.prod(shape)) * 2
    payload = recv_exact(sock, nbytes)
    tensor = torch.frombuffer(payload, dtype=torch.int16).view(torch.bfloat16).reshape(shape)
    return magic, request_id, tensor, REQUEST_HEADER.size + nbytes


def encode_response(request_id, payload_bytes, server_seconds, status=STATUS_OK):
    return RESPONSE_HEADER.pack(request_id, status, len(payload_bytes), server_seconds) + payload_bytes


def read_response(sock):
    """Returns (request_id, status, payload bytes, server_seconds, bytes_received)."""
    request_id, status, length, server_seconds = RESPONSE_HEADER.unpack(recv_exact(sock, RESPONSE_HEADER.size))
    payload = bytes(recv_exact(sock, length))
    return request_id, status, payload, server_seconds, RESPONSE_HEADER.size + length


def actions_to_bytes(actions):
    assert actions.shape == ACTION_SHAPE and actions.dtype == np.float64
    return actions.tobytes()


def actions_from_bytes(payload):
    return np.frombuffer(payload, dtype=np.float64).reshape(ACTION_SHAPE).copy()
