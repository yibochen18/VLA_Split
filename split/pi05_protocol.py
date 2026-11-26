"""Wire format between the LIBERO harness and the π0.5 server (responses reuse protocol.py's framing)."""

import json
import struct

import numpy as np

from protocol import MAGIC_STATS, encode_response, read_response, recv_exact

REQUEST_HEADER = struct.Struct("<4sQI")  # magic, request id, length of the JSON metadata
MAGIC_QUERY = b"PI5Q"
MAGIC_END = b"PI5E"  # end of episode: the server writes its recording for that episode
IMAGE_SHAPE = (256, 256, 3)  # rendered LIBERO frame, already rotated 180°; the server resizes as openpi does
STATE_DIM = 8
ACTION_SHAPE = (10, 7)


def encode_request(request_id, meta, image=None, wrist_image=None, state=None, magic=MAGIC_QUERY):
    meta_bytes = json.dumps(meta).encode()
    payload = b""
    if magic == MAGIC_QUERY:
        assert image.shape == IMAGE_SHAPE and wrist_image.shape == IMAGE_SHAPE and image.dtype == np.uint8
        payload = image.tobytes() + wrist_image.tobytes() + np.asarray(state, dtype=np.float64).tobytes()
    return REQUEST_HEADER.pack(magic, request_id, len(meta_bytes)) + meta_bytes + payload


def read_request(sock):
    """Returns (magic, request_id, meta, image, wrist_image, state, bytes_received); arrays are None unless a query."""
    magic, request_id, meta_len = REQUEST_HEADER.unpack(recv_exact(sock, REQUEST_HEADER.size))
    meta = json.loads(bytes(recv_exact(sock, meta_len)))
    n = REQUEST_HEADER.size + meta_len
    if magic != MAGIC_QUERY:
        return magic, request_id, meta, None, None, None, n
    image_bytes = int(np.prod(IMAGE_SHAPE))
    payload = recv_exact(sock, 2 * image_bytes + 8 * STATE_DIM)
    image = np.frombuffer(payload, dtype=np.uint8, count=image_bytes).reshape(IMAGE_SHAPE)
    wrist_image = np.frombuffer(payload, dtype=np.uint8, count=image_bytes, offset=image_bytes).reshape(IMAGE_SHAPE)
    state = np.frombuffer(payload, dtype=np.float64, offset=2 * image_bytes).copy()
    return magic, request_id, meta, image, wrist_image, state, n + len(payload)


def actions_to_bytes(actions):
    assert actions.shape == ACTION_SHAPE and actions.dtype == np.float64
    return actions.tobytes()


def actions_from_bytes(payload):
    return np.frombuffer(payload, dtype=np.float64).reshape(ACTION_SHAPE).copy()
