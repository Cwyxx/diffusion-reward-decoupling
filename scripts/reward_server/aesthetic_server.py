"""Aesthetic predictor reward server.

Usage:
    CUDA_VISIBLE_DEVICES=7 HF_ENDPOINT=https://hf-mirror.com python scripts/reward_server/aesthetic_server.py --port 18094
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import pickle
import traceback

import logging
import torch
from flask import Flask, request, Blueprint
from PIL import Image

from flow_grpo.aesthetic_scorer import AestheticScorer

root = Blueprint("root", __name__)
SCORER = None


def create_app(device="cuda", dtype=torch.float32):
    global SCORER
    SCORER = AestheticScorer(dtype=dtype).to(device)

    app = Flask(__name__)
    app.register_blueprint(root)
    # Suppress default Flask request logs, only show errors
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    return app


@root.route("/score", methods=["POST"])
def score():
    try:
        data = pickle.loads(request.get_data())

        # images: numpy uint8 array (N, H, W, C) -> list of PIL images
        images = [Image.fromarray(img) for img in data["images"]]

        scores = SCORER(images)
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().tolist()

        return pickle.dumps({"scores": scores}), 200

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return tb.encode("utf-8"), 500


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18094)
    args = parser.parse_args()

    app = create_app()
    print(f"Aesthetic server starting on {args.host}:{args.port}")
    app.run(args.host, args.port)
