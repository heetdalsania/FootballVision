#!/usr/bin/env python3
"""Optionally cache the public SigLIP team model; no login or token needed."""

from huggingface_hub import snapshot_download


MODEL = "google/siglip-base-patch16-224"


if __name__ == "__main__":
    print(f"Caching optional team model {MODEL} (the local colour model is the default)…")
    path = snapshot_download(MODEL)
    print(f"Ready: {path}")
    print("Use it with FOOTBALLVISION_TEAM_BACKEND=siglip python main.py")
