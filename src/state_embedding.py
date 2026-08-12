"""
Game-State Embedding + Similarity Retrieval for FootballVision

Arsenal's fourth model output is embedding-based retrieval: "find similar
situations". Given the current arrangement of players on the pitch, return
the moments from earlier in the match that looked most like it.

We represent a game state as a **per-team occupancy heatmap** — each team's
players are Gaussian-splatted onto a coarse pitch grid, the two grids are
concatenated and L2-normalised. This embedding is:

    - permutation-invariant : player identity / ordering doesn't matter
    - count-robust           : works with 9, 10, or 11 detected players
    - shape-aware            : encodes block position, width, compactness,
                               and which team occupies which zones

Similarity is cosine distance between embeddings. `SituationStore` keeps a
rolling buffer of past states and answers k-NN queries; it is thread-safe so
the capture thread (pipeline) and the query thread (API) can share it.

DuckDB/FAISS are natural production back-ends; for the prototype a normalised
numpy matrix with brute-force cosine is simpler, dependency-free, and plenty
fast for a match's worth of snapshots.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TEAM_A = 0
TEAM_B = 1


class StateEmbedder:
    """Turns a list of {team, x, y} players into a fixed-length unit vector."""

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        grid_w: int = 16,
        grid_h: int = 10,
        sigma_cells: float = 1.1,
    ):
        self.L = pitch_length
        self.W = pitch_width
        self.gw = grid_w
        self.gh = grid_h
        self.sigma = sigma_cells
        self.dim = 2 * grid_w * grid_h

        # Precompute grid-cell centre coordinates (in cell units) for splatting.
        gx = np.arange(grid_w) + 0.5
        gy = np.arange(grid_h) + 0.5
        self._cx, self._cy = np.meshgrid(gx, gy)  # (gh, gw)

    def embed(self, players: Sequence[Dict]) -> np.ndarray:
        """Return a unit-norm embedding for one game state."""
        grid_a = self._team_grid(players, TEAM_A)
        grid_b = self._team_grid(players, TEAM_B)
        vec = np.concatenate([grid_a.ravel(), grid_b.ravel()]).astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        return vec

    def _team_grid(self, players: Sequence[Dict], team: int) -> np.ndarray:
        grid = np.zeros((self.gh, self.gw), dtype=np.float32)
        two_sig2 = 2.0 * self.sigma * self.sigma
        for p in players:
            if p.get("team") != team:
                continue
            # Player position in cell units.
            px = np.clip(p["x"] / self.L, 0, 1) * self.gw
            py = np.clip(p["y"] / self.W, 0, 1) * self.gh
            d2 = (self._cx - px) ** 2 + (self._cy - py) ** 2
            grid += np.exp(-d2 / two_sig2)
        return grid


class SituationStore:
    """
    Rolling buffer of game-state embeddings + metadata with cosine k-NN.

    Thread-safe: the pipeline capture thread calls `add`, the API thread calls
    `query`; both take the same lock.
    """

    def __init__(self, dim: int, capacity: int = 4000):
        self.dim = dim
        self.capacity = capacity
        self._vecs: List[np.ndarray] = []
        self._metas: List[Dict] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._vecs)

    def add(self, vec: np.ndarray, meta: Dict) -> None:
        if vec.shape[0] != self.dim:
            raise ValueError(f"embedding dim {vec.shape[0]} != store dim {self.dim}")
        with self._lock:
            self._vecs.append(vec.astype(np.float32))
            self._metas.append(meta)
            if len(self._vecs) > self.capacity:
                # Drop the oldest snapshot.
                self._vecs.pop(0)
                self._metas.pop(0)

    def query(
        self,
        vec: np.ndarray,
        k: int = 5,
        exclude_within_s: float = 0.0,
        query_time_s: Optional[float] = None,
    ) -> List[Dict]:
        """
        Return up to k most-similar past states as
        [{"similarity", "meta"}], most similar first.

        Args:
            vec: query embedding (unit norm).
            k: number of neighbours.
            exclude_within_s: skip snapshots whose meta["t"] is within this many
                seconds of query_time_s (so "now" doesn't retrieve itself).
            query_time_s: reference time for the exclusion window.
        """
        with self._lock:
            if not self._vecs:
                return []
            mat = np.stack(self._vecs)          # (N, D), already unit norm
            metas = list(self._metas)

        # Both operands are unit norm, so the row-wise dot product is cosine
        # similarity. einsum (not matmul/@) avoids a spurious RuntimeWarning
        # some numpy builds emit on the matmul path.
        sims = np.einsum(
            "ij,j->i", mat.astype(np.float64), vec.astype(np.float64)
        )

        order = np.argsort(-sims)
        results: List[Dict] = []
        for idx in order:
            meta = metas[idx]
            if (
                exclude_within_s > 0
                and query_time_s is not None
                and abs(meta.get("t", 0.0) - query_time_s) <= exclude_within_s
            ):
                continue
            results.append({"similarity": round(float(sims[idx]), 4), "meta": meta})
            if len(results) >= k:
                break
        return results

    def clear(self) -> None:
        with self._lock:
            self._vecs.clear()
            self._metas.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        with self._lock:
            vecs = np.stack(self._vecs) if self._vecs else np.zeros((0, self.dim), np.float32)
            metas = list(self._metas)
        path = Path(path)
        np.save(path.with_suffix(".npy"), vecs)
        path.with_suffix(".json").write_text(
            json.dumps({"dim": self.dim, "capacity": self.capacity, "metas": metas})
        )

    @classmethod
    def load(cls, path: str | Path) -> "SituationStore":
        path = Path(path)
        info = json.loads(path.with_suffix(".json").read_text())
        store = cls(dim=info["dim"], capacity=info.get("capacity", 4000))
        vecs = np.load(path.with_suffix(".npy"))
        store._vecs = [v.astype(np.float32) for v in vecs]
        store._metas = list(info["metas"])
        return store


if __name__ == "__main__":
    # Build three distinct states; confirm a query matches its own family.
    rng = np.random.default_rng(0)

    def block(ax, bx, jitter=0.0):
        players = []
        for i in range(11):
            players.append({"team": 0, "x": ax + rng.normal(0, jitter),
                            "y": 10 + i * 5})
            players.append({"team": 1, "x": bx + rng.normal(0, jitter),
                            "y": 10 + i * 5})
        return players

    emb = StateEmbedder()
    store = SituationStore(dim=emb.dim)

    # Family 1: low block (teams deep). Family 2: high block (teams advanced).
    for t in range(10):
        store.add(emb.embed(block(25, 55, 1.5)), {"frame_id": t, "t": float(t), "family": "low"})
    for t in range(10, 20):
        store.add(emb.embed(block(70, 95, 1.5)), {"frame_id": t, "t": float(t), "family": "high"})

    q = emb.embed(block(26, 56, 0.5))  # looks like the "low" family
    hits = store.query(q, k=3)
    print(f"store size: {len(store)} | dim: {emb.dim}")
    for h in hits:
        print(f"  sim={h['similarity']:.3f}  family={h['meta']['family']}  frame={h['meta']['frame_id']}")
    assert all(h["meta"]["family"] == "low" for h in hits), "should retrieve the low-block family"
    print("\nEMBEDDING RETRIEVAL OK")
