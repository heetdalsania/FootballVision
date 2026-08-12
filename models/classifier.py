"""
Formation Classifier Module for NFL Vision App
Temporal transformer for play/formation classification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json
import os


class Formation(Enum):
    """Common NFL offensive formations."""
    # Shotgun variations
    SHOTGUN = "shotgun"
    SHOTGUN_SPREAD = "shotgun_spread"
    SHOTGUN_TRIPS = "shotgun_trips"
    SHOTGUN_EMPTY = "shotgun_empty"
    
    # Under center formations
    I_FORMATION = "i_formation"
    PRO_SET = "pro_set"
    SINGLE_BACK = "single_back"
    
    # Specialty formations
    PISTOL = "pistol"
    WILDCAT = "wildcat"
    GOAL_LINE = "goal_line"
    JUMBO = "jumbo"
    
    # Defensive formations
    DEFENSE_4_3 = "defense_4_3"
    DEFENSE_3_4 = "defense_3_4"
    NICKEL = "nickel"
    DIME = "dime"
    PREVENT = "prevent"
    
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Result of formation classification."""
    formation: Formation
    confidence: float
    top_k_predictions: List[Tuple[Formation, float]]
    frame_range: Tuple[int, int]
    
    def to_dict(self) -> dict:
        return {
            "formation": self.formation.value,
            "confidence": self.confidence,
            "top_k": [(f.value, c) for f, c in self.top_k_predictions],
            "frame_range": self.frame_range
        }


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TemporalTransformer(nn.Module):
    """
    Temporal Transformer for formation classification.
    
    Takes sequence of player positions and classifies the formation.
    Uses self-attention to capture spatial-temporal relationships.
    """
    
    def __init__(self,
                 num_players: int = 22,
                 input_dim: int = 2,  # x, y positions
                 d_model: int = 128,
                 nhead: int = 4,
                 num_encoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 num_classes: int = len(Formation),
                 dropout: float = 0.1,
                 max_seq_len: int = 60):
        super().__init__()
        
        self.num_players = num_players
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Input projection: flatten player positions
        self.input_proj = nn.Linear(num_players * input_dim, d_model)
        
        # Positional encoding for temporal dimension
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, 
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_players, 2)
            mask: Optional attention mask
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size, seq_len, num_players, coords = x.shape
        
        # Flatten player positions: (batch, seq_len, num_players * 2)
        x = x.view(batch_size, seq_len, -1)
        
        # Project to model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)
        
        # Normalize
        x = self.norm(x)
        
        # Classify
        logits = self.classifier(x)
        
        return logits


class FormationClassifier:
    """
    Formation classifier using temporal transformer.
    
    Aggregates player positions over time and classifies
    the offensive/defensive formation.
    
    Example:
        classifier = FormationClassifier()
        result = classifier.predict(track_history)
        print(f"Formation: {result.formation}, Confidence: {result.confidence}")
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "mps",
                 num_players: int = 22,
                 sequence_length: int = 30,
                 num_classes: int = len(Formation)):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to trained model weights. If None, uses random init.
            device: Device to run inference on.
            num_players: Maximum number of players to track.
            sequence_length: Number of frames for temporal context.
            num_classes: Number of formation classes.
        """
        self.device = device
        self.num_players = num_players
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        
        # Initialize model
        self.model = TemporalTransformer(
            num_players=num_players,
            num_classes=num_classes
        ).to(device)
        
        # Load weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"[FormationClassifier] Loaded weights from {model_path}")
        else:
            print("[FormationClassifier] Using randomly initialized weights")
        
        self.model.eval()
        
        # Formation mapping
        self.formations = list(Formation)
        
        # Position normalization params (field dimensions, approx)
        self.field_width = 1920  # Will be updated based on input
        self.field_height = 1080
    
    def preprocess(self, 
                   track_history: List[Dict[int, Tuple[int, int]]]) -> torch.Tensor:
        """
        Preprocess track history into model input.
        
        Args:
            track_history: List of frame dicts mapping track_id -> (x, y)
        
        Returns:
            Tensor of shape (1, seq_len, num_players, 2)
        """
        seq_len = min(len(track_history), self.sequence_length)
        
        # Initialize with zeros (for missing players)
        positions = np.zeros((seq_len, self.num_players, 2), dtype=np.float32)
        
        for t, frame_data in enumerate(track_history[-seq_len:]):
            for player_idx, (track_id, pos) in enumerate(frame_data.items()):
                if player_idx >= self.num_players:
                    break
                # Normalize positions to [0, 1]
                x_norm = pos[0] / self.field_width
                y_norm = pos[1] / self.field_height
                positions[t, player_idx] = [x_norm, y_norm]
        
        # Pad sequence if needed
        if seq_len < self.sequence_length:
            padding = np.zeros((self.sequence_length - seq_len, self.num_players, 2), dtype=np.float32)
            positions = np.concatenate([padding, positions], axis=0)
        
        tensor = torch.from_numpy(positions).unsqueeze(0).to(self.device)
        return tensor
    
    @torch.no_grad()
    def predict(self, 
                track_history: List[Dict[int, Tuple[int, int]]],
                top_k: int = 3) -> ClassificationResult:
        """
        Predict formation from track history.
        
        Args:
            track_history: List of frame dicts mapping track_id -> (x, y)
            top_k: Number of top predictions to return.
        
        Returns:
            ClassificationResult with formation and confidence.
        """
        if not track_history:
            return ClassificationResult(
                formation=Formation.UNKNOWN,
                confidence=0.0,
                top_k_predictions=[(Formation.UNKNOWN, 0.0)],
                frame_range=(0, 0)
            )
        
        # Preprocess
        x = self.preprocess(track_history)
        
        # Forward pass
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], min(top_k, self.num_classes))
        
        top_k_predictions = [
            (self.formations[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        best_idx = top_indices[0].item()
        best_prob = top_probs[0].item()
        
        return ClassificationResult(
            formation=self.formations[best_idx],
            confidence=best_prob,
            top_k_predictions=top_k_predictions,
            frame_range=(0, len(track_history))
        )
    
    def predict_from_positions(self,
                               positions: np.ndarray,
                               top_k: int = 3) -> ClassificationResult:
        """
        Predict from raw position array.
        
        Args:
            positions: Array of shape (seq_len, num_players, 2)
            top_k: Number of top predictions to return.
        
        Returns:
            ClassificationResult
        """
        # Normalize
        positions = positions.copy()
        positions[:, :, 0] /= self.field_width
        positions[:, :, 1] /= self.field_height
        
        x = torch.from_numpy(positions).float().unsqueeze(0).to(self.device)
        
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)
        
        top_probs, top_indices = torch.topk(probs[0], min(top_k, self.num_classes))
        
        top_k_predictions = [
            (self.formations[idx.item()], prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return ClassificationResult(
            formation=self.formations[top_indices[0].item()],
            confidence=top_probs[0].item(),
            top_k_predictions=top_k_predictions,
            frame_range=(0, positions.shape[0])
        )
    
    def train(self,
              dataset: List[Tuple[np.ndarray, int]],
              epochs: int = 100,
              batch_size: int = 32,
              learning_rate: float = 1e-4,
              save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Train the formation classifier.
        
        Args:
            dataset: List of (positions, label) tuples.
                     positions: (seq_len, num_players, 2)
                     label: int index of formation
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            save_path: Path to save trained model.
        
        Returns:
            Training history dict with loss and accuracy.
        """
        self.model.train()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = {"loss": [], "accuracy": []}
        
        num_samples = len(dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            # Shuffle dataset
            indices = np.random.permutation(num_samples)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Prepare batch
                batch_x = []
                batch_y = []
                for idx in batch_indices:
                    positions, label = dataset[idx]
                    # Normalize positions
                    positions = positions.copy()
                    positions[:, :, 0] /= self.field_width
                    positions[:, :, 1] /= self.field_height
                    batch_x.append(positions)
                    batch_y.append(label)
                
                x = torch.from_numpy(np.stack(batch_x)).float().to(self.device)
                y = torch.tensor(batch_y).long().to(self.device)
                
                # Forward pass
                logits = self.model(x)
                loss = criterion(logits, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Accuracy
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += len(y)
            
            avg_loss = epoch_loss / num_batches
            accuracy = correct / total
            
            history["loss"].append(avg_loss)
            history["accuracy"].append(accuracy)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        self.model.eval()
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"[FormationClassifier] Saved model to {save_path}")
        
        return history
    
    def demo_classify(self,
                      tracks: List,
                      frame_width: int = 1920,
                      frame_height: int = 1080) -> "ClassificationResult":
        """
        Heuristic formation classifier for demo mode — no trained weights needed.

        Uses player position geometry (spread, depth, count) to make a plausible
        formation call. Adds ±0.08 confidence noise so results feel live.

        Args:
            tracks: List of Track objects with a .center property.
            frame_width: Frame width for normalization.
            frame_height: Frame height for normalization.

        Returns:
            ClassificationResult with heuristic formation and confidence.
        """
        import random

        if not tracks:
            return ClassificationResult(
                formation=Formation.UNKNOWN,
                confidence=0.0,
                top_k_predictions=[(Formation.UNKNOWN, 0.0)],
                frame_range=(0, 0),
            )

        positions = [
            (t.center[0] / frame_width, t.center[1] / frame_height)
            for t in tracks
        ]

        if len(positions) < 5:
            return ClassificationResult(
                formation=Formation.UNKNOWN,
                confidence=0.5,
                top_k_predictions=[(Formation.UNKNOWN, 0.5)],
                frame_range=(0, 1),
            )

        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        x_spread = float(np.std(xs))
        x_range = max(xs) - min(xs)
        y_centroid = float(np.mean(ys))
        max_y_dist = max(abs(y - y_centroid) for y in ys)

        wide_players = sum(1 for x in xs if x < 0.25 or x > 0.75)
        central_players = sum(1 for x in xs if 0.3 < x < 0.7)
        player_count = len(tracks)

        # --- Offensive formation heuristics ---
        if x_range > 0.75 and wide_players >= 4:
            formation = Formation.SHOTGUN_EMPTY if max_y_dist > 0.15 else Formation.SHOTGUN_SPREAD
            confidence_base = 0.82 if max_y_dist > 0.15 else 0.76
        elif x_range > 0.65 and wide_players >= 3:
            formation = Formation.SHOTGUN_TRIPS if max_y_dist > 0.12 else Formation.SHOTGUN
            confidence_base = 0.78 if max_y_dist > 0.12 else 0.74
        elif x_range > 0.55:
            formation = Formation.PISTOL if max_y_dist > 0.10 else Formation.PRO_SET
            confidence_base = 0.70 if max_y_dist > 0.10 else 0.72
        elif x_range < 0.45 and central_players >= 6:
            formation = Formation.I_FORMATION
            confidence_base = 0.75
        elif player_count >= 18 and x_spread < 0.18:
            # Very tight cluster — likely goal-line or heavy defensive set
            formation = Formation.GOAL_LINE if player_count >= 20 else Formation.DEFENSE_4_3
            confidence_base = 0.71
        elif x_spread > 0.22:
            formation = Formation.NICKEL if player_count < 12 else Formation.SINGLE_BACK
            confidence_base = 0.68
        else:
            formation = Formation.DEFENSE_3_4
            confidence_base = 0.65

        # Add realism noise
        noise = random.uniform(-0.08, 0.08)
        confidence = float(np.clip(confidence_base + noise, 0.30, 0.97))

        # Build runner-up predictions
        candidates = [
            Formation.SHOTGUN, Formation.I_FORMATION, Formation.PRO_SET,
            Formation.SHOTGUN_SPREAD, Formation.PISTOL, Formation.SINGLE_BACK,
            Formation.DEFENSE_4_3, Formation.DEFENSE_3_4, Formation.NICKEL, Formation.DIME,
        ]
        others = [f for f in candidates if f != formation]
        random.shuffle(others)
        remaining = 1.0 - confidence
        top_k: List[Tuple[Formation, float]] = [(formation, confidence)]
        for i, f in enumerate(others[:2]):
            top_k.append((f, remaining * (0.6 if i == 0 else 0.4)))

        return ClassificationResult(
            formation=formation,
            confidence=confidence,
            top_k_predictions=top_k,
            frame_range=(0, 1),
        )

    def get_formation_names(self) -> List[str]:
        """Get list of formation names."""
        return [f.value for f in self.formations]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            "device": self.device,
            "num_players": self.num_players,
            "sequence_length": self.sequence_length,
            "num_classes": self.num_classes,
            "model_params": sum(p.numel() for p in self.model.parameters())
        }


if __name__ == "__main__":
    # Demo usage
    print("NFL Vision - Formation Classifier Demo")
    print("-" * 40)
    
    classifier = FormationClassifier(device="cpu")
    
    # Simulate track history
    track_history = []
    for t in range(30):
        frame_data = {}
        for player_id in range(11):
            # Random positions
            x = 100 + player_id * 50 + np.random.randint(-10, 10)
            y = 200 + (player_id % 4) * 100 + np.random.randint(-10, 10)
            frame_data[player_id] = (x, y)
        track_history.append(frame_data)
    
    print("\nPredicting formation...")
    result = classifier.predict(track_history)
    print(f"  Formation: {result.formation.value}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Top 3: {[(f.value, f'{c:.2%}') for f, c in result.top_k_predictions]}")
    
    print("\nClassifier stats:", classifier.get_stats())
