# wandb_config.py
from dataclasses import field
from typing import List
import wandb

WANDB_AVAILABLE = True

class WandBTracker:
    """Handles WandB initialization and metric logging."""
    
    def __init__(self, config, enabled=True):
        self.enabled = enabled and WANDB_AVAILABLE
        self.config = config
        
        # Initialize if enabled
        if self.enabled:
            wandb.init(
                project="tetris-dqn",
                config=self._make_config_dict()
            )
        
        # Metrics storage
        self.epoch_scores: List[float] = []
        self.epoch_pieces: List[int] = []
        self.epoch_lines: List[int] = []
        self.epoch_losses: List[float] = []
        self.epoch_rewards: List[float] = []
    
    def _make_config_dict(self) -> dict:
        """Convert config to WandB dict."""
        return {
            "num_epochs": self.config.num_epochs,
            "batch_size": self.config.batch_size,
            "lr": self.config.lr,
            "gamma": self.config.gamma,
            "initial_eps": self.config.initial_eps,
            "final_eps": self.config.final_eps,
            "decay_epochs": self.config.decay_epochs,
        }
    
    def log(self, data: dict):
        """Log metrics to WandB (does nothing if disabled)."""
        if self.enabled:
            wandb.log(data)
