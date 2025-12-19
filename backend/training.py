"""
Training Manager for Golf RL Agent
Uses Stable-Baselines3 PPO algorithm
"""

import os
import threading
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from golf_env import GolfEnv


class TrainingCallback(BaseCallback):
    """Custom callback to track training progress."""
    
    def __init__(self, training_manager, verbose=0):
        super().__init__(verbose)
        self.training_manager = training_manager
        self.episode_rewards = []
        self.episode_count = 0
    
    def _on_step(self) -> bool:
        # Check if training should stop
        if self.training_manager.should_stop:
            return False
        
        # Update progress
        self.training_manager.current_step = self.num_timesteps
        self.training_manager.progress = (
            self.num_timesteps / self.training_manager.total_timesteps * 100
        )
        
        # Track episode rewards
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_count += 1
                    self.training_manager.episodes = self.episode_count
                    
                    # Update mean reward (last 100 episodes)
                    if len(self.episode_rewards) > 0:
                        recent = self.episode_rewards[-100:]
                        self.training_manager.mean_reward = np.mean(recent)
        
        return True


class TrainingManager:
    """Manages the training process for the Golf RL agent."""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.model: Optional[PPO] = None
        self.env: Optional[GolfEnv] = None
        
        # Training state
        self.is_training = False
        self.should_stop = False
        self.training_thread: Optional[threading.Thread] = None
        
        # Progress tracking
        self.current_step = 0
        self.total_timesteps = 0
        self.progress = 0.0
        self.episodes = 0
        self.mean_reward = 0.0
        
        # Load existing model if available
        self._load_model()
    
    def _load_model(self):
        """Load existing model or create new one."""
        model_path = self.model_dir / "golf_agent.zip"
        
        self.env = GolfEnv()
        
        if model_path.exists():
            print(f"Loading existing model from {model_path}")
            self.model = PPO.load(model_path, env=self.env)
        else:
            print("Creating new PPO model")
            self.model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1
            )
    
    def start_training(self, learning_rate: float = 3e-4, total_timesteps: int = 10000):
        """Start training in a background thread."""
        if self.is_training:
            return {"status": "error", "message": "Already training"}
        
        self.is_training = True
        self.should_stop = False
        self.total_timesteps = total_timesteps
        self.current_step = 0
        self.progress = 0.0
        self.episodes = 0
        self.mean_reward = 0.0
        
        # Update learning rate
        self.model.learning_rate = learning_rate
        
        # Start training thread
        self.training_thread = threading.Thread(target=self._train)
        self.training_thread.start()
        
        return {"status": "started", "total_timesteps": total_timesteps}
    
    def _train(self):
        """Training loop (runs in background thread)."""
        try:
            callback = TrainingCallback(self)
            
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callback,
                reset_num_timesteps=False,
                progress_bar=False
            )
            
            # Save model after training
            model_path = self.model_dir / "golf_agent.zip"
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
            
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            self.is_training = False
            self.progress = 100.0
    
    def stop_training(self):
        """Stop the training process."""
        self.should_stop = True
        
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5)
        
        self.is_training = False
        
        # Save model
        model_path = self.model_dir / "golf_agent.zip"
        self.model.save(model_path)
        
        return {"status": "stopped"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "is_training": self.is_training,
            "progress": self.progress,
            "current_step": self.current_step,
            "total_timesteps": self.total_timesteps,
            "episodes": self.episodes,
            "mean_reward": self.mean_reward
        }
    
    def predict(self, state: Dict[str, float]) -> Dict[str, Any]:
        """Get action prediction for a given state."""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Convert state dict to observation array
        obs = np.array([
            state.get("ball_x", 0.5),
            state.get("ball_y", 0.9),
            state.get("hole_x", 0.5),
            state.get("hole_y", 0.1),
            state.get("distance", 0.8)
        ], dtype=np.float32)
        
        # Get action from model
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Convert action to angle and power
        angle = float(action[0]) * 90  # -90 to 90 degrees
        power = float(np.clip(action[1], 0, 1))  # 0 to 1
        
        # Adjust angle for frontend (shooting upward)
        final_angle = -90 + angle
        
        return {
            "action": {
                "angle": final_angle,
                "power": power
            },
            "raw_action": action.tolist()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        model_path = self.model_dir / "golf_agent.zip"
        
        return {
            "model_exists": model_path.exists(),
            "model_path": str(model_path),
            "policy": "MlpPolicy (PPO)",
            "observation_space": str(self.env.observation_space) if self.env else None,
            "action_space": str(self.env.action_space) if self.env else None
        }


# Global training manager instance
training_manager = TrainingManager()
