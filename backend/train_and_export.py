"""
Train Golf AI with 100,000 timesteps and export to single ONNX file.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from golf_env import GolfEnv


class PolicyNetwork(nn.Module):
    """Wrapper to extract just the policy network for ONNX export."""
    
    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        
    def forward(self, obs):
        features = self.mlp_extractor.forward_actor(obs)
        action_mean = self.action_net(features)
        return action_mean


def main():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "golf_agent.zip"
    onnx_path = model_dir / "golf_agent.onnx"
    root_onnx_path = Path("..") / "golf_agent.onnx"
    
    # Create environment
    env = GolfEnv()
    
    # Load or create model
    if model_path.exists():
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
    
    # Train for 20,000 timesteps
    print("\n" + "="*50)
    print("Starting training: 20,000 timesteps")
    print("="*50 + "\n")
    
    model.learn(total_timesteps=20000, progress_bar=False)
    
    # Save model
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Export to ONNX with all weights embedded
    print("\nExporting to ONNX format...")
    
    policy_net = PolicyNetwork(model.policy)
    policy_net.eval()
    
    dummy_input = torch.randn(1, 5, dtype=torch.float32)
    
    # Export WITHOUT external data (all weights embedded in single file)
    torch.onnx.export(
        policy_net,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )
    
    print(f"ONNX model saved to {onnx_path}")
    print(f"File size: {onnx_path.stat().st_size / 1024:.1f} KB")
    
    # Copy to root folder for GitHub Pages
    import shutil
    shutil.copy(onnx_path, root_onnx_path)
    print(f"Copied to {root_onnx_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification: PASSED")
    except Exception as e:
        print(f"ONNX verification warning: {e}")
    
    print("\n" + "="*50)
    print("Training and export complete!")
    print("="*50)


if __name__ == "__main__":
    main()
