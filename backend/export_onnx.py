"""
Export trained PPO model to ONNX format for browser inference.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO


class PolicyNetwork(nn.Module):
    """Wrapper to extract just the policy network for ONNX export."""
    
    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        
    def forward(self, obs):
        # Extract features
        features = self.mlp_extractor.forward_actor(obs)
        # Get action mean (for deterministic policy)
        action_mean = self.action_net(features)
        return action_mean


def export_to_onnx(model_path: str = "models/golf_agent.zip", output_path: str = "models/golf_agent.onnx"):
    """Export the trained model to ONNX format."""
    
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return False
    
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create policy wrapper
    policy_net = PolicyNetwork(model.policy)
    policy_net.eval()
    
    # Create dummy input (observation space: 5 dimensions)
    dummy_input = torch.randn(1, 5, dtype=torch.float32)
    
    print(f"Exporting to ONNX format...")
    
    # Export to ONNX
    torch.onnx.export(
        policy_net,
        dummy_input,
        str(output_path),
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
    
    print(f"Model exported successfully to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: PASSED")
    except ImportError:
        print("Note: Install 'onnx' package to verify the model")
    except Exception as e:
        print(f"ONNX model verification warning: {e}")
    
    return True


if __name__ == "__main__":
    export_to_onnx()
