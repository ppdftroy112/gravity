"""
Re-export ONNX model without external data file.
"""
import torch
import torch.nn as nn
from pathlib import Path
from stable_baselines3 import PPO
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golf_env import GolfEnv


class PolicyNetwork(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        
    def forward(self, obs):
        features = self.mlp_extractor.forward_actor(obs)
        return self.action_net(features)


def main():
    model_path = Path("models/golf_agent.zip")
    output_path = Path("../golf_agent.onnx")
    
    print("Loading trained model...")
    env = GolfEnv()
    model = PPO.load(model_path, env=env)
    
    print("Creating policy wrapper...")
    policy_net = PolicyNetwork(model.policy)
    policy_net.eval()
    
    # Move to CPU and ensure no external data
    policy_net = policy_net.cpu()
    
    dummy_input = torch.randn(1, 5, dtype=torch.float32)
    
    print("Exporting to ONNX (embedded weights)...")
    
    # Delete old file if exists
    if output_path.exists():
        output_path.unlink()
    
    # Export
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
    
    # Check file size - should be ~9KB without external data
    size = output_path.stat().st_size
    print(f"Output file: {output_path}")
    print(f"File size: {size} bytes ({size/1024:.1f} KB)")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX verification: PASSED")
        print(f"Inputs: {[i.name for i in onnx_model.graph.input]}")
        print(f"Outputs: {[o.name for o in onnx_model.graph.output]}")
    except Exception as e:
        print(f"Verification: {e}")
    
    print("\nDone! Model exported to root folder for GitHub Pages.")


if __name__ == "__main__":
    main()
