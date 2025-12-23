"""
Export ONNX with embedded weights (no external data).
Uses ONNX external_data_location trick to avoid split.
"""
import torch
import torch.nn as nn
from pathlib import Path
from stable_baselines3 import PPO
import sys
import os
import shutil
import tempfile

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
    policy_net = policy_net.cpu()
    
    dummy_input = torch.randn(1, 5, dtype=torch.float32)
    
    # Export to temp directory first (to not create .data in root)
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_onnx = Path(tmpdir) / "model.onnx"
        
        print("Exporting to ONNX...")
        torch.onnx.export(
            policy_net,
            dummy_input,
            str(temp_onnx),
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
        
        # Check if .data file was created
        data_file = Path(tmpdir) / "model.onnx.data"
        
        if data_file.exists():
            print("Merging external data into single file...")
            import onnx
            from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
            
            # Load and convert to all-internal
            onnx_model = onnx.load(str(temp_onnx))
            
            # Remove external data info
            for tensor in onnx_model.graph.initializer:
                if tensor.HasField("data_location"):
                    # Load the raw data 
                    pass
            
            # Simply re-save with onnx
            onnx.save(onnx_model, str(output_path))
        else:
            # No external data, just copy
            shutil.copy(temp_onnx, output_path)
    
    size = output_path.stat().st_size
    print(f"Output: {output_path}")
    print(f"Size: {size} bytes ({size/1024:.1f} KB)")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX verification: PASSED")
    except Exception as e:
        print(f"Warning: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
