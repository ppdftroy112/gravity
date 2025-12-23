"""
Golf Environment for Reinforcement Learning
Compatible with Gymnasium and Stable-Baselines3
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GolfEnv(gym.Env):
    """
    A simple 2D golf environment.
    
    State Space:
        - ball_x: Ball X position (normalized 0-1)
        - ball_y: Ball Y position (normalized 0-1)
        - hole_x: Hole X position (normalized 0-1)
        - hole_y: Hole Y position (normalized 0-1)
        - distance: Normalized distance to hole
    
    Action Space:
        - angle: Shot angle (-1 to 1, mapped to -90 to 90 degrees)
        - power: Shot power (0 to 1)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        
        # Canvas dimensions (matching frontend)
        self.width = 500
        self.height = 600
        
        # Physics parameters
        self.friction = 0.985
        self.min_velocity = 0.1
        self.max_power = 25
        self.ball_radius = 8
        self.hole_radius = 14

        # Gaussian Gravity Well Parameters
        self.well_sigma = 40.0
        self.well_depth = 20.0
        self.gravity_scale = 1.0
        
        # Hole capture
        self.max_hole_entry_velocity = 20.0
        
        # Max steps per episode
        self.max_steps = 20
        self.current_step = 0
        
        # Observation space: [ball_x, ball_y, hole_x, hole_y, distance]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # Action space: [angle (-1 to 1), power (0 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        
        # Initialize state
        self.ball_pos = np.array([0.0, 0.0])
        self.hole_pos = np.array([0.0, 0.0])
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset ball to starting position
        self.ball_pos = np.array([
            self.width / 2,
            self.height - 80
        ], dtype=np.float32)
        
        # Randomize hole position
        self.hole_pos = np.array([
            100 + self.np_random.random() * (self.width - 200),
            60 + self.np_random.random() * 100
        ], dtype=np.float32)
        
        self.current_step = 0
        self.shots = 0
        
        return self._get_obs(), {}
    
    def _get_obs(self):
        """Get normalized observation."""
        distance = np.linalg.norm(self.ball_pos - self.hole_pos)
        max_distance = np.sqrt(self.width**2 + self.height**2)
        
        return np.array([
            self.ball_pos[0] / self.width,
            self.ball_pos[1] / self.height,
            self.hole_pos[0] / self.width,
            self.hole_pos[1] / self.height,
            distance / max_distance
        ], dtype=np.float32)
    
    def step(self, action):
        self.current_step += 1
        self.shots += 1
        
        # Parse action
        angle = action[0] * 90  # -90 to 90 degrees
        power = np.clip(action[1], 0, 1) * self.max_power
        
        # Convert to radians (adjust for coordinate system)
        # Negative angle means shooting upward (toward hole)
        angle_rad = np.radians(-90 + angle)  # Shooting generally upward
        
        # Initial velocity
        vx = np.cos(angle_rad) * power
        vy = np.sin(angle_rad) * power
        
        # Simulate ball movement
        velocity = np.array([vx, vy], dtype=np.float32)
        
        sim_steps = 0
        while np.linalg.norm(velocity) > self.min_velocity and sim_steps < 100:
            sim_steps += 1
            
            # Apply Gaussian Gravity Well
            d_vec = self.hole_pos - self.ball_pos
            dist = np.linalg.norm(d_vec)
            
            # Derivative of Gaussian: d/dr (-Depth * exp(-r^2 / 2sigma^2))
            # = (r * Depth / sigma^2) * exp(-r^2 / 2sigma^2)
            if dist < self.well_sigma * 3:  # Optimization: only calculate if close enough
                exp_factor = np.exp(- (dist**2) / (2 * self.well_sigma**2))
                force_mag = (dist * self.well_depth / (self.well_sigma**2)) * exp_factor * self.gravity_scale
                
                # Force direction is towards hole (normalized d_vec)
                force_vec = (d_vec / (dist + 1e-6)) * force_mag
                velocity += force_vec

            # Update position
            self.ball_pos += velocity
            
            # Apply friction
            velocity *= self.friction
            
            # Boundary collision
            if self.ball_pos[0] < self.ball_radius:
                self.ball_pos[0] = self.ball_radius
                velocity[0] *= -0.8
            if self.ball_pos[0] > self.width - self.ball_radius:
                self.ball_pos[0] = self.width - self.ball_radius
                velocity[0] *= -0.8
            if self.ball_pos[1] < self.ball_radius:
                self.ball_pos[1] = self.ball_radius
                velocity[1] *= -0.8
            if self.ball_pos[1] > self.height - self.ball_radius:
                self.ball_pos[1] = self.height - self.ball_radius
                velocity[1] *= -0.8
        
        # Calculate distance to hole
        distance = np.linalg.norm(self.ball_pos - self.hole_pos)
        max_distance = np.sqrt(self.width**2 + self.height**2)
        
        # Check if ball is in hole (within radius AND slow enough)
        # Note: Velocity might be high from gravity well accel, so we need some tolerance
        current_speed = np.linalg.norm(velocity)
        in_hole = distance < self.hole_radius and current_speed < self.max_hole_entry_velocity
        
        # Calculate reward
        if in_hole:
            # Bonus for fewer shots
            if self.shots == 1:
                reward = 100.0  # Hole in one!
            else:
                reward = 50.0 - (self.shots - 1) * 5.0
        else:
            # Reward for getting closer to hole
            normalized_distance = distance / max_distance
            reward = -normalized_distance * 10.0  # Penalty based on distance
            
            # Small penalty per shot
            reward -= 1.0
        
        # Episode termination
        terminated = in_hole
        truncated = self.current_step >= self.max_steps
        
        info = {
            "shots": self.shots,
            "distance": distance,
            "in_hole": in_hole
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            # Could implement pygame rendering here
            pass
        return None
    
    def close(self):
        pass


# Test the environment
if __name__ == "__main__":
    env = GolfEnv()
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    
    # Take a random action
    action = env.action_space.sample()
    print(f"Action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Observation: {obs}")
    print(f"Reward: {reward}")
    print(f"Info: {info}")
