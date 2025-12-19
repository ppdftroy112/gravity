"""
API Routes for Golf RL Training
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from training import training_manager


router = APIRouter(prefix="/api", tags=["training"])


class TrainingConfig(BaseModel):
    """Training configuration."""
    learning_rate: float = 0.0003
    total_timesteps: int = 10000


class GameState(BaseModel):
    """Game state for prediction."""
    ball_x: float
    ball_y: float
    hole_x: float
    hole_y: float
    distance: float


@router.post("/train/start")
async def start_training(config: TrainingConfig):
    """Start the RL training process."""
    result = training_manager.start_training(
        learning_rate=config.learning_rate,
        total_timesteps=config.total_timesteps
    )
    
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    
    return result


@router.post("/train/stop")
async def stop_training():
    """Stop the current training process."""
    return training_manager.stop_training()


@router.get("/train/status")
async def get_training_status():
    """Get current training status."""
    return training_manager.get_status()


@router.post("/predict")
async def predict_action(state: GameState):
    """Get AI prediction for a given game state."""
    return training_manager.predict(state.model_dump())


@router.get("/model/info")
async def get_model_info():
    """Get information about the current model."""
    return training_manager.get_model_info()
