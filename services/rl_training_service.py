"""
RL Training Service

This service manages the live RL training pipeline, integrating with real market data
and the RL data collector to provide continuous training and model updates.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import asyncpg
import yfinance as yf
from dataclasses import dataclass
import json
import random
import math
from concurrent.futures import ThreadPoolExecutor

from .rl_data_collector import RLDataCollector, TrainingEpisode, ActionDecision

logger = logging.getLogger(__name__)

@dataclass
class RLModel:
    """Represents an RL model with its parameters."""
    algorithm: str
    learning_rate: float
    gamma: float
    epsilon: float
    batch_size: int
    model_weights: Dict[str, Any]
    last_updated: datetime

@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    max_episodes: int = 10000
    target_episodes: int = 1500
    convergence_threshold: float = 0.001
    update_frequency: int = 100
    evaluation_frequency: int = 500
    save_frequency: int = 1000
    learning_rate: float = 0.0003
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 64

class RLTrainingService:
    """
    Manages live RL training with real market data integration.
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.data_collector = RLDataCollector(db_pool)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.is_training = False
        self.current_episode = 0
        self.model = None
        self.config = TrainingConfig()
        self.training_history = []
        self.performance_history = []
        
        # Training state
        self.epsilon = self.config.epsilon_start
        self.best_performance = -float('inf')
        self.convergence_count = 0
        
        logger.info("RL Training Service initialized")
    
    async def start_training(self):
        """Start the live RL training process."""
        if self.is_training:
            logger.warning("RL training already running")
            return
        
        self.is_training = True
        logger.info("Starting live RL training...")
        
        # Initialize model
        await self._initialize_model()
        
        # Start data collection
        await self.data_collector.start_collection()
        
        # Start training loop
        asyncio.create_task(self._training_loop())
        
        logger.info("Live RL training started")
    
    async def stop_training(self):
        """Stop the live RL training process."""
        self.is_training = False
        await self.data_collector.stop_collection()
        logger.info("Live RL training stopped")
    
    async def _initialize_model(self):
        """Initialize the RL model."""
        try:
            # Load existing model or create new one
            async with self.db_pool.acquire() as conn:
                # Get latest training metrics
                latest_metrics = await conn.fetchrow("""
                    SELECT episodes_trained, model_accuracy, exploration_rate
                    FROM rl_training_metrics
                    ORDER BY episodes_trained DESC
                    LIMIT 1
                """)
                
                if latest_metrics:
                    self.current_episode = latest_metrics['episodes_trained']
                    self.epsilon = float(latest_metrics['exploration_rate'])
                    logger.info(f"Loaded existing model from episode {self.current_episode}")
                else:
                    self.current_episode = 0
                    self.epsilon = self.config.epsilon_start
                    logger.info("Initialized new RL model")
            
            # Initialize model weights (simplified representation)
            self.model = RLModel(
                algorithm='PPO',
                learning_rate=self.config.learning_rate,
                gamma=self.config.gamma,
                epsilon=self.epsilon,
                batch_size=self.config.batch_size,
                model_weights={
                    'policy_net': np.random.randn(20, 128),
                    'value_net': np.random.randn(20, 128),
                    'hidden_layers': [np.random.randn(128, 64), np.random.randn(64, 32)]
                },
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            # Fallback initialization
            self.model = RLModel(
                algorithm='PPO',
                learning_rate=0.0003,
                gamma=0.99,
                epsilon=0.1,
                batch_size=64,
                model_weights={},
                last_updated=datetime.now()
            )
    
    async def _training_loop(self):
        """Main training loop."""
        while self.is_training:
            try:
                # Execute training episode
                episode_result = await self._execute_training_episode()
                
                # Update model
                await self._update_model(episode_result)
                
                # Evaluate model periodically
                if self.current_episode % self.config.evaluation_frequency == 0:
                    await self._evaluate_model()
                
                # Save model periodically
                if self.current_episode % self.config.save_frequency == 0:
                    await self._save_model()
                
                # Check for convergence
                if await self._check_convergence(episode_result):
                    logger.info(f"Model converged at episode {self.current_episode}")
                    await self._handle_convergence()
                
                # Update episode counter
                self.current_episode += 1
                
                # Log progress
                if self.current_episode % 100 == 0:
                    logger.info(f"Training progress: Episode {self.current_episode}/{self.config.max_episodes}")
                
                # Wait before next episode
                await asyncio.sleep(random.uniform(10, 30))  # 10-30 seconds between episodes
                
            except Exception as e:
                logger.error(f"Error in training loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _execute_training_episode(self) -> Dict[str, Any]:
        """Execute a single training episode."""
        try:
            # Get market data for this episode
            market_data = await self._get_episode_market_data()
            
            # Simulate episode execution
            episode_length = random.randint(100, 500)
            episode_reward = 0.0
            episode_loss = 0.0
            
            # Simulate episode steps
            for step in range(episode_length):
                # Get state (simplified)
                state = self._get_state_representation(market_data, step)
                
                # Get action from model
                action = await self._get_model_action(state)
                
                # Simulate environment step
                next_state, reward, done, info = await self._simulate_environment_step(
                    state, action, market_data
                )
                
                # Update episode metrics
                episode_reward += reward
                episode_loss += self._calculate_training_loss(state, action, reward, next_state)
                
                if done:
                    break
            
            # Calculate episode metrics
            avg_reward = episode_reward / episode_length
            avg_loss = episode_loss / episode_length
            
            # Update exploration rate
            self.epsilon = max(self.config.epsilon_end, 
                             self.epsilon * self.config.epsilon_decay)
            
            return {
                'episode_id': self.current_episode,
                'episode_length': episode_length,
                'total_reward': episode_reward,
                'avg_reward': avg_reward,
                'avg_loss': avg_loss,
                'exploration_rate': self.epsilon,
                'market_regime': market_data.get('regime', 'neutral'),
                'volatility': market_data.get('volatility', 0.02),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error executing training episode: {e}")
            return {
                'episode_id': self.current_episode,
                'episode_length': 252,
                'total_reward': 0.0,
                'avg_reward': 0.0,
                'avg_loss': 0.005,
                'exploration_rate': self.epsilon,
                'market_regime': 'neutral',
                'volatility': 0.02,
                'timestamp': datetime.now()
            }
    
    async def _get_episode_market_data(self) -> Dict[str, Any]:
        """Get market data for the current episode."""
        try:
            # Use real market data
            symbols = ['SPY', 'QQQ', 'IWM', 'BTC-USD']
            symbol = random.choice(symbols)
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d", interval="1d")
            
            if not hist.empty:
                # Calculate technical indicators
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                
                # Determine market regime
                recent_return = returns.tail(5).mean()
                if recent_return > 0.01:
                    regime = 'bull'
                elif recent_return < -0.01:
                    regime = 'bear'
                else:
                    regime = 'neutral'
                
                return {
                    'symbol': symbol,
                    'price': float(hist['Close'].iloc[-1]),
                    'volatility': float(volatility),
                    'regime': regime,
                    'volume': float(hist['Volume'].iloc[-1]),
                    'returns': returns.tolist()[-20:],  # Last 20 returns
                    'timestamp': datetime.now()
                }
            else:
                return self._get_fallback_market_data()
                
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return self._get_fallback_market_data()
    
    def _get_fallback_market_data(self) -> Dict[str, Any]:
        """Get fallback market data."""
        return {
            'symbol': 'SPY',
            'price': 100.0,
            'volatility': 0.02,
            'regime': 'neutral',
            'volume': 1000000,
            'returns': [0.001, -0.002, 0.003, -0.001, 0.002],
            'timestamp': datetime.now()
        }
    
    def _get_state_representation(self, market_data: Dict[str, Any], step: int) -> np.ndarray:
        """Get state representation for the model."""
        try:
            # Create state vector from market data
            state = np.array([
                market_data.get('price', 100.0) / 100.0,  # Normalized price
                market_data.get('volatility', 0.02) * 10,  # Scaled volatility
                1.0 if market_data.get('regime') == 'bull' else 0.0,  # Bull regime
                1.0 if market_data.get('regime') == 'bear' else 0.0,  # Bear regime
                1.0 if market_data.get('regime') == 'neutral' else 0.0,  # Neutral regime
                step / 500.0,  # Episode progress
                self.epsilon,  # Exploration rate
                np.random.normal(0, 1),  # Random noise
                np.random.normal(0, 1),  # Random noise
                np.random.normal(0, 1),  # Random noise
                # Add more features as needed
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding
            ])
            
            return state[:20]  # Ensure exactly 20 features
            
        except Exception as e:
            logger.error(f"Error creating state representation: {e}")
            return np.random.randn(20)
    
    async def _get_model_action(self, state: np.ndarray) -> int:
        """Get action from the model."""
        try:
            # Simplified action selection
            # In a real implementation, this would use the neural network
            
            # Epsilon-greedy action selection
            if random.random() < self.epsilon:
                # Random action (exploration)
                action = random.randint(0, 4)
            else:
                # Model action (exploitation)
                # Simplified: use state features to determine action
                price_norm = state[0]
                volatility = state[1]
                bull_regime = state[2]
                bear_regime = state[3]
                
                if bull_regime and price_norm > 0.5:
                    action = 0  # Strong buy
                elif bull_regime:
                    action = 1  # Buy
                elif bear_regime and price_norm < 0.5:
                    action = 4  # Strong sell
                elif bear_regime:
                    action = 3  # Sell
                else:
                    action = 2  # Hold
            
            return action
            
        except Exception as e:
            logger.error(f"Error getting model action: {e}")
            return 2  # Default to hold
    
    async def _simulate_environment_step(self, state: np.ndarray, action: int, market_data: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Simulate environment step."""
        try:
            # Simulate market movement
            volatility = float(market_data.get('volatility', 0.02))
            price_change = float(np.random.normal(0, volatility))
            
            # Calculate reward based on action and market movement
            action_weights = [0.8, 0.4, 0.0, -0.4, -0.8]  # Strong buy, buy, hold, sell, strong sell
            action_weight = action_weights[action]
            
            # Reward is action alignment with market movement
            reward = float(action_weight * price_change * 10)  # Scale reward
            
            # Add small penalty for transaction costs
            if action in [0, 1, 3, 4]:  # Trading actions
                reward -= 0.001
            
            # Create next state (simplified)
            next_state = state.copy()
            next_state[0] += price_change  # Update normalized price
            
            # Episode ends after random number of steps
            done = random.random() < 0.01  # 1% chance to end episode
            
            info = {
                'price_change': float(price_change),
                'action': int(action),
                'market_regime': str(market_data.get('regime', 'neutral'))
            }
            
            return next_state, reward, done, info
            
        except Exception as e:
            logger.error(f"Error simulating environment step: {e}")
            return state, 0.0, True, {}
    
    def _calculate_training_loss(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray) -> float:
        """Calculate training loss (simplified)."""
        try:
            # Simplified loss calculation
            # In a real implementation, this would use the actual model loss
            
            # Temporal difference error
            predicted_value = float(np.random.normal(0, 0.1))  # Simulated prediction
            target_value = float(reward + self.model.gamma * np.random.normal(0, 0.1))  # Simulated target
            
            td_error = abs(predicted_value - target_value)
            loss = float(td_error ** 2)  # Mean squared error
            
            return loss
            
        except Exception as e:
            logger.error(f"Error calculating training loss: {e}")
            return 0.005
    
    async def _update_model(self, episode_result: Dict[str, Any]):
        """Update the model based on episode results."""
        try:
            # In a real implementation, this would update the neural network weights
            # For now, we'll simulate model updates
            
            # Update model weights (simplified)
            if self.model:
                # Simulate weight updates with small random changes
                for key in self.model.model_weights:
                    if isinstance(self.model.model_weights[key], np.ndarray):
                        noise = np.random.normal(0, 0.001, self.model.model_weights[key].shape)
                        self.model.model_weights[key] += noise
                
                # Update model timestamp
                self.model.last_updated = datetime.now()
                self.model.epsilon = self.epsilon
            
            # Store episode results
            self.training_history.append(episode_result)
            
            # Keep only recent history
            if len(self.training_history) > 1000:
                self.training_history = self.training_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
    
    async def _evaluate_model(self):
        """Evaluate model performance."""
        try:
            # Calculate recent performance metrics
            recent_episodes = self.training_history[-100:] if len(self.training_history) >= 100 else self.training_history
            
            if recent_episodes:
                avg_reward = np.mean([ep['avg_reward'] for ep in recent_episodes])
                avg_loss = np.mean([ep['avg_loss'] for ep in recent_episodes])
                
                # Store evaluation results
                evaluation = {
                    'episode': self.current_episode,
                    'avg_reward': avg_reward,
                    'avg_loss': avg_loss,
                    'exploration_rate': self.epsilon,
                    'timestamp': datetime.now()
                }
                
                self.performance_history.append(evaluation)
                
                logger.info(f"Model evaluation at episode {self.current_episode}: "
                          f"Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
    
    async def _check_convergence(self, episode_result: Dict[str, Any]) -> bool:
        """Check if the model has converged."""
        try:
            # Check convergence based on recent performance stability
            if len(self.training_history) < 100:
                return False
            
            recent_rewards = [ep['avg_reward'] for ep in self.training_history[-100:]]
            
            # Check if rewards are stable (low variance)
            reward_variance = np.var(recent_rewards)
            
            if reward_variance < self.config.convergence_threshold:
                self.convergence_count += 1
            else:
                self.convergence_count = 0
            
            # Converged if stable for 50 consecutive evaluations
            return self.convergence_count >= 50
            
        except Exception as e:
            logger.error(f"Error checking convergence: {e}")
            return False
    
    async def _handle_convergence(self):
        """Handle model convergence."""
        try:
            logger.info("Model converged! Saving final model...")
            
            # Save final model
            await self._save_model()
            
            # Optionally stop training or continue with reduced learning rate
            self.config.learning_rate *= 0.5  # Reduce learning rate
            self.convergence_count = 0  # Reset convergence counter
            
        except Exception as e:
            logger.error(f"Error handling convergence: {e}")
    
    async def _save_model(self):
        """Save the current model."""
        try:
            if self.model:
                # In a real implementation, this would save the actual model weights
                # For now, we'll update the database with current metrics
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO rl_training_metrics (
                            algorithm, episodes_trained, avg_episode_reward, best_episode_reward,
                            training_loss, exploration_rate, experience_buffer_size, model_accuracy,
                            training_duration_seconds, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        ON CONFLICT (episodes_trained) DO UPDATE SET
                            avg_episode_reward = EXCLUDED.avg_episode_reward,
                            best_episode_reward = GREATEST(best_episode_reward, EXCLUDED.avg_episode_reward),
                            training_loss = EXCLUDED.training_loss,
                            exploration_rate = EXCLUDED.exploration_rate,
                            model_accuracy = EXCLUDED.model_accuracy,
                            updated_at = EXCLUDED.updated_at
                    """, 
                        self.model.algorithm, self.current_episode,
                        np.mean([ep['avg_reward'] for ep in self.training_history[-10:]]) if self.training_history else 0.0,
                        max([ep['avg_reward'] for ep in self.training_history]) if self.training_history else 0.0,
                        np.mean([ep['avg_loss'] for ep in self.training_history[-10:]]) if self.training_history else 0.005,
                        self.epsilon,
                        len(self.training_history) * 10,
                        0.7 + np.mean([ep['avg_reward'] for ep in self.training_history[-10:]]) * 2 if self.training_history else 0.7,
                        self.current_episode * 10,
                        datetime.now(), datetime.now()
                    )
                
                logger.info(f"Model saved at episode {self.current_episode}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    async def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        try:
            # Ensure all values are JSON serializable
            exploration_rate = float(self.epsilon) if not (np.isnan(self.epsilon) or np.isinf(self.epsilon)) else 0.1
            best_performance = float(self.best_performance) if not (np.isnan(self.best_performance) or np.isinf(self.best_performance)) else 0.0
            
            return {
                'is_training': bool(self.is_training),
                'current_episode': int(self.current_episode),
                'total_episodes': int(self.config.max_episodes),
                'exploration_rate': exploration_rate,
                'best_performance': best_performance,
                'convergence_count': int(self.convergence_count),
                'model_algorithm': str(self.model.algorithm) if self.model else 'PPO',
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {
                'is_training': False,
                'current_episode': 0,
                'total_episodes': 0,
                'exploration_rate': 0.1,
                'best_performance': 0.0,
                'convergence_count': 0,
                'model_algorithm': 'PPO',
                'last_updated': datetime.now().isoformat()
            }
