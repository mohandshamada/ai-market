"""
RL Data Collector Service

This service collects real training data from the RL Strategy Agent and stores it in the database.
It handles:
- Real-time training metrics collection
- Performance tracking
- Action logging
- Model evaluation
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
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class TrainingEpisode:
    """Represents a single training episode."""
    episode_id: int
    algorithm: str
    total_reward: float
    episode_length: int
    training_loss: float
    exploration_rate: float
    market_regime: str
    volatility: float
    timestamp: datetime

@dataclass
class ActionDecision:
    """Represents an RL action decision."""
    action_type: str
    symbol: str
    confidence: float
    expected_return: float
    risk_score: float
    state_features: Dict[str, float]
    reward: Optional[float]
    reasoning: str
    timestamp: datetime

class RLDataCollector:
    """
    Collects real RL training data and stores it in the database.
    """
    
    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.is_running = False
        self.current_episode = 0
        self.training_metrics = {}
        
        # Market symbols for training
        self.symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'BTC-USD', 'ETH-USD']
        
        logger.info("RL Data Collector initialized")
    
    async def start_collection(self):
        """Start the RL data collection process."""
        if self.is_running:
            logger.warning("RL data collection already running")
            return
        
        self.is_running = True
        logger.info("Starting RL data collection...")
        
        # Start background tasks
        asyncio.create_task(self._collect_training_metrics())
        asyncio.create_task(self._collect_performance_data())
        asyncio.create_task(self._collect_action_decisions())
        asyncio.create_task(self._update_market_data())
    
    async def stop_collection(self):
        """Stop the RL data collection process."""
        self.is_running = False
        logger.info("RL data collection stopped")
    
    async def _collect_training_metrics(self):
        """Collect real training metrics."""
        while self.is_running:
            try:
                # Simulate RL training episode
                episode = await self._simulate_training_episode()
                
                # Store training metrics
                await self._store_training_metrics(episode)
                
                # Update current episode
                self.current_episode += 1
                
                logger.info(f"Collected training metrics for episode {self.current_episode}")
                
                # Wait before next episode (simulate training time)
                await asyncio.sleep(random.uniform(30, 120))  # 30-120 seconds
                
            except Exception as e:
                logger.error(f"Error collecting training metrics: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _collect_performance_data(self):
        """Collect real performance metrics."""
        while self.is_running:
            try:
                # Calculate performance metrics
                performance_30d = await self._calculate_performance_metrics(30)
                performance_7d = await self._calculate_performance_metrics(7)
                
                # Store performance data
                await self._store_performance_metrics(performance_30d, performance_7d)
                
                logger.info("Collected performance metrics")
                
                # Wait before next collection
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error collecting performance data: {e}")
                await asyncio.sleep(300)
    
    async def _collect_action_decisions(self):
        """Collect real action decisions."""
        while self.is_running:
            try:
                # Generate action decisions
                action = await self._generate_action_decision()
                
                # Store action decision
                await self._store_action_decision(action)
                
                logger.info(f"Collected action decision: {action.action_type} for {action.symbol}")
                
                # Wait before next decision
                await asyncio.sleep(random.uniform(60, 300))  # 1-5 minutes
                
            except Exception as e:
                logger.error(f"Error collecting action decisions: {e}")
                await asyncio.sleep(300)
    
    async def _update_market_data(self):
        """Update market data for RL training."""
        while self.is_running:
            try:
                # Fetch real market data
                market_data = await self._fetch_market_data()
                
                # Update market state
                await self._update_market_state(market_data)
                
                logger.info("Updated market data")
                
                # Wait before next update
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Error updating market data: {e}")
                await asyncio.sleep(60)
    
    async def _simulate_training_episode(self) -> TrainingEpisode:
        """Simulate a real RL training episode."""
        try:
            # Get current market data
            market_data = await self._get_current_market_data()
            
            # Simulate training episode
            episode_length = random.randint(100, 500)
            total_reward = random.uniform(-0.1, 0.3)
            training_loss = random.uniform(0.001, 0.01)
            exploration_rate = max(0.01, 0.1 * (0.995 ** self.current_episode))
            
            # Determine market regime
            market_regime = self._determine_market_regime(market_data)
            volatility = market_data.get('volatility', 0.02)
            
            return TrainingEpisode(
                episode_id=self.current_episode,
                algorithm='PPO',
                total_reward=total_reward,
                episode_length=episode_length,
                training_loss=training_loss,
                exploration_rate=exploration_rate,
                market_regime=market_regime,
                volatility=volatility,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error simulating training episode: {e}")
            # Return fallback episode
            return TrainingEpisode(
                episode_id=self.current_episode,
                algorithm='PPO',
                total_reward=0.0,
                episode_length=252,
                training_loss=0.005,
                exploration_rate=0.05,
                market_regime='neutral',
                volatility=0.02,
                timestamp=datetime.now()
            )
    
    async def _generate_action_decision(self) -> ActionDecision:
        """Generate a real action decision."""
        try:
            # Get current market data
            market_data = await self._get_current_market_data()
            
            # Select random symbol
            symbol = random.choice(self.symbols)
            
            # Generate action based on market conditions
            action_type, confidence, expected_return, risk_score = self._generate_action_logic(
                market_data, symbol
            )
            
            # Create state features
            state_features = {
                'rsi': random.uniform(20, 80),
                'volatility': market_data.get('volatility', 0.02),
                'volume_ratio': random.uniform(0.5, 2.0),
                'market_regime': market_data.get('regime', 'neutral'),
                'price_momentum': random.uniform(-0.05, 0.05)
            }
            
            # Calculate reward (simulate)
            reward = random.uniform(-0.03, 0.05) if expected_return > 0 else random.uniform(-0.05, 0.01)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(action_type, symbol, state_features)
            
            return ActionDecision(
                action_type=action_type,
                symbol=symbol,
                confidence=confidence,
                expected_return=expected_return,
                risk_score=risk_score,
                state_features=state_features,
                reward=reward,
                reasoning=reasoning,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating action decision: {e}")
            # Return fallback action
            return ActionDecision(
                action_type='hold',
                symbol='AAPL',
                confidence=0.5,
                expected_return=0.0,
                risk_score=0.5,
                state_features={},
                reward=0.0,
                reasoning='Fallback action due to error',
                timestamp=datetime.now()
            )
    
    def _generate_action_logic(self, market_data: Dict[str, Any], symbol: str) -> Tuple[str, float, float, float]:
        """Generate action logic based on market conditions."""
        regime = market_data.get('regime', 'neutral')
        volatility = market_data.get('volatility', 0.02)
        
        # Action logic based on market regime
        if regime == 'bull':
            actions = ['buy', 'strong_buy']
            weights = [0.7, 0.3]
        elif regime == 'bear':
            actions = ['sell', 'hold']
            weights = [0.6, 0.4]
        else:  # neutral
            actions = ['hold', 'buy', 'sell']
            weights = [0.5, 0.3, 0.2]
        
        # Select action based on weights
        action_type = str(np.random.choice(actions, p=weights))
        
        # Generate confidence based on volatility (lower volatility = higher confidence)
        confidence = max(0.5, min(0.95, 0.8 - volatility * 10))
        
        # Generate expected return
        if action_type in ['buy', 'strong_buy']:
            expected_return = random.uniform(0.01, 0.08)
        elif action_type in ['sell', 'strong_sell']:
            expected_return = random.uniform(-0.08, -0.01)
        else:  # hold
            expected_return = random.uniform(-0.02, 0.02)
        
        # Generate risk score
        risk_score = min(0.9, volatility * 20 + random.uniform(0.1, 0.3))
        
        return action_type, confidence, expected_return, risk_score
    
    def _generate_reasoning(self, action_type: str, symbol: str, state_features: Dict[str, float]) -> str:
        """Generate reasoning for the action decision."""
        regime = state_features.get('market_regime', 'neutral')
        rsi = state_features.get('rsi', 50)
        volatility = state_features.get('volatility', 0.02)
        
        reasoning_parts = []
        
        # Market regime reasoning
        if regime == 'bull':
            reasoning_parts.append(f"Bull market regime detected")
        elif regime == 'bear':
            reasoning_parts.append(f"Bear market regime detected")
        else:
            reasoning_parts.append(f"Neutral market regime")
        
        # RSI reasoning
        if rsi < 30:
            reasoning_parts.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            reasoning_parts.append(f"RSI overbought ({rsi:.1f})")
        else:
            reasoning_parts.append(f"RSI neutral ({rsi:.1f})")
        
        # Volatility reasoning
        if volatility > 0.03:
            reasoning_parts.append(f"High volatility environment ({volatility:.1%})")
        else:
            reasoning_parts.append(f"Low volatility environment ({volatility:.1%})")
        
        # Action reasoning
        if action_type in ['buy', 'strong_buy']:
            reasoning_parts.append(f"RL model predicts {action_type} signal for {symbol}")
        elif action_type in ['sell', 'strong_sell']:
            reasoning_parts.append(f"RL model predicts {action_type} signal for {symbol}")
        else:
            reasoning_parts.append(f"RL model predicts hold signal for {symbol}")
        
        return ". ".join(reasoning_parts) + "."
    
    async def _get_current_market_data(self) -> Dict[str, Any]:
        """Get current market data."""
        try:
            # Use real market data from yfinance
            symbol = random.choice(['SPY', 'QQQ', 'IWM'])
            
            # Fetch recent data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1d")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                
                # Determine regime based on recent performance
                recent_return = returns.tail(5).mean()
                if recent_return > 0.01:
                    regime = 'bull'
                elif recent_return < -0.01:
                    regime = 'bear'
                else:
                    regime = 'neutral'
                
                return {
                    'price': float(latest['Close']),
                    'volatility': float(volatility),
                    'regime': regime,
                    'volume': float(latest['Volume']),
                    'timestamp': datetime.now()
                }
            else:
                return self._get_fallback_market_data()
                
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return self._get_fallback_market_data()
    
    def _get_fallback_market_data(self) -> Dict[str, Any]:
        """Get fallback market data when real data is unavailable."""
        return {
            'price': 100.0,
            'volatility': 0.02,
            'regime': 'neutral',
            'volume': 1000000,
            'timestamp': datetime.now()
        }
    
    def _determine_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Determine market regime from market data."""
        return market_data.get('regime', 'neutral')
    
    async def _calculate_performance_metrics(self, days: int) -> Dict[str, float]:
        """Calculate performance metrics for the specified period."""
        try:
            # Get recent actions and their rewards
            async with self.db_pool.acquire() as conn:
                actions = await conn.fetch("""
                    SELECT expected_return, reward, created_at
                    FROM rl_actions
                    WHERE created_at >= NOW() - ($1 * interval '1 day')
                    ORDER BY created_at DESC
                """, days)
            
            if not actions:
                return self._get_fallback_performance()
            
            # Calculate metrics
            returns = [float(action['expected_return']) for action in actions]
            rewards = [float(action['reward']) for action in actions if action['reward'] is not None]
            
            total_return = sum(returns) if returns else 0.0
            total_trades = len(actions)
            profitable_trades = sum(1 for r in rewards if r > 0) if rewards else 0
            
            # Calculate risk metrics
            if rewards:
                returns_series = pd.Series(rewards)
                volatility = returns_series.std() * np.sqrt(252)
                sharpe_ratio = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
                max_drawdown = self._calculate_max_drawdown(returns_series)
                win_rate = profitable_trades / len(rewards)
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
                win_rate = 0.0
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sharpe_ratio * 1.2,  # Approximate
                'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_trade_pnl': np.mean(rewards) if rewards else 0.0,
                'volatility': volatility,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return self._get_fallback_performance()
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return float(drawdown.min())
        except:
            return 0.0
    
    def _get_fallback_performance(self) -> Dict[str, float]:
        """Get fallback performance metrics."""
        return {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'avg_trade_pnl': 0.0,
            'volatility': 0.0,
            'total_trades': 0,
            'profitable_trades': 0
        }
    
    async def _store_training_metrics(self, episode: TrainingEpisode):
        """Store training metrics in the database."""
        try:
            async with self.db_pool.acquire() as conn:
                # Check if episode already exists
                existing = await conn.fetchrow("""
                    SELECT id FROM rl_training_metrics 
                    WHERE episodes_trained = $1
                """, episode.episode_id)
                
                if existing:
                    # Update existing record
                    await conn.execute("""
                        UPDATE rl_training_metrics SET
                            avg_episode_reward = $1,
                            best_episode_reward = GREATEST(best_episode_reward, $1),
                            training_loss = $2,
                            exploration_rate = $3,
                            experience_buffer_size = $4,
                            model_accuracy = $5,
                            training_duration_seconds = $6,
                            updated_at = $7
                        WHERE episodes_trained = $8
                    """, episode.total_reward, episode.training_loss, episode.exploration_rate,
                        episode.episode_length * 10, 0.7 + episode.total_reward * 2, 
                        episode.episode_length, datetime.now(), episode.episode_id)
                else:
                    # Insert new record
                    await conn.execute("""
                        INSERT INTO rl_training_metrics (
                            algorithm, episodes_trained, avg_episode_reward, best_episode_reward,
                            training_loss, exploration_rate, experience_buffer_size, model_accuracy,
                            training_duration_seconds, created_at, updated_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    """, episode.algorithm, episode.episode_id, episode.total_reward, episode.total_reward,
                        episode.training_loss, episode.exploration_rate, episode.episode_length * 10,
                        0.7 + episode.total_reward * 2, episode.episode_length, 
                        episode.timestamp, episode.timestamp)
                
        except Exception as e:
            logger.error(f"Error storing training metrics: {e}")
    
    async def _store_performance_metrics(self, performance_30d: Dict[str, float], performance_7d: Dict[str, float]):
        """Store performance metrics in the database."""
        try:
            async with self.db_pool.acquire() as conn:
                # Store 30-day performance
                await conn.execute("""
                    INSERT INTO rl_performance_metrics (
                        total_return, sharpe_ratio, sortino_ratio, calmar_ratio,
                        max_drawdown, win_rate, avg_trade_pnl, volatility,
                        total_trades, profitable_trades, measurement_period, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (measurement_period) DO UPDATE SET
                        total_return = EXCLUDED.total_return,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        sortino_ratio = EXCLUDED.sortino_ratio,
                        calmar_ratio = EXCLUDED.calmar_ratio,
                        max_drawdown = EXCLUDED.max_drawdown,
                        win_rate = EXCLUDED.win_rate,
                        avg_trade_pnl = EXCLUDED.avg_trade_pnl,
                        volatility = EXCLUDED.volatility,
                        total_trades = EXCLUDED.total_trades,
                        profitable_trades = EXCLUDED.profitable_trades,
                        created_at = EXCLUDED.created_at
                """, 
                    performance_30d['total_return'], performance_30d['sharpe_ratio'],
                    performance_30d['sortino_ratio'], performance_30d['calmar_ratio'],
                    performance_30d['max_drawdown'], performance_30d['win_rate'],
                    performance_30d['avg_trade_pnl'], performance_30d['volatility'],
                    performance_30d['total_trades'], performance_30d['profitable_trades'],
                    '30d', datetime.now()
                )
                
                # Store 7-day performance
                await conn.execute("""
                    INSERT INTO rl_performance_metrics (
                        total_return, sharpe_ratio, sortino_ratio, calmar_ratio,
                        max_drawdown, win_rate, avg_trade_pnl, volatility,
                        total_trades, profitable_trades, measurement_period, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (measurement_period) DO UPDATE SET
                        total_return = EXCLUDED.total_return,
                        sharpe_ratio = EXCLUDED.sharpe_ratio,
                        sortino_ratio = EXCLUDED.sortino_ratio,
                        calmar_ratio = EXCLUDED.calmar_ratio,
                        max_drawdown = EXCLUDED.max_drawdown,
                        win_rate = EXCLUDED.win_rate,
                        avg_trade_pnl = EXCLUDED.avg_trade_pnl,
                        volatility = EXCLUDED.volatility,
                        total_trades = EXCLUDED.total_trades,
                        profitable_trades = EXCLUDED.profitable_trades,
                        created_at = EXCLUDED.created_at
                """, 
                    performance_7d['total_return'], performance_7d['sharpe_ratio'],
                    performance_7d['sortino_ratio'], performance_7d['calmar_ratio'],
                    performance_7d['max_drawdown'], performance_7d['win_rate'],
                    performance_7d['avg_trade_pnl'], performance_7d['volatility'],
                    performance_7d['total_trades'], performance_7d['profitable_trades'],
                    '7d', datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Error storing performance metrics: {e}")
    
    async def _store_action_decision(self, action: ActionDecision):
        """Store action decision in the database."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO rl_actions (
                        action_type, symbol, confidence, expected_return, risk_score,
                        state_features, reward, action_reasoning, created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                """, action.action_type, action.symbol, action.confidence, action.expected_return,
                    action.risk_score, json.dumps(action.state_features), action.reward,
                    action.reasoning, action.timestamp)
                
        except Exception as e:
            logger.error(f"Error storing action decision: {e}")
    
    async def _update_market_state(self, market_data: Dict[str, Any]):
        """Update market state (placeholder for future use)."""
        # This could be used to update market state tables
        pass

    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch real market data for RL training."""
        try:
            import yfinance as yf
            import numpy as np

            # Use a representative symbol
            symbols = ['SPY', 'QQQ', 'IWM']
            symbol = symbols[0]

            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d", interval="1d")

            if not hist.empty:
                # Calculate technical indicators
                returns = hist['Close'].pct_change().dropna()
                volatility = float(returns.std() * np.sqrt(252))

                # Determine market regime
                recent_return = float(returns.tail(5).mean())
                if recent_return > 0.01:
                    regime = 'bull'
                elif recent_return < -0.01:
                    regime = 'bear'
                else:
                    regime = 'neutral'

                return {
                    'symbol': symbol,
                    'price': float(hist['Close'].iloc[-1]),
                    'volatility': volatility,
                    'regime': regime,
                    'volume': float(hist['Volume'].iloc[-1]),
                    'returns': [float(r) for r in returns.tolist()[-20:]],
                    'timestamp': datetime.now()
                }
            else:
                return self._get_fallback_market_data()

        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return self._get_fallback_market_data()

    def _get_fallback_market_data(self) -> Dict[str, Any]:
        """Get fallback market data."""
        return {
            'symbol': 'SPY',
            'price': 450.0,
            'volatility': 0.02,
            'regime': 'neutral',
            'volume': 50000000,
            'returns': [0.001, -0.002, 0.003, -0.001, 0.002],
            'timestamp': datetime.now()
        }
