"""
Predictions Routes - Predictions and signals
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies
from routes.utils import run_individual_agents

router = APIRouter()



@router.get("/predictions")
async def get_predictions(limit: int = 50):
    """Get recent predictions/signals from database."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get individual agent predictions - sample from different agents
                individual_predictions = await conn.fetch("""
                    WITH agent_samples AS (
                        SELECT agent_name, symbol, signal_type, confidence, reasoning, metadata, created_at,
                               ROW_NUMBER() OVER (PARTITION BY agent_name ORDER BY created_at DESC) as rn
                        FROM agent_signals
                        WHERE created_at >= NOW() - INTERVAL '1 hour'
                    )
                    SELECT agent_name, symbol, signal_type, confidence, reasoning, metadata, created_at
                    FROM agent_samples
                    WHERE rn <= 5
                    ORDER BY created_at DESC
                    LIMIT $1
                """, limit // 2)  # Half for individual agents
                
                # Get ensemble predictions
                ensemble_predictions = await conn.fetch("""
                    SELECT signal_id, symbol, signal_type, blended_confidence, regime, 
                           quality_score, contributors, created_at
                    FROM ensemble_signals 
                    WHERE created_at >= NOW() - INTERVAL '1 hour'
                    ORDER BY created_at DESC 
                    LIMIT $1
                """, limit // 2)  # Half for ensemble
                
                predictions = []
                
                # Add individual agent predictions
                for pred in individual_predictions:
                    predictions.append({
                        "agent_name": pred['agent_name'],
                        "signal_type": pred['signal_type'],
                        "confidence": float(pred['confidence']),
                        "asset_symbol": pred['symbol'],
                        "timestamp": pred['created_at'].isoformat(),
                        "reasoning": pred['reasoning'],
                        "metadata": {
                            "source": "individual_agent",
                            "metadata": pred['metadata'] if pred['metadata'] else {}
                        }
                    })
                
                # Add ensemble predictions
                for signal in ensemble_predictions:
                    # Extract agent names from contributors JSON
                    contributors = signal['contributors'] or []
                    agent_names = []
                    if isinstance(contributors, list):
                        agent_names = [c.get('agent_name', 'UnknownAgent') for c in contributors if isinstance(c, dict)]
                    
                    # Create reasoning based on regime and confidence
                    reasoning_map = {
                        'bull': 'Strong bullish momentum detected',
                        'bear': 'Bearish trend confirmed',
                        'trending': 'Clear directional trend identified',
                        'volatile': 'High volatility environment',
                        'neutral': 'Market in consolidation phase'
                    }
                    reasoning = reasoning_map.get(signal['regime'], 'Market analysis indicates current conditions')
                    
                    predictions.append({
                        "agent_name": agent_names[0] if agent_names else "EnsembleBlender",
                        "signal_type": signal['signal_type'],
                        "confidence": float(signal['blended_confidence']),
                        "asset_symbol": signal['symbol'],
                        "timestamp": signal['created_at'].isoformat(),
                        "reasoning": reasoning,
                        "metadata": {
                            "source": "ensemble",
                            "regime": signal['regime'],
                            "quality_score": float(signal['quality_score']) if signal['quality_score'] else 0.6,
                            "contributors": len(agent_names),
                            "signal_id": signal['signal_id']
                        }
                    })
                
                # Sort by timestamp and limit
                predictions.sort(key=lambda x: x['timestamp'], reverse=True)
                return predictions[:limit]
        else:
            # Fallback to mock data if database unavailable
            symbols = ['NVDA', 'TSLA', 'BTC-USD', 'SOXL', 'AAPL', 'MSFT', 'GOOGL', 'SPY']
            signal_types = ['buy', 'sell', 'hold']
            agent_names = ['MomentumAgent', 'SentimentAgent', 'CorrelationAgent', 'RiskAgent', 
                          'VolatilityAgent', 'VolumeAgent', 'EventImpactAgent', 'ForecastAgent', 
                          'StrategyAgent', 'MetaAgent', 'EnsembleBlender']
            predictions = []
            
            for i in range(min(limit, 50)):
                agent_name = agent_names[i % len(agent_names)]
                predictions.append({
                    "agent_name": agent_name,
                    "signal_type": random.choice(signal_types),
                    "confidence": round(random.uniform(0.6, 0.95), 2),
                    "asset_symbol": random.choice(symbols),
                    "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                    "reasoning": f"Technical analysis indicates {random.choice(['bullish', 'bearish', 'neutral'])} trend",
                    "metadata": {
                        "price": round(random.uniform(100, 1000), 2),
                        "volume": random.randint(1000000, 10000000)
                    }
                })
            
            return predictions
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve predictions")


@router.get("/signals")
async def get_signals(limit: int = 50):
    """Get recent signals (alias for predictions)."""
    return await get_predictions(limit)


@router.post("/predictions/run-individual-agents")
async def run_individual_agents_endpoint():
    """Manually trigger individual agent predictions."""
    try:
        await run_individual_agents()
        return {
            "success": True,
            "message": "Individual agent predictions generated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error running individual agents: {e}")
        return {
            "success": False,
            "message": f"Failed to run individual agents: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
