#!/usr/bin/env python3
"""
AI Market Analysis System - Final Complete System (Modular Version)
Combines PostgreSQL database with full agent system functionality on port 8001.
"""

import asyncio
import os
import sys
from pathlib import Path
from loguru import logger
from datetime import datetime

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import FastAPI and other components
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# Import route modules
from routes import health, agent_monitor, agent_router, execution_agent
from routes import rag_event_agent, rl_strategy, meta_evaluation, latent_pattern
from routes import ensemble_blender, predictions, symbols
from routes import ticker_discovery, forecasting, risk_analysis, ab_testing, reports
from routes import dependencies
from routes.utils import run_individual_agents, run_automated_ticker_discovery

# Global variables
STARTUP_TIME = datetime.now()

# Global schedulers
ticker_discovery_scheduler = None
individual_agent_scheduler = None


async def start_individual_agent_scheduler():
    """Initialize and start individual agent scheduler every 30 minutes."""
    try:
        global individual_agent_scheduler
        if not individual_agent_scheduler:
            individual_agent_scheduler = AsyncIOScheduler()
            
            # Run every 30 minutes during market hours
            individual_agent_scheduler.add_job(
                run_individual_agents,
                'interval',
                minutes=30,
                id='individual_agents_30min',
                name='Individual Agents 30min'
            )
            
            individual_agent_scheduler.start()
            logger.info("✅ Individual agent scheduler started (runs every 30 minutes)")
    except Exception as e:
        logger.error(f"❌ Failed to start individual agent scheduler: {e}")


async def start_ticker_discovery_scheduler():
    """Initialize and start ticker discovery scheduler at 2x per day."""
    try:
        global ticker_discovery_scheduler
        ticker_discovery_scheduler = AsyncIOScheduler()
        
        # Schedule at 09:30 and 15:30 daily (market hours)
        ticker_discovery_scheduler.add_job(
            run_automated_ticker_discovery,
            trigger=CronTrigger(hour='9,15', minute='30'),
            id='automated_ticker_discovery',
            name='Automated Ticker Discovery',
            replace_existing=True
        )
        
        ticker_discovery_scheduler.start()
        logger.info("✅ Ticker discovery scheduler started (runs at 09:30 and 15:30 daily)")
        
    except Exception as e:
        logger.error(f"❌ Failed to start ticker discovery scheduler: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("🚀 Starting AI Market Analysis System v4.18.1 (Modular Architecture)...")
    
    try:
        # Initialize PostgreSQL connection
        import asyncpg
        host = os.getenv('POSTGRES_HOST', 'localhost')  
        port = int(os.getenv('POSTGRES_PORT', '5433' if host == 'localhost' else '5432'))
        
        dependencies.db_pool = await asyncpg.create_pool(
            host=host,
            port=port,
            database=os.getenv('POSTGRES_DB', 'ai_market_system'),
            user=os.getenv('POSTGRES_USER', 'postgres'),
            password=os.getenv('POSTGRES_PASSWORD', 'password'),
            min_size=5,
            max_size=20
        )
        logger.info("✅ PostgreSQL connection pool created")
        
        # Test connection
        async with dependencies.db_pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM symbols")
            logger.info(f"✅ Database test successful - {result} symbols found")
        
        # Initialize real data service
        from services.real_data_service import RealDataService, RealDataConfig
        real_data_config = RealDataConfig(
            symbols=['BTC-USD', 'SOXL', 'NVDA', 'RIVN', 'TSLA', 'META', 'AMD', 'INTC', 'SPY', 'TQQQ'],
            enable_real_time=True
        )
        dependencies.real_data_service = RealDataService(real_data_config)
        await dependencies.real_data_service.start()
        logger.info("✅ Real data service initialized and started")
        
        # Initialize enhanced data sources
        try:
            from data.enhanced_data_sources import create_enhanced_data_manager
            dependencies.enhanced_data_manager = create_enhanced_data_manager()
            logger.info("✅ Enhanced data sources initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced data sources: {e}")
        
        # Initialize alternative data sources
        try:
            from data.alternative_data_sources import create_alternative_data_manager
            dependencies.alternative_data_manager = create_alternative_data_manager()
            logger.info("✅ Alternative data sources initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize alternative data sources: {e}")
        
        # Initialize data quality system
        try:
            from data.data_quality_validator import create_data_quality_system
            data_quality_validator, data_enhancer, data_lineage_tracker = create_data_quality_system()
            dependencies.data_quality_validator = data_quality_validator
            logger.info("✅ Data quality system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize data quality system: {e}")
        
        # Initialize advanced ML models
        try:
            from ml import create_advanced_models
            if create_advanced_models:
                dependencies.advanced_ml_manager = create_advanced_models()
                logger.info("✅ Advanced ML models initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize advanced ML models: {e}")
        
        # Initialize model interpretability
        try:
            from ml import create_model_interpretability
            if create_model_interpretability:
                dependencies.model_interpretability = create_model_interpretability()
                logger.info("✅ Model interpretability system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize model interpretability: {e}")
        
        # Initialize real-time learning
        try:
            from ml import create_real_time_learning_system
            if create_real_time_learning_system:
                dependencies.real_time_learning_manager = create_real_time_learning_system()
                logger.info("✅ Real-time learning system initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize real-time learning: {e}")
        
        # Initialize RL Training Pipeline
        try:
            from services.rl_data_collector import RLDataCollector
            from services.rl_training_service import RLTrainingService
            
            dependencies.rl_data_collector = RLDataCollector(dependencies.db_pool)
            dependencies.rl_training_service = RLTrainingService(dependencies.db_pool)
            
            await dependencies.rl_data_collector.start_collection()
            await dependencies.rl_training_service.start_training()
            logger.info("✅ RL Training Pipeline initialized and started")
        except Exception as e:
            logger.error(f"❌ Failed to initialize RL Training Pipeline: {e}")
        
        # Initialize Meta-Evaluation Service
        try:
            from services.meta_evaluation_service import MetaEvaluationService
            dependencies.meta_evaluation_service = MetaEvaluationService(dependencies.db_pool)
            await dependencies.meta_evaluation_service.start_evaluation()
            logger.info("✅ Meta-Evaluation Service initialized and started")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Meta-Evaluation Service: {e}")
        
        # Initialize Latent Pattern Service
        try:
            from services.latent_pattern_service import LatentPatternService
            dependencies.latent_pattern_service = LatentPatternService(dependencies.db_pool)
            await dependencies.latent_pattern_service.start_pattern_detection()
            logger.info("✅ Latent Pattern Service initialized and started")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Latent Pattern Service: {e}")
        
        # Initialize Ensemble Blender Service
        try:
            from services.ensemble_blender_service import EnsembleBlenderService
            dependencies.ensemble_blender_service = EnsembleBlenderService(dependencies.db_pool)
            await dependencies.ensemble_blender_service.start_ensemble_blending()
            logger.info("✅ Ensemble Blender Service initialized and started")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Ensemble Blender Service: {e}")
        
        # Initialize Enhanced Forecasting Service
        try:
            if dependencies.db_pool:
                from services.enhanced_forecasting_service import EnhancedForecastingService
                dependencies.enhanced_forecasting_service = EnhancedForecastingService(dependencies.db_pool)
                logger.info("✅ Enhanced Forecasting Service initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Enhanced Forecasting Service: {e}")
        
        # Initialize and start Automated RAG Service
        try:
            from services.automated_rag_service import AutomatedRAGService
            dependencies.automated_rag_service = AutomatedRAGService(dependencies.db_pool, fetch_interval_minutes=30)
            await dependencies.automated_rag_service.start()
            logger.info("✅ Automated RAG Service initialized and started (30-minute updates)")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Automated RAG Service: {e}")
        
        # Start schedulers
        await start_individual_agent_scheduler()
        await start_ticker_discovery_scheduler()
        
        logger.info("🎉 System initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down system...")
    
    if dependencies.real_data_service:
        dependencies.real_data_service.stop()
        logger.info("🛑 Real data service stopped")
    
    if dependencies.rl_training_service:
        await dependencies.rl_training_service.stop_training()
        logger.info("🛑 RL training service stopped")
    
    if dependencies.rl_data_collector:
        await dependencies.rl_data_collector.stop_collection()
        logger.info("🛑 RL data collector stopped")
    
    if dependencies.meta_evaluation_service:
        await dependencies.meta_evaluation_service.stop_evaluation()
        logger.info("🛑 Meta-evaluation service stopped")
    
    if dependencies.latent_pattern_service:
        await dependencies.latent_pattern_service.stop_pattern_detection()
        logger.info("🛑 Latent pattern service stopped")
    
    if dependencies.ensemble_blender_service:
        await dependencies.ensemble_blender_service.stop_ensemble_blending()
        logger.info("🛑 Ensemble blender service stopped")
    
    if dependencies.automated_rag_service:
        await dependencies.automated_rag_service.stop()
        logger.info("🛑 Automated RAG service stopped")
    
    if individual_agent_scheduler:
        individual_agent_scheduler.shutdown()
        logger.info("🛑 Individual agent scheduler stopped")
    
    if ticker_discovery_scheduler:
        ticker_discovery_scheduler.shutdown()
        logger.info("🛑 Ticker discovery scheduler stopped")
    
    if dependencies.real_time_learning_manager:
        dependencies.real_time_learning_manager.stop_learning()
        logger.info("🛑 Real-time learning system stopped")
    
    if dependencies.db_pool:
        await dependencies.db_pool.close()
        logger.info("🛑 Database connection pool closed")


# Create FastAPI app
app = FastAPI(
    title="AI Market Analysis System",
    description="Advanced AI-powered market analysis and trading system with PostgreSQL",
    version="4.18.1",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route modules
app.include_router(health.router, tags=["Health & Status"])
app.include_router(agent_monitor.router, tags=["Agent Monitor"])
app.include_router(agent_router.router, tags=["Agent Router"])
app.include_router(execution_agent.router, tags=["Execution Agent"])
app.include_router(rag_event_agent.router, tags=["RAG Event Agent"])
app.include_router(rl_strategy.router, tags=["RL Strategy"])
app.include_router(meta_evaluation.router, tags=["Meta Evaluation"])
app.include_router(latent_pattern.router, tags=["Latent Pattern"])
app.include_router(ensemble_blender.router, tags=["Ensemble Blender"])
app.include_router(predictions.router, tags=["Predictions"])
app.include_router(symbols.router, tags=["Symbols"])
app.include_router(ticker_discovery.router, tags=["Ticker Discovery"])
app.include_router(forecasting.router, tags=["Forecasting"])
app.include_router(risk_analysis.router, tags=["Risk Analysis"])
app.include_router(ab_testing.router, tags=["A/B Testing"])
app.include_router(reports.router, tags=["Reports"])

logger.info("✅ All route modules registered")


if __name__ == "__main__":
    logger.info("Starting AI Market Analysis System API v4.18.1 (Modular Architecture)...")
    logger.info("Database: PostgreSQL (localhost:5433)")
    logger.info("API: http://localhost:8001")
    logger.info("Health: http://localhost:8001/health")
    logger.info("Status: http://localhost:8001/status")
    logger.info("Docs: http://localhost:8001/docs")
    
    uvicorn.run(
        "start_system_final:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )

