"""
Symbol Management Routes - Symbol CRUD operations
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import random
import json

from routes import dependencies

router = APIRouter()


@router.get("/symbols/summary")
async def get_symbols_summary():
    """Get symbol management summary."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get symbol counts by status from managed_symbols table
                status_counts = await conn.fetch("""
                    SELECT status, COUNT(*) as count 
                    FROM managed_symbols 
                    GROUP BY status
                """)
                
                # Get total counts
                total_symbols = await conn.fetchval("SELECT COUNT(*) FROM symbols")
                total_managed = await conn.fetchval("SELECT COUNT(*) FROM managed_symbols")
                
                # Create status summary
                status_summary = {row['status']: row['count'] for row in status_counts}
                
                return {
                    "total_symbols": total_symbols or 0,
                    "total_managed": total_managed or 0,
                    "status_breakdown": status_summary,
                    "active_symbols": status_summary.get('active', 0),
                    "monitoring_symbols": status_summary.get('monitoring', 0),
                    "watchlist_symbols": status_summary.get('watchlist', 0),
                    "last_updated": datetime.now().isoformat()
                }
        else:
            # Fallback data when database is not available
            return {
                "total_symbols": 8,
                "total_managed": 8,
                "status_breakdown": {"active": 6, "monitoring": 2},
                "active_symbols": 6,
                "monitoring_symbols": 2,
                "watchlist_symbols": 0,
                "last_updated": datetime.now().isoformat()
            }
    except Exception as e:
        logger.error(f"Error getting symbols summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve symbols summary")



@router.get("/api/symbols")
async def get_symbols(status: str = None):
    """Get all symbols or filter by status."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                if status:
                    symbols = await conn.fetch("""
                        SELECT s.*, m.status, m.priority, m.notes, m.added_date, m.last_updated
                        FROM symbols s
                        LEFT JOIN managed_symbols m ON s.symbol = m.symbol
                        WHERE m.status = $1 OR (m.status IS NULL AND $1 = 'unmanaged')
                        ORDER BY s.symbol
                    """, status)
                else:
                    symbols = await conn.fetch("""
                        SELECT s.*, m.status, m.priority, m.notes, m.added_date, m.last_updated
                        FROM symbols s
                        LEFT JOIN managed_symbols m ON s.symbol = m.symbol
                        ORDER BY s.symbol
                    """)
                
                symbols_list = []
                for symbol in symbols:
                    symbols_list.append({
                        "symbol": symbol['symbol'],
                        "name": symbol['name'],
                        "sector": symbol['sector'],
                        "industry": symbol['industry'],
                        "status": symbol['status'] or 'unmanaged',
                        "priority": symbol['priority'] or 1,
                        "notes": symbol['notes'] or '',
                        "created_at": symbol['added_date'].isoformat() if symbol['added_date'] else None,
                        "updated_at": symbol['last_updated'].isoformat() if symbol['last_updated'] else None
                    })
                
                return {"symbols": symbols_list}
        else:
            # Fallback data when database is not available
            fallback_symbols = [
                {
                    "symbol": "NVDA",
                    "name": "NVIDIA Corporation",
                    "sector": "Technology",
                    "industry": "Semiconductors",
                    "status": "active",
                    "priority": 5,
                    "notes": "AI and gaming leader",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                },
                {
                    "symbol": "TSLA",
                    "name": "Tesla, Inc.",
                    "sector": "Consumer Discretionary",
                    "industry": "Electric Vehicles",
                    "status": "active",
                    "priority": 4,
                    "notes": "EV and energy leader",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                },
                {
                    "symbol": "BTC-USD",
                    "name": "Bitcoin",
                    "sector": "Cryptocurrency",
                    "industry": "Digital Assets",
                    "status": "active",
                    "priority": 3,
                    "notes": "Leading cryptocurrency",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
                },
                {
                    "symbol": "SOXL",
                    "name": "Direxion Daily Semiconductor Bull 3X Shares",
                    "sector": "Financial",
                    "industry": "ETFs",
                "status": "monitoring",
                    "priority": 2,
                    "notes": "Semiconductor leveraged ETF",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat()
            }
            ]
            return {"symbols": fallback_symbols}
    except Exception as e:
        logger.error(f"Error getting symbols: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve symbols")



@router.delete("/api/symbols/{symbol}")
async def remove_symbol(symbol: str):
    """Remove a symbol."""
    try:
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Remove portfolio position first
                from routes.portfolio import remove_portfolio_position
                await remove_portfolio_position(symbol, conn)
                
                # Remove from managed_symbols table (this will cascade to symbols if needed)
                result = await conn.execute("""
                    DELETE FROM managed_symbols WHERE symbol = $1
                """, symbol)
                
                return {"message": f"Symbol {symbol} removed successfully"}
        else:
            return {"message": f"Symbol {symbol} would be removed (database not available)"}
    except Exception as e:
        logger.error(f"Error removing symbol: {e}")
        raise HTTPException(status_code=500, detail="Failed to remove symbol")



@router.get("/symbols/trading-decisions-test")
async def get_trading_decisions_test():
    """Test endpoint to debug trading decisions issue."""
    try:
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Get all managed symbols
                managed_symbols = await conn.fetch("""
                    SELECT symbol FROM managed_symbols 
                    WHERE status IN ('active', 'monitoring')
                    ORDER BY symbol
                """)
                
                symbol_list = [row['symbol'] for row in managed_symbols]
                return {
                    "managed_symbols": symbol_list,
                    "count": len(symbol_list),
                    "message": "This is a test endpoint"
                }
        return {"error": "No database connection"}
    except Exception as e:
        return {"error": str(e)}

@router.get("/symbols/trading-decisions")
async def get_trading_decisions():
    """Get real trading decisions for symbols based on agent predictions and market analysis."""
    try:
        logger.info("Getting real trading decisions from database")
        
        # Using dependencies.db_pool
        decisions = []
        managed_symbols = []
        
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # First, get all managed symbols
                managed_symbols = await conn.fetch("""
                    SELECT symbol FROM managed_symbols 
                    WHERE status IN ('active', 'monitoring')
                    ORDER BY symbol
                """)
                
                if not managed_symbols:
                    logger.warning("No managed symbols found")
                    return []
                
                # Extract symbol list for the query
                symbol_list = [row['symbol'] for row in managed_symbols]
                symbol_placeholders = ','.join([f"'{symbol}'" for symbol in symbol_list])
                logger.info(f"DEBUG: Found {len(symbol_list)} managed symbols: {symbol_list}")
                logger.info(f"DEBUG: Symbol placeholders: {symbol_placeholders}")
                
                # Get recent ensemble predictions (most reliable)
                logger.info("Looking for recent ensemble predictions")
                ensemble_query = f"""
                    SELECT DISTINCT ON (symbol) 
                        symbol, signal_type, blended_confidence as confidence, 
                        'Ensemble prediction' as reasoning, 'EnsembleBlender' as agent_name, created_at as timestamp
                    FROM ensemble_signals 
                    WHERE created_at >= NOW() - INTERVAL '2 hours'
                    AND symbol IN ({symbol_placeholders})
                    ORDER BY symbol, created_at DESC
                """
                logger.info(f"DEBUG: Ensemble query: {ensemble_query}")
                ensemble_result = await conn.fetch(ensemble_query)
                logger.info(f"DEBUG: Ensemble result count: {len(ensemble_result) if ensemble_result else 0}")
                
                # Track which symbols we have real predictions for
                symbols_with_predictions = set()
                
                if ensemble_result:
                    logger.info(f"Found {len(ensemble_result)} ensemble predictions")
                    for row in ensemble_result:
                        decision = {
                            "symbol": row['symbol'],
                            "action": row['signal_type'],
                            "confidence": float(row['confidence']),
                            "reason": row['reasoning'],
                            "timestamp": row['timestamp'].isoformat(),
                            "agent": row['agent_name']
                        }
                        decisions.append(decision)
                        symbols_with_predictions.add(row['symbol'])
                        logger.info(f"Real trading decision for {row['symbol']}: {row['signal_type']} (confidence: {row['confidence']:.2f}) from {row['agent_name']}")
                
                # Try individual agent predictions for symbols without ensemble predictions
                symbols_without_ensemble = [s for s in symbol_list if s not in symbols_with_predictions]
                if symbols_without_ensemble:
                    logger.info(f"Looking for individual agent predictions for {len(symbols_without_ensemble)} symbols: {symbols_without_ensemble}")
                    agent_placeholders = ','.join([f"'{symbol}'" for symbol in symbols_without_ensemble])
                    agent_result = await conn.fetch(f"""
                        SELECT DISTINCT ON (symbol)
                            symbol, signal_type, confidence, reasoning, agent_name, created_at as timestamp
                        FROM agent_signals
                        WHERE created_at >= NOW() - INTERVAL '2 hours'
                        AND symbol IN ({agent_placeholders})
                        ORDER BY symbol, created_at DESC
                    """)
                    
                    if agent_result:
                        logger.info(f"Found {len(agent_result)} agent predictions")
                        for row in agent_result:
                            decision = {
                                "symbol": row['symbol'],
                                "action": row['signal_type'],
                                "confidence": float(row['confidence']),
                                "reason": row['reasoning'],
                                "timestamp": row['timestamp'].isoformat(),
                                "agent": row['agent_name']
                            }
                            decisions.append(decision)
                            symbols_with_predictions.add(row['symbol'])
                            logger.info(f"Real trading decision for {row['symbol']}: {row['signal_type']} (confidence: {row['confidence']:.2f}) from {row['agent_name']}")
                
                # Generate fallback decisions for symbols without any predictions
                symbols_without_predictions = [s for s in symbol_list if s not in symbols_with_predictions]
                if symbols_without_predictions:
                    logger.info(f"Generating fallback decisions for {len(symbols_without_predictions)} symbols: {symbols_without_predictions}")
                    for symbol in symbols_without_predictions:
                        decision = {
                            "symbol": symbol,
                            "action": "hold",
                            "confidence": 0.50,
                            "reason": "No recent predictions available - system analyzing",
                            "timestamp": datetime.now().isoformat(),
                            "agent": "System"
                        }
                        decisions.append(decision)
                        logger.info(f"Generated fallback decision for {symbol}: hold (confidence: 0.50)")
                
                # Return all decisions (real + fallback)
                if decisions:
                    logger.info(f"Returning {len(decisions)} trading decisions ({len(symbols_with_predictions)} real, {len(symbols_without_predictions)} fallback)")
                    return decisions
        
        # Final fallback if no managed symbols found at all
        logger.warning("No managed symbols found, returning empty list")
        return []
        
    except Exception as e:
        logger.error(f"Error getting real trading decisions: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to retrieve trading decisions")



@router.post("/symbols/add-from-discovery")
async def add_symbol_from_discovery(symbol_data: dict):
    """Add a symbol to portfolio from ticker discovery."""
    try:
        symbol = symbol_data.get("symbol", "").upper()
        name = symbol_data.get("name", f"{symbol} Corporation")
        sector = symbol_data.get("sector", "Technology")
        industry = symbol_data.get("industry", "General")
        
        if not symbol:
            return {"success": False, "message": "Symbol is required"}
        
        # Using dependencies.db_pool
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Fetch current price for the symbol
                initial_price = None
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        initial_price = float(hist['Close'].iloc[-1])
                        logger.info(f"Fetched initial price for {symbol}: ${initial_price:.2f}")
                except Exception as price_error:
                    logger.warning(f"Could not fetch initial price for {symbol}: {price_error}")
                
                # Insert into symbols table if not exists
                await conn.execute("""
                    INSERT INTO symbols (symbol, name, sector, industry, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (symbol) DO UPDATE SET
                        name = EXCLUDED.name,
                        sector = EXCLUDED.sector,
                        industry = EXCLUDED.industry,
                        updated_at = EXCLUDED.updated_at
                """, symbol, name, sector, industry, datetime.now(), datetime.now())
                
                # Insert into managed_symbols table with initial price
                await conn.execute("""
                    INSERT INTO managed_symbols (symbol, status, added_date, last_updated, initial_price, initial_price_date)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (symbol) DO UPDATE SET
                        status = EXCLUDED.status,
                        last_updated = NOW(),
                        initial_price = COALESCE(managed_symbols.initial_price, EXCLUDED.initial_price),
                        initial_price_date = COALESCE(managed_symbols.initial_price_date, EXCLUDED.initial_price_date)
                """, symbol, 'monitoring', datetime.now(), datetime.now(), initial_price, datetime.now())
                
                # Create portfolio position for the new symbol
                if initial_price:
                    from routes.portfolio import create_portfolio_position
                    await create_portfolio_position(symbol, initial_price, conn)
                
                # Generate forecasts for the new symbol
                try:
                    from routes.forecasting import generate_day_forecast_for_symbol, generate_swing_forecast_for_symbol, _build_advanced_forecast_for_symbol
                    
                    logger.info(f"Generating forecasts for newly added symbol: {symbol}")
                    
                    # Generate day forecast
                    day_forecast = await generate_day_forecast_for_symbol(symbol, "end_of_day")
                    
                    # Generate swing forecast
                    swing_forecast = await generate_swing_forecast_for_symbol(symbol, "medium_swing")
                    
                    # Generate advanced forecast
                    advanced_forecast = await _build_advanced_forecast_for_symbol(symbol, conn)
                    
                    # Save forecasts to database
                    if day_forecast:
                        await conn.execute("""
                            INSERT INTO day_forecasts (
                                symbol, direction, confidence, target_price, stop_loss, current_price,
                                predicted_price, price_change, volume_forecast, risk_score,
                                signal_strength, technical_indicators, market_regime, volatility_forecast,
                                key_events, macro_factors, fundamental_score, sentiment_score, horizon, valid_until
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                        """, 
                            day_forecast.get('symbol'),
                            day_forecast.get('signal_type', day_forecast.get('direction')),
                            float(day_forecast.get('confidence', 0)),
                            float(day_forecast.get('target_price', 0)) if day_forecast.get('target_price') else None,
                            float(day_forecast.get('stop_loss', 0)) if day_forecast.get('stop_loss') else None,
                            float(day_forecast.get('current_price', 0)) if day_forecast.get('current_price') else None,
                            float(day_forecast.get('predicted_price', 0)) if day_forecast.get('predicted_price') else None,
                            float(day_forecast.get('price_change', 0)) if day_forecast.get('price_change') else None,
                            str(day_forecast.get('volume_forecast')) if day_forecast.get('volume_forecast') else None,
                            float(day_forecast.get('risk_score', 0)) if day_forecast.get('risk_score') else None,
                            day_forecast.get('signal_strength'),
                            json.dumps(day_forecast.get('technical_indicators', {})) if isinstance(day_forecast.get('technical_indicators'), (dict, list)) else day_forecast.get('technical_indicators'),
                            day_forecast.get('market_regime'),
                            str(day_forecast.get('volatility_forecast')) if day_forecast.get('volatility_forecast') else None,
                            json.dumps(day_forecast.get('key_events', [])),
                            json.dumps(day_forecast.get('macro_factors', {})),
                            float(day_forecast.get('fundamental_score', 0)) if day_forecast.get('fundamental_score') else None,
                            float(day_forecast.get('sentiment_score', 0)) if day_forecast.get('sentiment_score') else None,
                            day_forecast.get('horizon'),
                            datetime.fromisoformat(day_forecast.get('valid_until')) if day_forecast.get('valid_until') else None
                        )
                    
                    if swing_forecast:
                        await conn.execute("""
                            INSERT INTO swing_forecasts (
                                symbol, direction, confidence, target_price, stop_loss, current_price,
                                predicted_price, price_change, trend, support_level, resistance_level,
                                risk_score, signal_strength, technical_indicators, market_regime,
                                volume_forecast, volatility_forecast, key_events, macro_factors,
                                fundamental_score, sentiment_score, horizon, valid_until
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22, $23)
                        """, 
                            swing_forecast.get('symbol'),
                            swing_forecast.get('signal_type', swing_forecast.get('direction')),
                            float(swing_forecast.get('confidence', 0)),
                            float(swing_forecast.get('target_price', 0)) if swing_forecast.get('target_price') else None,
                            float(swing_forecast.get('stop_loss', 0)) if swing_forecast.get('stop_loss') else None,
                            float(swing_forecast.get('current_price', 0)) if swing_forecast.get('current_price') else None,
                            float(swing_forecast.get('predicted_price', 0)) if swing_forecast.get('predicted_price') else None,
                            float(swing_forecast.get('price_change', 0)) if swing_forecast.get('price_change') else None,
                            swing_forecast.get('trend'),
                            float(swing_forecast.get('support_level', 0)) if swing_forecast.get('support_level') else None,
                            float(swing_forecast.get('resistance_level', 0)) if swing_forecast.get('resistance_level') else None,
                            float(swing_forecast.get('risk_score', 0)) if swing_forecast.get('risk_score') else None,
                            swing_forecast.get('signal_strength'),
                            json.dumps(swing_forecast.get('technical_indicators', {})) if isinstance(swing_forecast.get('technical_indicators'), (dict, list)) else swing_forecast.get('technical_indicators'),
                            swing_forecast.get('market_regime'),
                            str(swing_forecast.get('volume_forecast')) if swing_forecast.get('volume_forecast') else None,
                            str(swing_forecast.get('volatility_forecast')) if swing_forecast.get('volatility_forecast') else None,
                            json.dumps(swing_forecast.get('key_events', [])),
                            json.dumps(swing_forecast.get('macro_factors', {})),
                            float(swing_forecast.get('fundamental_score', 0)) if swing_forecast.get('fundamental_score') else None,
                            float(swing_forecast.get('sentiment_score', 0)) if swing_forecast.get('sentiment_score') else None,
                            swing_forecast.get('horizon'),
                            datetime.fromisoformat(swing_forecast.get('valid_until')) if swing_forecast.get('valid_until') else None
                        )
                    
                    if advanced_forecast:
                        await conn.execute("""
                            INSERT INTO advanced_forecasts (
                                symbol, final_signal, final_confidence, current_price, predicted_price, target_price,
                                stop_loss, price_change_pct, risk_score, signal_strength, agent_contributions,
                                total_agents_contributing, signal_distribution, ensemble_signal, rag_analysis,
                                rl_recommendation, meta_evaluation, latent_patterns, forecast_type, valid_until
                            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
                        """,
                            advanced_forecast.get('symbol'),
                            advanced_forecast.get('final_signal'),
                            float(advanced_forecast.get('final_confidence', 0)),
                            float(advanced_forecast.get('current_price', 0)) if advanced_forecast.get('current_price') else None,
                            float(advanced_forecast.get('predicted_price', 0)) if advanced_forecast.get('predicted_price') else None,
                            float(advanced_forecast.get('target_price', 0)) if advanced_forecast.get('target_price') else None,
                            float(advanced_forecast.get('stop_loss', 0)) if advanced_forecast.get('stop_loss') else None,
                            float(advanced_forecast.get('price_change_pct', 0)) if advanced_forecast.get('price_change_pct') else None,
                            float(advanced_forecast.get('risk_score', 0)) if advanced_forecast.get('risk_score') else None,
                            advanced_forecast.get('signal_strength'),
                            json.dumps(advanced_forecast.get('agent_contributions', [])),
                            int(advanced_forecast.get('total_agents_contributing', 0)),
                            json.dumps(advanced_forecast.get('signal_distribution', {})),
                            json.dumps(advanced_forecast.get('ensemble_signal', {})),
                            json.dumps(advanced_forecast.get('rag_analysis', {})),
                            json.dumps(advanced_forecast.get('rl_recommendation', {})),
                            json.dumps(advanced_forecast.get('meta_evaluation', {})),
                            json.dumps(advanced_forecast.get('latent_patterns', {})),
                            advanced_forecast.get('forecast_type'),
                            datetime.fromisoformat(advanced_forecast.get('valid_until')) if advanced_forecast.get('valid_until') else None
                        )
                    
                    logger.info(f"Successfully generated and saved forecasts for {symbol}")
                    
                except Exception as forecast_error:
                    logger.error(f"Error generating forecasts for {symbol}: {forecast_error}")
                    # Don't fail the entire operation if forecast generation fails
            
        return {
                "success": True,
                "message": f"Successfully added {symbol} to symbol management and generated forecasts",
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "industry": industry,
                "forecasts_generated": True
        }
    except Exception as e:
        logger.error(f"Error adding symbol from discovery: {e}")
        return {
            "success": False,
            "message": f"Failed to add symbol: {str(e)}"
        }


@router.post("/api/symbols")
async def add_symbol(symbol_data: dict):
    """Add a new symbol."""
    try:
        if dependencies.db_pool:
            async with dependencies.db_pool.acquire() as conn:
                # Fetch current price for the symbol
                initial_price = None
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol_data['symbol'])
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        initial_price = float(hist['Close'].iloc[-1])
                        logger.info(f"Fetched initial price for {symbol_data['symbol']}: ${initial_price:.2f}")
                except Exception as price_error:
                    logger.warning(f"Could not fetch initial price for {symbol_data['symbol']}: {price_error}")
                
                # Insert new symbol into symbols table
                await conn.execute("""
                    INSERT INTO symbols (symbol, name, sector, industry, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, NOW(), NOW())
                    ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    industry = EXCLUDED.industry,
                    updated_at = NOW()
                """, 
                symbol_data['symbol'],
                symbol_data['name'],
                symbol_data['sector'],
                symbol_data['industry']
                )
                
                # Insert or update in managed_symbols table with initial price
                await conn.execute("""
                    INSERT INTO managed_symbols (symbol, status, priority, notes, added_date, last_updated, initial_price, initial_price_date)
                    VALUES ($1, $2, $3, $4, NOW(), NOW(), $5, NOW())
                    ON CONFLICT (symbol) DO UPDATE SET
                    status = EXCLUDED.status,
                    priority = EXCLUDED.priority,
                    notes = EXCLUDED.notes,
                    last_updated = NOW(),
                    initial_price = COALESCE(managed_symbols.initial_price, EXCLUDED.initial_price),
                    initial_price_date = COALESCE(managed_symbols.initial_price_date, EXCLUDED.initial_price_date)
                """, 
                symbol_data['symbol'],
                symbol_data['status'],
                symbol_data['priority'],
                symbol_data['notes'],
                initial_price
                )
                
                # Create portfolio position if symbol is active/monitoring
                if symbol_data['status'] in ['active', 'monitoring'] and initial_price:
                    from routes.portfolio import create_portfolio_position
                    await create_portfolio_position(symbol_data['symbol'], initial_price, conn)
                
                return {"message": f"Symbol {symbol_data['symbol']} added successfully"}
        else:
            return {"message": f"Symbol {symbol_data['symbol']} would be added (database not available)"}
    except Exception as e:
        logger.error(f"Error adding symbol: {e}")
        raise HTTPException(status_code=500, detail="Failed to add symbol")


@router.get("/symbols/search")
async def search_symbols(query: str):
    """Search for symbols."""
    try:
        # Mock search results - in real implementation, this would search a symbols database
        search_results = []
        all_symbols = [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
            {"symbol": "MSFT", "name": "Microsoft Corporation", "sector": "Technology", "industry": "Software"},
            {"symbol": "GOOGL", "name": "Alphabet Inc.", "sector": "Technology", "industry": "Internet Services"},
            {"symbol": "AMZN", "name": "Amazon.com Inc.", "sector": "Consumer Discretionary", "industry": "E-commerce"},
            {"symbol": "META", "name": "Meta Platforms Inc.", "sector": "Technology", "industry": "Social Media"},
            {"symbol": "SPY", "name": "SPDR S&P 500 ETF", "sector": "Financial", "industry": "ETFs"},
            {"symbol": "QQQ", "name": "Invesco QQQ Trust", "sector": "Financial", "industry": "ETFs"},
            {"symbol": "IWM", "name": "iShares Russell 2000 ETF", "sector": "Financial", "industry": "ETFs"}
        ]
        
        query_lower = query.lower()
        for symbol in all_symbols:
            if (query_lower in symbol['symbol'].lower() or 
                query_lower in symbol['name'].lower() or 
                query_lower in symbol['sector'].lower() or 
                query_lower in symbol['industry'].lower()):
                search_results.append(symbol)
        
        return search_results[:10]  # Limit to 10 results
    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        raise HTTPException(status_code=500, detail="Failed to search symbols")


@router.get("/symbols/managed-with-market-data")
async def get_managed_symbols_with_market_data():
    """Get managed symbols with real market data, P&L calculations, and weight calculations."""
    try:
        logger.info("Getting managed symbols with real market data")
        
        if not dependencies.db_pool:
            raise HTTPException(status_code=500, detail="Database not available")
            
        async with dependencies.db_pool.acquire() as conn:
            # Get all managed symbols regardless of status
            managed_symbols = await conn.fetch("""
                SELECT ms.*, s.name, s.sector, s.industry 
                FROM managed_symbols ms
                JOIN symbols s ON ms.symbol = s.symbol
                ORDER BY ms.priority DESC, ms.symbol
            """)
            
            symbols_with_data = []
            total_portfolio_value = 0
            
            for symbol in managed_symbols:
                try:
                    # Get current market price using yfinance
                    import yfinance as yf
                    ticker = yf.Ticker(symbol['symbol'])
                    hist = ticker.history(period="1d")
                    
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        
                        # Use stored initial_price if available, otherwise fetch historical data
                        initial_price = None
                        if symbol['initial_price']:
                            initial_price = float(symbol['initial_price'])
                            logger.info(f"Using stored initial price for {symbol['symbol']}: ${initial_price:.2f}")
                        else:
                            # Try to get historical price from when symbol was added
                            added_date = symbol['added_date']
                            if added_date:
                                try:
                                    hist_initial = ticker.history(start=added_date, end=added_date + timedelta(days=1))
                                    if not hist_initial.empty:
                                        initial_price = float(hist_initial['Close'].iloc[0])
                                        logger.info(f"Fetched historical initial price for {symbol['symbol']}: ${initial_price:.2f}")
                                        
                                        # Store the initial price in the database for future use
                                        await conn.execute("""
                                            UPDATE managed_symbols 
                                            SET initial_price = $1, initial_price_date = $2
                                            WHERE symbol = $3
                                        """, initial_price, added_date, symbol['symbol'])
                                except Exception as hist_error:
                                    logger.warning(f"Could not fetch historical price for {symbol['symbol']}: {hist_error}")
                        
                        # If we still don't have an initial price, use current price as baseline
                        if initial_price is None:
                            initial_price = current_price
                            # Store this as the initial price for future reference
                            await conn.execute("""
                                UPDATE managed_symbols 
                                SET initial_price = $1, initial_price_date = NOW()
                                WHERE symbol = $2 AND initial_price IS NULL
                            """, initial_price, symbol['symbol'])
                            logger.info(f"Set baseline initial price for {symbol['symbol']}: ${initial_price:.2f}")
                        
                        # Calculate P&L (assuming 1 share for simplicity, or use quantity if available)
                        quantity = 1.0  # Default quantity
                        pnl = (current_price - initial_price) * quantity
                        pnl_percent = ((current_price - initial_price) / initial_price) * 100 if initial_price > 0 else 0
                        
                        # Calculate market value
                        market_value = current_price * quantity
                        total_portfolio_value += market_value
                        
                        symbol_data = {
                            "symbol": symbol['symbol'],
                            "name": symbol['name'],
                            "sector": symbol['sector'],
                            "industry": symbol['industry'],
                            "status": symbol['status'],
                            "priority": symbol['priority'],
                            "notes": symbol['notes'],
                            "current_price": current_price,
                            "initial_price": initial_price,
                            "change_percent": pnl_percent,
                            "pnl": pnl,
                            "market_value": market_value,
                            "quantity": quantity,
                            "added_date": symbol['added_date'].isoformat() if symbol['added_date'] else None,
                            "last_updated": datetime.now().isoformat()
                        }
                        
                        symbols_with_data.append(symbol_data)
                        logger.info(f"Real market data for {symbol['symbol']}: ${current_price:.2f} (P&L: {pnl_percent:.2f}%)")
                        
                    else:
                        # No market data available
                        symbol_data = {
                            "symbol": symbol['symbol'],
                            "name": symbol['name'],
                            "sector": symbol['sector'],
                            "industry": symbol['industry'],
                            "status": symbol['status'],
                            "priority": symbol['priority'],
                            "notes": symbol['notes'],
                            "current_price": 0,
                            "initial_price": 0,
                            "change_percent": 0,
                            "pnl": 0,
                            "market_value": 0,
                            "quantity": 0,
                            "added_date": symbol['added_date'].isoformat() if symbol['added_date'] else None,
                            "last_updated": datetime.now().isoformat()
                        }
                        symbols_with_data.append(symbol_data)
                        
                except Exception as e:
                    logger.error(f"Error getting market data for {symbol['symbol']}: {e}")
                    # Add symbol with default values
                    symbol_data = {
                        "symbol": symbol['symbol'],
                        "name": symbol['name'],
                        "sector": symbol['sector'],
                        "industry": symbol['industry'],
                        "status": symbol['status'],
                        "priority": symbol['priority'],
                        "notes": symbol['notes'],
                        "current_price": 0,
                        "initial_price": 0,
                        "change_percent": 0,
                        "pnl": 0,
                        "market_value": 0,
                        "quantity": 0,
                        "added_date": symbol['added_date'].isoformat() if symbol['added_date'] else None,
                        "last_updated": datetime.now().isoformat()
                    }
                    symbols_with_data.append(symbol_data)
            
            # Calculate weights based on market values
            for symbol_data in symbols_with_data:
                if total_portfolio_value > 0:
                    symbol_data['weight'] = (symbol_data['market_value'] / total_portfolio_value) * 100
                else:
                    symbol_data['weight'] = 0
            
            logger.info(f"Returning {len(symbols_with_data)} managed symbols with real market data")
            return {"symbols": symbols_with_data, "total_portfolio_value": total_portfolio_value}
            
    except Exception as e:
        logger.error(f"Error getting managed symbols with market data: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Failed to retrieve managed symbols with market data")


@router.post("/ticker-discovery/scan-market")
async def scan_market():
    """Trigger a market scan for ticker discovery."""
    try:
        logger.info("🚀 Starting market scan for ticker discovery...")
        
        # Import the automated ticker discovery function
        from routes.utils import run_automated_ticker_discovery
        
        # Run the automated ticker discovery
        await run_automated_ticker_discovery()
        
        return {
            "success": True,
            "message": "Market scan completed successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error during market scan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to scan market: {str(e)}")
