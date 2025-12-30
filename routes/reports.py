"""
Reports Routes - Report generation and AI explanations
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from loguru import logger
from typing import Optional, Dict, Any
from dataclasses import asdict

from routes import dependencies
from agents.report_agent import ReportAgent, ReportType, ReportFormat
from agents.llm_explain_agent import LLMExplainAgent, ExplanationType, ExplanationTone
from services.report_generation_service import ReportGenerationService
from services.real_data_service import RealDataService, RealDataConfig

router = APIRouter()

# Initialize services
_report_agent: Optional[ReportAgent] = None
_report_generation_service: Optional[ReportGenerationService] = None
_llm_explain_agent: Optional[LLMExplainAgent] = None


def get_report_agent() -> ReportAgent:
    global _report_agent
    if _report_agent is None:
        real_data_service = RealDataService(RealDataConfig())
        _report_agent = ReportAgent(real_data_service)
    return _report_agent


def get_report_generation_service() -> ReportGenerationService:
    global _report_generation_service
    if _report_generation_service is None:
        _report_generation_service = ReportGenerationService()
    return _report_generation_service


def get_llm_explain_agent() -> LLMExplainAgent:
    global _llm_explain_agent
    if _llm_explain_agent is None:
        _llm_explain_agent = LLMExplainAgent()
    return _llm_explain_agent


@router.get("/reports/summary")
async def get_reports_summary():
    """Get reporting system summary."""
    try:
        report_agent = get_report_agent()
        report_service = get_report_generation_service()

        agent_summary = await report_agent.get_report_summary()
        service_summary = await report_service.get_report_summary()

        return {
            "report_agent": agent_summary,
            "report_service": service_summary,
            "status": "operational",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting reports summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/daily")
async def generate_daily_report(date: Optional[str] = None, format: str = "json"):
    """Generate daily trading report."""
    try:
        report_agent = get_report_agent()
        report_service = get_report_generation_service()

        # Generate mock data if needed
        if len(report_agent.trade_history) == 0:
            await report_agent.generate_mock_data()

        # Generate the report summary
        report_summary = await report_agent.generate_daily_report(date)

        # Convert to appropriate format
        report_format = ReportFormat.JSON if format.lower() == "json" else ReportFormat.HTML
        generated_report = await report_service.generate_report(report_summary, report_format)

        return {
            "report_id": generated_report.report_id,
            "report_type": "daily",
            "format": generated_report.format.value,
            "generated_at": generated_report.generated_at,
            "file_size": generated_report.file_size,
            "file_path": generated_report.file_path,
            "summary": {
                "total_trades": report_summary.total_trades,
                "successful_trades": report_summary.successful_trades,
                "failed_trades": report_summary.failed_trades,
                "win_rate": report_summary.win_rate,
                "total_pnl": report_summary.total_pnl,
                "best_performing_agent": report_summary.best_performing_agent
            }
        }
    except Exception as e:
        logger.error(f"Error generating daily report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/weekly")
async def generate_weekly_report(week_start: Optional[str] = None, format: str = "json"):
    """Generate weekly trading report."""
    try:
        report_agent = get_report_agent()
        report_service = get_report_generation_service()

        # Generate mock data if needed
        if len(report_agent.trade_history) == 0:
            await report_agent.generate_mock_data()

        # Generate the report summary
        report_summary = await report_agent.generate_weekly_report(week_start)

        # Convert to appropriate format
        report_format = ReportFormat.JSON if format.lower() == "json" else ReportFormat.MARKDOWN
        generated_report = await report_service.generate_report(report_summary, report_format)

        return {
            "report_id": generated_report.report_id,
            "report_type": "weekly",
            "format": generated_report.format.value,
            "generated_at": generated_report.generated_at,
            "file_size": generated_report.file_size,
            "file_path": generated_report.file_path,
            "summary": {
                "total_trades": report_summary.total_trades,
                "successful_trades": report_summary.successful_trades,
                "failed_trades": report_summary.failed_trades,
                "win_rate": report_summary.win_rate,
                "total_pnl": report_summary.total_pnl,
                "best_performing_agent": report_summary.best_performing_agent
            }
        }
    except Exception as e:
        logger.error(f"Error generating weekly report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/reports/agent-performance")
async def generate_agent_performance_report(format: str = "json"):
    """Generate agent performance report."""
    try:
        report_agent = get_report_agent()
        report_service = get_report_generation_service()

        # Generate mock data if needed
        if len(report_agent.trade_history) == 0:
            await report_agent.generate_mock_data()

        # Generate the report summary
        report_summary = await report_agent.generate_agent_performance_report()

        # Convert to appropriate format
        report_format = ReportFormat.JSON if format.lower() == "json" else ReportFormat.HTML
        generated_report = await report_service.generate_report(report_summary, report_format)

        return {
            "report_id": generated_report.report_id,
            "report_type": "agent-performance",
            "format": generated_report.format.value,
            "generated_at": generated_report.generated_at,
            "file_size": generated_report.file_size,
            "file_path": generated_report.file_path,
            "summary": {
                "total_trades": report_summary.total_trades,
                "successful_trades": report_summary.successful_trades,
                "failed_trades": report_summary.failed_trades,
                "win_rate": report_summary.win_rate,
                "total_pnl": report_summary.total_pnl,
                "best_performing_agent": report_summary.best_performing_agent,
                "worst_performing_agent": report_summary.worst_performing_agent
            }
        }
    except Exception as e:
        logger.error(f"Error generating agent performance report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reports/explain")
async def generate_explanation(request: Dict[str, Any]):
    """Generate AI-powered explanation for trading decisions, agent performance, or market analysis."""
    try:
        llm_agent = get_llm_explain_agent()

        explanation_type = request.get("explanation_type", "trade_decision")
        data = request.get("data", {})
        tone_str = request.get("tone", "professional")

        # Map tone string to enum
        tone_map = {
            "professional": ExplanationTone.PROFESSIONAL,
            "conversational": ExplanationTone.CONVERSATIONAL,
            "technical": ExplanationTone.TECHNICAL,
            "simplified": ExplanationTone.SIMPLIFIED
        }
        tone = tone_map.get(tone_str.lower(), ExplanationTone.PROFESSIONAL)

        # Generate explanation based on type
        if explanation_type == "trade_decision":
            explanation = await llm_agent.explain_trade_decision(data, tone)
        elif explanation_type == "agent_performance":
            explanation = await llm_agent.explain_agent_performance(data, tone)
        elif explanation_type == "market_analysis":
            explanation = await llm_agent.explain_market_analysis(data, tone)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown explanation type: {explanation_type}")

        return {
            "explanation_type": explanation_type,
            "title": explanation.title,
            "summary": explanation.summary,
            "detailed_explanation": explanation.detailed_explanation,
            "key_factors": explanation.key_factors,
            "recommendations": explanation.recommendations,
            "confidence_level": explanation.confidence_level,
            "generated_at": explanation.generated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
