#!/usr/bin/env python3
"""
AI Market Alert Monitor
Checks for high-confidence trading signals and sends alerts.
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import List, Dict

# Configuration
API_URL = "http://localhost:8003"
CONFIDENCE_THRESHOLD = 0.70  # Alert on signals >= 70% confidence
CHECK_INTERVAL = 300  # Check every 5 minutes (300 seconds)
ALERT_LOG_FILE = "/home/mohands/ai-market/logs/alerts.log"

class AlertMonitor:
    def __init__(self):
        self.last_alerts = {}  # Track sent alerts to avoid duplicates

    async def fetch_trading_decisions(self) -> List[Dict]:
        """Fetch current trading decisions from API."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{API_URL}/symbols/trading-decisions") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            print(f"Error fetching decisions: {e}")
        return []

    async def fetch_forecasts(self) -> Dict:
        """Fetch forecast summary."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{API_URL}/forecasting/day-forecast/summary") as resp:
                    if resp.status == 200:
                        return await resp.json()
        except Exception as e:
            print(f"Error fetching forecasts: {e}")
        return {}

    def log_alert(self, alert: Dict):
        """Log alert to file."""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {alert['type']}: {alert['symbol']} - {alert['action']} ({alert['confidence']:.0%}) - {alert['message']}\n"

        print(log_entry.strip())

        try:
            with open(ALERT_LOG_FILE, "a") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Error writing log: {e}")

    def check_high_confidence_signals(self, decisions: List[Dict]) -> List[Dict]:
        """Check for high-confidence trading signals."""
        alerts = []

        for decision in decisions:
            symbol = decision.get('symbol')
            confidence = decision.get('confidence', 0)
            action = decision.get('action', 'hold')

            # Skip low confidence or hold signals
            if confidence < CONFIDENCE_THRESHOLD or action == 'hold':
                continue

            # Create alert key to avoid duplicates
            alert_key = f"{symbol}_{action}_{confidence:.2f}"

            if alert_key not in self.last_alerts:
                alert = {
                    'type': 'HIGH_CONFIDENCE_SIGNAL',
                    'symbol': symbol,
                    'action': action.upper(),
                    'confidence': confidence,
                    'agent': decision.get('agent', 'Unknown'),
                    'message': f"Strong {action.upper()} signal detected!"
                }
                alerts.append(alert)
                self.last_alerts[alert_key] = datetime.now()

        return alerts

    def check_market_regime_change(self, forecasts: Dict) -> List[Dict]:
        """Check for significant market changes."""
        alerts = []

        buy_signals = forecasts.get('buy_signals', 0)
        sell_signals = forecasts.get('sell_signals', 0)
        total = buy_signals + sell_signals

        if total > 0:
            buy_ratio = buy_signals / total

            if buy_ratio > 0.8:
                alerts.append({
                    'type': 'MARKET_REGIME',
                    'symbol': 'MARKET',
                    'action': 'BULLISH',
                    'confidence': buy_ratio,
                    'message': f"Strong bullish market: {buy_signals} buy vs {sell_signals} sell signals"
                })
            elif buy_ratio < 0.2:
                alerts.append({
                    'type': 'MARKET_REGIME',
                    'symbol': 'MARKET',
                    'action': 'BEARISH',
                    'confidence': 1 - buy_ratio,
                    'message': f"Strong bearish market: {sell_signals} sell vs {buy_signals} buy signals"
                })

        return alerts

    async def run_check(self):
        """Run a single check cycle."""
        print(f"\n{'='*50}")
        print(f"Alert Check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}")

        # Fetch data
        decisions = await self.fetch_trading_decisions()
        forecasts = await self.fetch_forecasts()

        # Check for alerts
        all_alerts = []
        all_alerts.extend(self.check_high_confidence_signals(decisions))
        all_alerts.extend(self.check_market_regime_change(forecasts))

        # Log alerts
        if all_alerts:
            print(f"\n🚨 {len(all_alerts)} ALERT(S) DETECTED:")
            for alert in all_alerts:
                self.log_alert(alert)
        else:
            print("✅ No new alerts")

        # Show current high-confidence signals
        high_conf = [d for d in decisions if d.get('confidence', 0) >= CONFIDENCE_THRESHOLD and d.get('action') != 'hold']
        if high_conf:
            print(f"\n📊 Current High-Confidence Signals (>={CONFIDENCE_THRESHOLD:.0%}):")
            for signal in sorted(high_conf, key=lambda x: x['confidence'], reverse=True):
                emoji = "🟢" if signal['action'] == 'buy' else "🔴"
                print(f"   {emoji} {signal['symbol']}: {signal['action'].upper()} ({signal['confidence']:.0%})")

    async def run_continuous(self):
        """Run continuous monitoring."""
        print(f"""
╔══════════════════════════════════════════════════════╗
║     AI MARKET ALERT MONITOR                          ║
║     Confidence Threshold: {CONFIDENCE_THRESHOLD:.0%}                       ║
║     Check Interval: {CHECK_INTERVAL} seconds                     ║
║     Log File: {ALERT_LOG_FILE}      ║
╚══════════════════════════════════════════════════════╝
        """)

        while True:
            try:
                await self.run_check()
                print(f"\nNext check in {CHECK_INTERVAL} seconds...")
                await asyncio.sleep(CHECK_INTERVAL)
            except KeyboardInterrupt:
                print("\n\nMonitor stopped by user.")
                break
            except Exception as e:
                print(f"Error in monitor loop: {e}")
                await asyncio.sleep(60)

async def main():
    monitor = AlertMonitor()
    await monitor.run_continuous()

if __name__ == "__main__":
    asyncio.run(main())
