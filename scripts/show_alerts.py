#!/usr/bin/env python3
"""Quick script to show current high-confidence signals."""
import requests
import json

resp = requests.get("http://localhost:8003/symbols/trading-decisions")
decisions = resp.json()

print("🔔 HIGH-CONFIDENCE SIGNALS (>=70%):")
print("="*50)
count = 0
for d in sorted(decisions, key=lambda x: x['confidence'], reverse=True):
    if d['confidence'] >= 0.70 and d['action'] != 'hold':
        emoji = "🟢 BUY " if d['action'] == 'buy' else "🔴 SELL"
        print(f"{emoji} | {d['symbol']:8} | {d['confidence']:.0%} | {d['agent']}")
        count += 1

if count == 0:
    print("No high-confidence signals at this time.")
print("="*50)
