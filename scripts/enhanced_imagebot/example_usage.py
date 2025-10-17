#!/usr/bin/env python3
"""
Example Usage Script for TradingView Multi-Agent Analyzer

This script demonstrates how to use the TradingView Multi-Agent Analyzer
to perform comprehensive stock analysis with multiple AI expert perspectives.

Before running:
1. Install dependencies: pip install -r requirements.txt
2. Set up your environment variables in a .env file:
   - OPENAI_API_KEY=your_openai_api_key
   - IMAGEKIT_PRIVATE_KEY=your_imagekit_private_key
   - IMAGEKIT_PUBLIC_KEY=your_imagekit_public_key
   - IMAGEKIT_URL_ENDPOINT=your_imagekit_url_endpoint
3. Ensure Chrome/Chromium is installed on your system
"""

import os
import sys
from datetime import datetime
from tradingview_multi_agent_analyzer import TradingViewMultiAgentAnalyzer
from dotenv import load_dotenv

load_dotenv()

def analyze_multiple_tickers():
    """Example: Analyze multiple tickers and save reports."""
    # Initialize the analyzer
    analyzer = TradingViewMultiAgentAnalyzer(screenshot_dir="./analysis_screenshots")
    
    # List of tickers to analyze
    tickers = ["SPY", "AAPL", "TSLA", "NVDA", "QQQ"]
    
    print("🚀 Starting Multi-Agent TradingView Analysis")
    print("=" * 60)
    
    for ticker in tickers:
        print(f"\n📊 Analyzing {ticker}...")
        
        # Perform analysis
        result = analyzer.analyze_ticker(ticker, cleanup_files=True)
        
        if result:
            # Generate markdown report
            report = analyzer.generate_markdown_report(result)
            
            # Save report to file
            filename = f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report)
            
            print(f"✅ Analysis complete for {ticker}")
            print(f"📄 Report saved: {filename}")
            print(f"🌐 Chart image: {result.image_url}")
            print(f"🎯 Consensus: {result.consensus_sentiment.upper()} - "
                  f"${result.consensus_target_range[0]:.2f} to ${result.consensus_target_range[1]:.2f}")
            
            # Print expert summary
            print("\n👨‍🔬 Expert Opinions:")
            for pred in result.expert_predictions:
                print(f"  • {pred.expert_name}: {pred.sentiment.upper()} "
                      f"(${pred.price_target_low:.2f}-${pred.price_target_high:.2f}, "
                      f"{pred.confidence}% confidence)")
        else:
            print(f"❌ Analysis failed for {ticker}")
        
        print("-" * 40)

def analyze_single_ticker_detailed():
    """Example: Detailed analysis of a single ticker with console output."""
    ticker = "SPY"
    
    print(f"🔍 Detailed Analysis for {ticker}")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = TradingViewMultiAgentAnalyzer()
    
    # Perform analysis
    result = analyzer.analyze_ticker(ticker)
    
    if not result:
        print(f"❌ Failed to analyze {ticker}")
        return
    
    # Display results
    print(f"\n📈 Analysis Results for {ticker}")
    print(f"🕐 Timestamp: {result.analysis_timestamp}")
    print(f"🖼️  Chart: {result.image_url}")
    print(f"🎯 Consensus: {result.consensus_sentiment.upper()}")
    print(f"💰 Price Range: ${result.consensus_target_range[0]:.2f} - ${result.consensus_target_range[1]:.2f}")
    
    print("\n👨‍🔬 Expert Analysis:")
    print("-" * 80)
    
    for i, pred in enumerate(result.expert_predictions, 1):
        print(f"{i}. {pred.expert_name}")
        print(f"   Sentiment: {pred.sentiment.upper()}")
        print(f"   Price Target: ${pred.price_target_low:.2f} - ${pred.price_target_high:.2f}")
        print(f"   Confidence: {pred.confidence}%")
        print(f"   Time Horizon: {pred.time_horizon}")
        print(f"   Reasoning: {pred.key_reasoning}")
        print()
    
    # Generate and display markdown report
    print("📋 Generating Markdown Report...")
    report = analyzer.generate_markdown_report(result)
    
    # Save report
    filename = f"{ticker}_detailed_analysis.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"✅ Detailed report saved: {filename}")
    
    # Display first few lines of the report
    print("\n📄 Report Preview:")
    print("-" * 40)
    lines = report.split('\n')[:15]
    for line in lines:
        print(line)
    print("... (see full report in file)")

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = [
        "OPENAI_API_KEY",
        "IMAGEKIT_PRIVATE_KEY", 
        "IMAGEKIT_PUBLIC_KEY",
        "IMAGEKIT_URL_ENDPOINT"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these in your .env file or environment.")
        return False
    
    print("✅ All required environment variables are set.")
    return True

def main():
    """Main function to run examples."""
    print("TradingView Multi-Agent Analyzer - Example Usage")
    print("=" * 60)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    print("\nChoose an analysis mode:")
    print("1. Single ticker detailed analysis (SPY)")
    print("2. Multiple tickers analysis (SPY, AAPL, TSLA, NVDA, QQQ)")
    print("3. Custom ticker analysis")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        analyze_single_ticker_detailed()
    elif choice == "2":
        analyze_multiple_tickers()
    elif choice == "3":
        ticker = input("Enter ticker symbol: ").strip().upper()
        if ticker:
            print(f"\n🔍 Analyzing {ticker}...")
            analyzer = TradingViewMultiAgentAnalyzer()
            result = analyzer.analyze_ticker(ticker)
            
            if result:
                report = analyzer.generate_markdown_report(result)
                filename = f"{ticker}_analysis.md"
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(report)
                print(f"✅ Analysis complete! Report saved: {filename}")
                print(f"🎯 Consensus: {result.consensus_sentiment.upper()}")
                print(f"💰 Price Range: ${result.consensus_target_range[0]:.2f} - ${result.consensus_target_range[1]:.2f}")
            else:
                print(f"❌ Analysis failed for {ticker}")
        else:
            print("❌ Invalid ticker symbol")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()