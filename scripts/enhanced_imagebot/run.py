from tradingview_multi_agent_analyzer import TradingViewMultiAgentAnalyzer

# Custom screenshot directory
analyzer = TradingViewMultiAgentAnalyzer(screenshot_dir="./my_charts")
tickr = input("Enter a ticker: ")
# Analyze with custom cleanup settings
result = analyzer.analyze_ticker(tickr, cleanup_files=False)

if result:
    # Generate markdown report
    report = analyzer.generate_markdown_report(result)
    

    
    print(f"Consensus: {result.consensus_sentiment}")
    print(f"Price Range: ${result.consensus_target_range[0]:.2f} - ${result.consensus_target_range[1]:.2f}")
# Access individual expert predictions
for prediction in result.expert_predictions:
    print(f"{prediction.expert_name}: {prediction.sentiment}")
    print(f"Target: ${prediction.price_target_low}-${prediction.price_target_high}")
    print(f"Confidence: {prediction.confidence}%")
    print(f"Reasoning: {prediction.key_reasoning}")

# Save report
with open(f"{tickr}_analysis.md", "w", encoding="utf-8") as f:
    f.write(report)