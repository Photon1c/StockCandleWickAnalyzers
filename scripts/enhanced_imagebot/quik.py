from tradingview_multi_agent_analyzer import TradingViewMultiAgentAnalyzer

# Initialize analyzer
analyzer = TradingViewMultiAgentAnalyzer()

tickr = input("Enter a ticker: ")

# Analyze a stock
result = analyzer.analyze_ticker(tickr)

if result:
    # Access individual expert predictions
    for prediction in result.expert_predictions:
        print(f"{prediction.expert_name}: {prediction.sentiment}")
        print(f"Low/High Targets: ${prediction.price_target_low}-${prediction.price_target_high}")
        print(f"Confidence: {prediction.confidence}%")
        print(f"Reasoning: {prediction.key_reasoning}")

    # Generate markdown report
    report = analyzer.generate_markdown_report(result)
    
    # Save report
    with open(f"{tickr}_analysis.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Consensus: {result.consensus_sentiment}")
    print(f"Consensus Price Range: ${result.consensus_target_range[0]:.2f} - ${result.consensus_target_range[1]:.2f}")


