# TradingView Multi-Agent Analyzer 🤖📈

An advanced stock analysis tool that combines web scraping, computer vision, and multi-agent AI to provide comprehensive market insights. This system captures TradingView charts and analyzes them using multiple AI expert personas, generating detailed markdown reports with price predictions and investment recommendations.

## 🌟 Features

- **Automated Chart Capture**: Selenium-powered TradingView chart screenshots
- **Multi-Agent AI Analysis**: Four specialized AI experts analyze charts from different perspectives:
  - 📊 **Technical Analyst** (Dr. Sarah Chen) - Chart patterns & indicators
  - 📈 **Fundamental Analyst** (Dr. Michael Rodriguez) - Economic factors
  - ⚠️ **Risk Analyst** (Dr. Jennifer Kim) - Volatility & risk assessment  
  - 🧠 **Behavioral Analyst** (Dr. Robert Thompson) - Market psychology
- **Consensus Generation**: Weighted consensus predictions from all experts
- **Markdown Reports**: Professional GitHub-style reports with tables
- **Image Hosting**: Automatic ImageKit integration for chart hosting
- **Comprehensive Logging**: Full audit trail and error handling

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Chrome/Chromium browser installed
- OpenAI API key
- ImageKit account (for image hosting)

### Installation

1. **Clone/Download the files**:
   ```bash
   # Download the main files:
   # - tradingview_multi_agent_analyzer.py
   # - example_usage.py  
   # - requirements.txt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with your API credentials:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   IMAGEKIT_PRIVATE_KEY=your_imagekit_private_key
   IMAGEKIT_PUBLIC_KEY=your_imagekit_public_key  
   IMAGEKIT_URL_ENDPOINT=https://ik.imagekit.io/your_endpoint
   ```

4. **Run the example**:
   ```bash
   python example_usage.py
   ```

## 📊 Usage Examples

### Basic Usage

```python
from tradingview_multi_agent_analyzer import TradingViewMultiAgentAnalyzer

# Initialize analyzer
analyzer = TradingViewMultiAgentAnalyzer()

# Analyze a stock
result = analyzer.analyze_ticker("AAPL")

if result:
    # Generate markdown report
    report = analyzer.generate_markdown_report(result)
    
    # Save report
    with open("AAPL_analysis.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"Consensus: {result.consensus_sentiment}")
    print(f"Price Range: ${result.consensus_target_range[0]:.2f} - ${result.consensus_target_range[1]:.2f}")
```

### Advanced Usage

```python
# Custom screenshot directory
analyzer = TradingViewMultiAgentAnalyzer(screenshot_dir="./my_charts")

# Analyze with custom cleanup settings
result = analyzer.analyze_ticker("TSLA", cleanup_files=False)

# Access individual expert predictions
for prediction in result.expert_predictions:
    print(f"{prediction.expert_name}: {prediction.sentiment}")
    print(f"Target: ${prediction.price_target_low}-${prediction.price_target_high}")
    print(f"Confidence: {prediction.confidence}%")
    print(f"Reasoning: {prediction.key_reasoning}")
```

## 🏗️ Architecture

### Core Components

1. **TradingViewCaptureService**: Handles browser automation and chart capture
2. **ImageUploadService**: Manages ImageKit uploads and URL generation  
3. **MultiAgentAnalysisEngine**: Orchestrates AI expert analysis
4. **TradingViewMultiAgentAnalyzer**: Main coordinator class

### Expert Personas

Each AI expert has a specialized background and analytical focus:

| Expert | Specialty | Focus Areas |
|--------|-----------|-------------|
| Dr. Sarah Chen | Technical Analysis | Chart patterns, indicators, support/resistance |
| Dr. Michael Rodriguez | Fundamental Analysis | Economic data, earnings, macro factors |
| Dr. Jennifer Kim | Risk Management | Volatility analysis, downside protection |
| Dr. Robert Thompson | Behavioral Finance | Market sentiment, psychology |

### Data Flow

```
Ticker Input → Chart Capture → Image Upload → Multi-Agent Analysis → Consensus → Markdown Report
```

## 📋 Output Format

The analyzer generates comprehensive markdown reports including:

- **Expert Predictions Table**: Individual expert analysis with price targets
- **Consensus Summary**: Weighted consensus sentiment and price range
- **Investment Recommendations**: Time-horizon based strategies
- **Visual Elements**: Emoji indicators and professional formatting

### Sample Output

```markdown
# Multi-Agent Analysis Report: AAPL

## Expert Predictions

| Expert | Specialty | Sentiment | Price Target Range | Confidence | Time Horizon | Key Reasoning |
|--------|-----------|-----------|-------------------|------------|--------------|---------------|
| Dr. Sarah Chen | Technical Analysis | 🟢 Bullish | $180.00 - $195.00 | 85% | 3-6 months | Strong momentum indicators and breakout pattern |
| Dr. Michael Rodriguez | Fundamental Analysis | 🟢 Bullish | $175.00 - $190.00 | 78% | 6-12 months | Solid earnings growth and market position |
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT-4 Vision | ✅ |
| `IMAGEKIT_PRIVATE_KEY` | ImageKit private key | ✅ |
| `IMAGEKIT_PUBLIC_KEY` | ImageKit public key | ✅ |
| `IMAGEKIT_URL_ENDPOINT` | ImageKit URL endpoint | ✅ |

### Optional Settings

- **Screenshot Directory**: Customize where charts are temporarily stored
- **Cleanup Files**: Control whether to delete local files after analysis
- **Timeframe**: Chart timeframe (default: 1D)

## 🔧 Troubleshooting

### Common Issues

1. **Chrome Driver Issues**:
   ```bash
   # Update Chrome driver
   pip install --upgrade webdriver-manager
   ```

2. **Environment Variables**:
   ```bash
   # Check if .env file is loaded
   python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```

3. **ImageKit Upload Failures**:
   - Verify API credentials
   - Check network connectivity
   - Ensure image file exists

4. **UnicodeEncodeError on Windows when saving reports**:
   - Symptom: `UnicodeEncodeError: 'charmap' codec can't encode character ...`
   - Cause: Windows default code page (cp1252) cannot encode emoji used in the report
   - Fix: Open files with UTF-8 encoding when writing reports:
     ```python
     with open("AAPL_analysis.md", "w", encoding="utf-8") as f:
         f.write(report)
     ```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🛡️ Rate Limiting & Costs

- **OpenAI API**: ~$0.01-0.05 per analysis (GPT-4 Vision)
- **TradingView**: No API calls, web scraping only
- **ImageKit**: Free tier includes 20GB bandwidth/month

### Best Practices

- Use rate limiting between analyses (built-in 1-second delays)
- Monitor API usage and costs
- Consider batching multiple analyses
- Cache results when possible

## 🤝 Contributing

This is an AI-generated tool. To improve it:

1. **Report Issues**: Document any bugs or limitations
2. **Suggest Features**: Propose new expert personas or analysis types
3. **Optimize Performance**: Suggest improvements to speed or accuracy
4. **Enhance Documentation**: Add more usage examples

## ⚠️ Disclaimers

- **Not Financial Advice**: This tool is for educational and research purposes only
- **AI-Generated Analysis**: Predictions are based on chart patterns, not comprehensive market analysis
- **No Guarantees**: Past performance and technical analysis do not guarantee future results
- **Use Responsibly**: Always consult qualified financial professionals before making investment decisions

## 📄 License

MIT License - Feel free to use, modify, and distribute as needed.

## 🔗 Dependencies

- `selenium` - Web browser automation
- `openai` - AI analysis via GPT-4 Vision
- `imagekitio` - Image hosting and management
- `Pillow` - Image processing
- `webdriver-manager` - Chrome driver management
- `python-dotenv` - Environment variable management

---

*Generated by AI Assistant - Optimized for professional trading analysis workflows*
