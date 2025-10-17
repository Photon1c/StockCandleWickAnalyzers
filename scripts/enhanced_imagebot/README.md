# 📊 Enhanced ImageBot — AI-Powered Chart Analysis Suite

A comprehensive toolkit for automated TradingView chart capture, AI-powered multi-agent analysis, and intelligent stock market insights using GPT-4 Vision.

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green?style=flat-square)
![Selenium](https://img.shields.io/badge/Selenium-WebDriver-orange?style=flat-square)

---

## 🚀 Quick Start Guide

### Prerequisites

1. **Install Dependencies:**
```bash
pip install selenium webdriver-manager imagekitio openai python-dotenv pillow httpx
```

2. **Set Up Environment Variables:**
Create a `.env` file in this directory:
```env
OPENAI_API_KEY=your_openai_api_key
IMAGEKIT_PRIVATE_KEY=your_imagekit_private_key
IMAGEKIT_PUBLIC_KEY=your_imagekit_public_key
IMAGEKIT_URL_ENDPOINT=your_imagekit_url_endpoint
```

3. **Ensure Chrome/Chromium is Installed** (for Selenium WebDriver)

---

## 📁 File Navigator

### 🎯 Quick Analysis Scripts (Start Here!)

#### `quik.py` — Fastest Multi-Agent Analysis
**Purpose:** Simple interactive script for quick ticker analysis  
**Best For:** Quick market checks, beginner-friendly  
**Usage:**
```bash
python quik.py
# Enter ticker when prompted (e.g., SPY)
```
**Output:** `{TICKER}_analysis.md` with multi-expert consensus

**What it does:**
- ✅ Prompts for ticker input
- ✅ Captures TradingView chart
- ✅ Multi-agent AI analysis (4 experts)
- ✅ Generates markdown report with consensus
- ✅ Minimal configuration needed

---

#### `run.py` — Customizable Quick Runner
**Purpose:** Similar to `quik.py` but with custom settings  
**Best For:** Users who want control over screenshot directory  
**Usage:**
```bash
python run.py
# Enter ticker when prompted
```
**Key Features:**
- Custom screenshot directory: `./my_charts`
- Option to keep screenshot files (`cleanup_files=False`)
- Shows consensus + individual expert opinions

---

### 🏗️ Core Engines (The Workhorses)

#### `tradingview_multi_agent_analyzer.py` — Full-Featured Analysis Engine
**Purpose:** Complete multi-agent analysis system with debate mechanism  
**Best For:** Production use, comprehensive analysis  
**Class:** `TradingViewMultiAgentAnalyzer`

**Key Features:**
- 🤖 **5 Expert AI Agents:**
  - Technical Analyst (price action, momentum)
  - Fundamental Analyst (value, earnings)
  - Risk Manager (drawdowns, volatility)
  - Options Strategist (gamma, IV)
  - Market Sentiment Analyst (flows, positioning)
- 📊 Captures TradingView charts via Selenium
- 🌐 Uploads to ImageKit CDN
- 🧠 GPT-4 Vision analysis
- 📈 Consensus generation via weighted voting
- 📝 Beautiful markdown reports

**Usage:**
```python
from tradingview_multi_agent_analyzer import TradingViewMultiAgentAnalyzer

analyzer = TradingViewMultiAgentAnalyzer()
result = analyzer.analyze_ticker("SPY")

if result:
    print(f"Consensus: {result.consensus_sentiment}")
    print(f"Price Range: ${result.consensus_target_range[0]:.2f} - ${result.consensus_target_range[1]:.2f}")
    
    # Generate report
    report = analyzer.generate_markdown_report(result)
    with open("SPY_analysis.md", "w") as f:
        f.write(report)
```

**Output Structure:**
- `AnalysisResult` dataclass with:
  - `expert_predictions`: List of `ExpertPrediction` objects
  - `consensus_sentiment`: "bullish", "bearish", or "neutral"
  - `consensus_target_range`: (low, high) tuple
  - `image_url`: Chart image URL
  - `analysis_timestamp`: When analysis was performed

---

#### `optimized_chart_analyzer.py` — Performance-Optimized Version
**Purpose:** Streamlined, high-performance chart analysis  
**Best For:** Batch processing, API integration  
**Class:** `ChartAnalyzer`

**Key Features:**
- ⚡ Optimized for speed and reliability
- 🔄 Fallback mechanisms (ImageKit → base64 inline)
- 🎯 Single-agent or multi-agent modes
- 📊 Structured prediction dataclass
- 🛡️ Comprehensive error handling

**Command Line Usage:**
```bash
# Multi-agent debate (default)
python optimized_chart_analyzer.py SPY

# Single-agent quick analysis
python optimized_chart_analyzer.py SPY --single

# Use existing image (skip capture)
python optimized_chart_analyzer.py --use-png /path/to/chart.png
```

**Python API:**
```python
from optimized_chart_analyzer import ChartAnalyzer, Prediction

analyzer = ChartAnalyzer(screenshot_dir="./screenshots")

# Multi-agent analysis
predictions = analyzer.analyze_chart_multi_agent("SPY")
for pred in predictions:
    print(f"{pred.expert_name}: {pred.sentiment} ({pred.confidence}%)")

# Single-agent analysis
prediction = analyzer.analyze_chart_single_agent("AAPL")
print(prediction.rationale)
```

**Differences from `tradingview_multi_agent_analyzer.py`:**
- Lighter weight, fewer dependencies
- Can work without ImageKit (uses base64)
- Command-line friendly
- More modular design

---

### 📅 Weekly Outlook Generators

#### `weekly_outlook_agents.py` — Weekly Report Generator (v1)
**Purpose:** Generate concise weekly outlook reports with 4 AI experts  
**Best For:** Weekly market summaries, team briefings  

**Experts:**
- 📊 Macro Strategist (rates, inflation, liquidity)
- 📈 Technical Specialist (price action, momentum)
- 🔢 Quant Analyst (volatility, probabilities)
- 💹 Flow Trader (options, gamma, positioning)

**Usage:**
```bash
python weekly_outlook_agents.py --ticker SPY --model gpt-4o
python weekly_outlook_agents.py --ticker QQQ --output-dir ./reports
```

**CLI Arguments:**
- `--ticker TICKER`: Stock symbol (default: SPY)
- `--model MODEL`: OpenAI model (default: gpt-4o)
- `--output-dir DIR`: Report output directory
- `--screenshot-dir DIR`: Screenshot directory
- `--no-capture`: Skip chart capture (text-only)
- `--no-headless`: Show browser window

**Output:**
- Markdown report with:
  - Consensus emoji (📈 Bullish, 📉 Bearish, ↔️ Neutral)
  - Sentiment breakdown table
  - Average confidence score
  - Dominant keywords
  - Individual expert outlooks (1-2 sentences each)

---

#### `weekly_outlook_agents_v2.py` — Gamma-Aware Weekly Reports
**Purpose:** Enhanced weekly outlook with gamma regime context  
**Best For:** Options traders, gamma-sensitive strategies  

**What's Different:**
- 🌊 **Gamma-hill mental model**: Agents use bowl/ridge/plateau framework
- 🎯 **More opinionated**: Reduces neutral predictions, pushes toward decisive calls
- 📊 **Gamma context**: Includes `--gamma-state` parameter
- 💪 **Stronger convictions**: Prompts encourage bold takes when appropriate

**Usage:**
```bash
# With gamma context
python weekly_outlook_agents_v2.py --ticker SPY --gamma-state positive

# Negative gamma regime
python weekly_outlook_agents_v2.py --ticker SPY --gamma-state negative --model gpt-4o
```

**Gamma States:**
- `positive`: Market in positive gamma (stabilizing, dealers buy dips)
- `negative`: Market in negative gamma (amplifying, dealers sell into weakness)
- `neutral`: Balanced regime
- `unknown`: No gamma info (default)

**When to Use v2 vs v1:**
- **Use v1**: General market analysis, no options context
- **Use v2**: Options trading, gamma-sensitive strategies, want decisive calls

---

### 🦅 Specialized Tools

#### `eaglevision.py` — Legacy TinyTroupe Agent System
**Purpose:** Early experimental multi-agent system using TinyTroupe framework  
**Best For:** Understanding agent-based reasoning, research  
**Status:** Legacy/experimental

**Features:**
- Uses TinyTroupe's `TinyPerson` agent framework
- Two personas: Leslie (experienced) and Ashley (energetic)
- Demonstrates agent communication and world-building
- More complex, less production-ready

**Note:** This was an early prototype. For production, use `tradingview_multi_agent_analyzer.py` or `optimized_chart_analyzer.py`.

---

#### `enhanced_imagebot.py` — Core Utilities Module
**Purpose:** Shared utilities and base classes  
**Best For:** Library usage, building custom tools  

**Contains:**
- `ChartAnalyzer` base class
- `Prediction` dataclass
- Screenshot capture utilities
- ImageKit upload helpers
- OpenAI vision integration

**This is imported by:** `quik.py`, `run.py`, other scripts

---

### 📖 Documentation & Examples

#### `example_usage.py` — Comprehensive Usage Examples
**Purpose:** Demonstrates all major features and use cases  
**Best For:** Learning the API, copy-paste examples  

**Examples Included:**
1. **Batch Analysis:** Multiple tickers at once
2. **Single Ticker Detailed:** Deep dive with full output
3. **Custom Configuration:** Custom dirs, models, options
4. **Production Workflow:** Robust error handling

**Usage:**
```bash
python example_usage.py
```

**Functions demonstrated:**
- `analyze_multiple_tickers()`: Batch processing
- `analyze_single_ticker_detailed()`: Verbose single analysis
- `custom_configuration_example()`: Custom settings
- `production_workflow()`: Real-world patterns

---

## 🎯 Which File Should I Use?

### For Quick Analysis
```
┌─────────────────────────────────────┐
│ Need: Fast ticker check             │
│ Use:  quik.py or run.py             │
│ Time: ~30 seconds per ticker        │
└─────────────────────────────────────┘
```

### For Production/Integration
```
┌─────────────────────────────────────┐
│ Need: API/script integration        │
│ Use:  tradingview_multi_agent_      │
│       analyzer.py                   │
│ Why:  Most feature-complete         │
└─────────────────────────────────────┘
```

### For CLI/Batch Processing
```
┌─────────────────────────────────────┐
│ Need: Command-line tool              │
│ Use:  optimized_chart_analyzer.py   │
│ Why:  Argparse-based, scriptable    │
└─────────────────────────────────────┘
```

### For Weekly Reports
```
┌─────────────────────────────────────┐
│ Need: Weekly market summary          │
│ Use:  weekly_outlook_agents.py (v1) │
│       weekly_outlook_agents_v2.py   │
│ Why:  Concise 1-2 sentence format   │
└─────────────────────────────────────┘
```

### For Options Traders
```
┌─────────────────────────────────────┐
│ Need: Gamma-aware analysis           │
│ Use:  weekly_outlook_agents_v2.py   │
│ Why:  Gamma regime context built-in │
└─────────────────────────────────────┘
```

---

## 📊 Comparison Table

| File | Agents | Gamma-Aware | CLI | API | Output | Speed |
|------|--------|-------------|-----|-----|--------|-------|
| `quik.py` | 4 | ❌ | ❌ | ❌ | Markdown | Fast |
| `run.py` | 4 | ❌ | ❌ | ❌ | Markdown | Fast |
| `tradingview_multi_agent_analyzer.py` | 5 | ✅ | ❌ | ✅ | Dataclass + MD | Medium |
| `optimized_chart_analyzer.py` | 1-5 | ❌ | ✅ | ✅ | Dataclass + MD | Fast |
| `weekly_outlook_agents.py` | 4 | ❌ | ✅ | ✅ | Markdown | Fast |
| `weekly_outlook_agents_v2.py` | 4 | ✅ | ✅ | ✅ | Markdown | Fast |
| `eaglevision.py` | 2 | ❌ | ❌ | ⚠️ | Text | Slow |
| `enhanced_imagebot.py` | - | ❌ | ❌ | ✅ | Library | - |

---

## 🔧 Common Workflows

### Workflow 1: Daily Morning Check
```bash
# Quick check on your watchlist
python quik.py  # Enter SPY
python quik.py  # Enter AAPL
python quik.py  # Enter TSLA
```

### Workflow 2: Weekly Team Report
```bash
# Generate gamma-aware weekly outlook
python weekly_outlook_agents_v2.py --ticker SPY --gamma-state positive
python weekly_outlook_agents_v2.py --ticker QQQ --gamma-state negative
```

### Workflow 3: Batch Analysis
```bash
# Use example_usage.py for multiple tickers
python example_usage.py
# Edit the tickers list in the file first
```

### Workflow 4: Programmatic Integration
```python
from tradingview_multi_agent_analyzer import TradingViewMultiAgentAnalyzer

analyzer = TradingViewMultiAgentAnalyzer()
watchlist = ["SPY", "QQQ", "AAPL", "TSLA", "NVDA"]

for ticker in watchlist:
    result = analyzer.analyze_ticker(ticker)
    if result:
        print(f"{ticker}: {result.consensus_sentiment.upper()}")
        # Save to database, send alerts, etc.
```

---

## 🤖 Understanding the AI Agents

### Core Expert Profiles

#### 1. **Technical Analyst** 📈
- Focus: Price action, support/resistance, momentum indicators
- Sentiment: Based on chart patterns and technical levels
- Strengths: Short-term moves, trend identification

#### 2. **Fundamental Analyst** 💼
- Focus: Company value, earnings, sector health
- Sentiment: Based on intrinsic value vs market price
- Strengths: Long-term outlook, value assessment

#### 3. **Risk Manager** 🛡️
- Focus: Volatility, drawdowns, risk-adjusted returns
- Sentiment: Defensive, protective positioning
- Strengths: Downside protection, risk assessment

#### 4. **Options Strategist** 🎯
- Focus: Implied volatility, gamma exposure, options flows
- Sentiment: Based on options market signals
- Strengths: Volatility plays, gamma regimes

#### 5. **Market Sentiment Analyst** 📊
- Focus: Market psychology, positioning, flows
- Sentiment: Contrarian or momentum-following
- Strengths: Market timing, sentiment extremes

### Consensus Mechanism

The system uses **weighted voting** based on confidence levels:
1. Each expert provides: sentiment + confidence (0-100%)
2. Weighted average determines consensus direction
3. Price targets averaged with outlier filtering
4. Final report shows both individual and consensus views

---

## 📝 Output Formats

### Markdown Reports

All scripts generate markdown reports with:

**Header Section:**
```markdown
# {TICKER} Stock Analysis
**Analysis Date:** 2025-10-17 10:30:15
**Chart Image:** [View Chart](https://imagekit.io/...)
```

**Expert Analysis Table:**
```markdown
| Expert | Sentiment | Price Target | Confidence | Time Horizon | Key Reasoning |
|--------|-----------|--------------|------------|--------------|---------------|
| Technical Analyst | BULLISH | $580-$590 | 75% | 5-10 days | Strong momentum... |
```

**Consensus Summary:**
```markdown
## Consensus Analysis
**Consensus Sentiment:** BULLISH
**Consensus Price Target:** $578.50 - $595.25
**Average Confidence:** 72%
```

---

## ⚙️ Advanced Configuration

### Custom Screenshot Directory
```python
analyzer = TradingViewMultiAgentAnalyzer(screenshot_dir="./my_custom_charts")
```

### Different OpenAI Model
```python
analyzer = TradingViewMultiAgentAnalyzer(openai_model="gpt-4o-mini")
```

### Keep Screenshot Files
```python
result = analyzer.analyze_ticker("SPY", cleanup_files=False)
# Screenshots remain in directory after analysis
```

### Skip Chart Capture (Use Existing Image)
```bash
python optimized_chart_analyzer.py --use-png /path/to/existing_chart.png
```

---

## 🐛 Troubleshooting

### "OPENAI_API_KEY not found"
**Solution:** Create `.env` file with your API key
```env
OPENAI_API_KEY=sk-...
```

### "ImageKit upload failed"
**Solution:** Either:
1. Add ImageKit credentials to `.env`
2. OR accept base64 fallback (automatic, works without ImageKit)

### "Chrome driver not found"
**Solution:**
```bash
# The script auto-installs chromedriver, but you can manually:
pip install --upgrade webdriver-manager
```

### Charts not capturing
**Checks:**
1. Chrome/Chromium installed?
2. Internet connection working?
3. TradingView not blocked by firewall?
4. Try `--no-headless` to see browser window

### Analysis seems repetitive
**This is by design!** The multi-agent system can produce similar views when all experts agree. For more diversity:
- Try different tickers with diverse characteristics
- Use v2 with gamma context for more opinionated views
- Check that you're using different expert prompts

---

## 📚 Learning Path

### Beginner
1. Start with `quik.py` to understand basic flow
2. Read the generated `.md` reports
3. Try `run.py` with custom settings

### Intermediate
4. Use `example_usage.py` to see API patterns
5. Import `TradingViewMultiAgentAnalyzer` in your own scripts
6. Experiment with `weekly_outlook_agents.py`

### Advanced
7. Study `optimized_chart_analyzer.py` for performance patterns
8. Use `weekly_outlook_agents_v2.py` with gamma context
9. Extend with custom agents/prompts
10. Build production pipelines with error handling

---

## 📂 Directory Structure

```
enhanced_imagebot/
├── quik.py                              # Quick interactive runner
├── run.py                               # Customizable quick runner
├── tradingview_multi_agent_analyzer.py  # Main 5-agent system
├── optimized_chart_analyzer.py          # Optimized performance version
├── weekly_outlook_agents.py             # Weekly reports (v1)
├── weekly_outlook_agents_v2.py          # Gamma-aware weekly reports (v2)
├── eaglevision.py                       # Legacy TinyTroupe experiment
├── enhanced_imagebot.py                 # Core utilities library
├── example_usage.py                     # Comprehensive examples
├── screenshots/                         # Auto-captured charts
├── my_charts/                          # Custom screenshot location
├── reports/                            # Generated reports
├── .env                                # Your API keys (create this!)
└── README.md                           # This file
```

---

## 🔬 Technical Details

### Chart Capture Pipeline
1. **Selenium WebDriver** launches Chrome
2. Navigates to `https://www.tradingview.com/chart/?symbol={TICKER}`
3. Waits 5 seconds for chart to load
4. Captures full-page screenshot
5. Converts to proper PNG format (PIL)
6. Optionally uploads to ImageKit CDN

### AI Analysis Pipeline
1. **Image Input**: URL (ImageKit) or base64 data URI
2. **GPT-4 Vision**: Analyzes candlestick patterns, indicators, volume
3. **Structured Output**: JSON with sentiment, targets, confidence
4. **Multi-Agent Debate**: Each expert independently analyzes
5. **Consensus**: Weighted voting across all expert opinions
6. **Report Generation**: Beautiful markdown with tables

### Error Handling
- **Screenshot failures**: Graceful degradation
- **Upload failures**: Falls back to base64
- **API errors**: Retry logic and fallbacks
- **Missing dependencies**: Clear error messages

---

## 💡 Best Practices

### 1. Rate Limiting
```python
# For batch processing, add delays
import time
for ticker in watchlist:
    result = analyzer.analyze_ticker(ticker)
    time.sleep(5)  # Avoid OpenAI rate limits
```

### 2. Error Recovery
```python
# Always check for None results
result = analyzer.analyze_ticker("SPY")
if result is None:
    print("Analysis failed, check logs")
    # Implement retry or alert logic
```

### 3. Cost Management
- Each analysis costs ~$0.01-0.05 (GPT-4 Vision API calls)
- Multi-agent = 5 API calls per ticker
- Single-agent = 1 API call per ticker
- Consider using `gpt-4o-mini` for testing

### 4. Screenshot Cleanup
```python
# Clean up old screenshots weekly
analyzer.analyze_ticker("SPY", cleanup_files=True)
```

---

## 🚀 Production Deployment

### Example: Daily Report Service
```python
import schedule
import time
from tradingview_multi_agent_analyzer import TradingViewMultiAgentAnalyzer

def daily_analysis():
    analyzer = TradingViewMultiAgentAnalyzer()
    tickers = ["SPY", "QQQ", "IWM"]
    
    for ticker in tickers:
        result = analyzer.analyze_ticker(ticker)
        if result:
            report = analyzer.generate_markdown_report(result)
            with open(f"reports/{ticker}_{datetime.now():%Y%m%d}.md", "w") as f:
                f.write(report)
            # Send email, post to Slack, etc.

# Run every weekday at 9:30 AM
schedule.every().monday.at("09:30").do(daily_analysis)
schedule.every().tuesday.at("09:30").do(daily_analysis)
# ... etc

while True:
    schedule.run_pending()
    time.sleep(60)
```

---

## 📋 Requirements

**Core:**
- Python 3.8+
- OpenAI API key (GPT-4 Vision access)

**Image Upload (Optional):**
- ImageKit account (free tier available)
- Or use base64 fallback (automatic)

**Chart Capture:**
- Selenium WebDriver
- Chrome/Chromium browser
- webdriver-manager (auto-installs driver)

**Install All:**
```bash
pip install selenium webdriver-manager imagekitio openai python-dotenv pillow httpx
```

---

## 🎓 Learn More

### Concepts Explained

**Multi-Agent Analysis:**
Multiple AI experts analyze independently, then consensus is derived. This reduces bias and provides diverse perspectives.

**GPT-4 Vision:**
OpenAI's vision-capable model that can "see" and interpret chart images, identifying patterns, trends, and technical signals.

**Gamma Regimes:**
Market structure where dealer hedging behavior changes based on options positioning:
- **Positive Gamma**: Stabilizing (dealers buy dips)
- **Negative Gamma**: Amplifying (dealers sell weakness)

### Further Reading
- Multi-agent systems in finance
- Computer vision for trading
- Options market microstructure
- Ensemble forecasting

---

## 🤝 Contributing Ideas

Want to extend this toolkit? Consider:
- Additional expert personas (e.g., Macro Economist, Crypto Specialist)
- Integration with other chart sources (Bloomberg, Yahoo Finance)
- Real-time alert system based on sentiment shifts
- Backtesting framework to validate predictions
- Web dashboard for interactive analysis

---

## 📄 License

Free for personal and educational use.

---

## 🙏 Acknowledgments

Built with:
- **OpenAI GPT-4 Vision** — Chart analysis
- **Selenium** — Web automation
- **ImageKit** — CDN hosting
- **TradingView** — Chart source
- **Pillow** — Image processing

---

**Happy Trading! 📈🤖**

*Disclaimer: This is an AI-powered tool for educational and research purposes. Not financial advice. Always do your own due diligence.*

