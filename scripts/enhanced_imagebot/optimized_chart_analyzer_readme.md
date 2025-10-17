# cursor-agent-workspace
Repository for autonomous cursor agents to work in, continually updated.

## Optimized Chart Analyzer

Create a `.env` file with your credentials:

```
OPENAI_API_KEY=...
IMAGEKIT_PRIVATE_KEY=...
IMAGEKIT_PUBLIC_KEY=...
IMAGEKIT_URL_ENDPOINT=...
SCREENSHOT_DIR=/workspace/screenshots
OPENAI_VISION_MODEL=gpt-4o
```

Install dependencies:

```
pip install -r requirements.txt
```

Run multi-agent debate (default):

```
python optimized_chart_analyzer.py SPY
```

Run single-agent analysis:

```
python optimized_chart_analyzer.py SPY --single
```

Use an existing PNG (skip capture):

```
python optimized_chart_analyzer.py --use-png /abs/path/to/chart.png
```
