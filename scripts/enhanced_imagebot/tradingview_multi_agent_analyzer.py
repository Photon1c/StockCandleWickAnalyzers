"""
TradingView Multi-Agent Chart Analyzer

An optimized module for capturing, analyzing, and generating multi-expert predictions
for stock charts using TradingView data, computer vision, and AI-powered analysis.

This module implements a debate system between postdoctoral-level AI experts who
provide diverse perspectives on market analysis and generate consensus predictions.

Author: AI Assistant
License: MIT
"""

import os
import time
import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from imagekitio import ImageKit
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image


@dataclass
class ExpertPrediction:
    """Data class for expert prediction results."""
    expert_name: str
    sentiment: str
    price_target_low: float
    price_target_high: float
    confidence: int
    time_horizon: str
    key_reasoning: str


@dataclass
class AnalysisResult:
    """Data class for complete analysis results."""
    ticker: str
    current_price: Optional[float]
    expert_predictions: List[ExpertPrediction]
    consensus_sentiment: str
    consensus_target_range: Tuple[float, float]
    image_url: str
    analysis_timestamp: datetime


class TradingViewCaptureService:
    """Service for capturing TradingView charts with enhanced reliability."""
    
    def __init__(self, screenshot_dir: str = "./screenshots"):
        """
        Initialize the capture service.
        
        Args:
            screenshot_dir: Directory to save screenshots
        """
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def _setup_driver(self) -> webdriver.Chrome:
        """Set up Chrome WebDriver with optimized options."""
        chrome_options = Options()
        chrome_options.add_argument("--window-size=1920x1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-web-security")
        chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        return driver
    
    def capture_chart(self, symbol: str, timeframe: str = "1D") -> Optional[str]:
        """
        Capture a TradingView chart for the given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'SPY', 'AAPL')
            timeframe: Chart timeframe (e.g., '1D', '1W', '1M')
            
        Returns:
            Path to the saved screenshot or None if failed
        """
        driver = None
        try:
            self.logger.info(f"Capturing chart for {symbol}")
            driver = self._setup_driver()
            
            # Navigate to TradingView
            url = f"https://www.tradingview.com/chart/?symbol={symbol}&interval={timeframe}"
            driver.get(url)
            
            # Wait for chart to load
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-name='legend-source-item']")))
            
            # Additional wait for chart rendering
            time.sleep(5)
            
            # Take screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_path = self.screenshot_dir / f"{symbol}_{timestamp}_raw.png"
            final_path = self.screenshot_dir / f"{symbol}_{timestamp}.png"
            
            driver.save_screenshot(str(raw_path))
            
            # Convert to proper PNG format
            img = Image.open(raw_path)
            img = img.convert("RGB")
            img.save(final_path, "PNG", optimize=True)
            
            # Clean up raw file
            raw_path.unlink(missing_ok=True)
            
            self.logger.info(f"Chart captured successfully: {final_path}")
            return str(final_path)
            
        except Exception as e:
            self.logger.error(f"Error capturing chart for {symbol}: {e}")
            return None
        finally:
            if driver:
                driver.quit()


class ImageUploadService:
    """Service for uploading images to ImageKit with enhanced error handling."""
    
    def __init__(self):
        """Initialize ImageKit client with environment variables."""
        load_dotenv()
        self.imagekit = ImageKit(
            private_key=os.getenv("IMAGEKIT_PRIVATE_KEY"),
            public_key=os.getenv("IMAGEKIT_PUBLIC_KEY"),
            url_endpoint=os.getenv("IMAGEKIT_URL_ENDPOINT")
        )
        self.logger = logging.getLogger(__name__)
    
    def upload_image(self, image_path: str) -> Optional[str]:
        """
        Upload image to ImageKit and return HTTPS URL.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            HTTPS URL of uploaded image or None if failed
        """
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return None
            
            self.logger.info(f"Uploading {image_path} to ImageKit")
            
            with open(image_path, "rb") as img_file:
                upload = self.imagekit.upload(
                    file=img_file,
                    file_name=os.path.basename(image_path)
                )
            
            self.logger.info("Image uploaded successfully")
            return upload.url
            
        except Exception as e:
            self.logger.error(f"Error uploading image: {e}")
            return None


class MultiAgentAnalysisEngine:
    """Multi-agent analysis engine with postdoctoral-level expert personas."""
    
    def __init__(self):
        """Initialize the analysis engine with OpenAI client."""
        load_dotenv()
        self.client = OpenAI()
        self.logger = logging.getLogger(__name__)
        
        # Define expert personas
        self.experts = {
            "technical_analyst": {
                "name": "Dr. Sarah Chen",
                "specialty": "Technical Analysis & Chart Patterns",
                "background": "PhD in Financial Engineering, 15+ years in quantitative trading",
                "perspective": "Focus on technical indicators, support/resistance, and pattern recognition"
            },
            "fundamental_analyst": {
                "name": "Dr. Michael Rodriguez",
                "specialty": "Fundamental Analysis & Economic Indicators", 
                "background": "PhD in Economics, former Federal Reserve researcher",
                "perspective": "Focus on economic fundamentals, earnings trends, and macro factors"
            },
            "risk_analyst": {
                "name": "Dr. Jennifer Kim",
                "specialty": "Risk Management & Volatility Analysis",
                "background": "PhD in Financial Mathematics, specializes in derivatives and risk modeling",
                "perspective": "Focus on volatility patterns, risk-adjusted returns, and downside protection"
            },
            "behavioral_analyst": {
                "name": "Dr. Robert Thompson",
                "specialty": "Behavioral Finance & Market Psychology",
                "background": "PhD in Behavioral Economics, studies market sentiment and crowd psychology",
                "perspective": "Focus on market sentiment, investor psychology, and contrarian indicators"
            }
        }
    
    def _get_expert_analysis(self, expert_key: str, image_url: str, ticker: str) -> Optional[ExpertPrediction]:
        """Get analysis from a specific expert persona."""
        expert = self.experts[expert_key]
        
        prompt = f"""
        You are {expert['name']}, a {expert['specialty']} expert with {expert['background']}.
        
        Your analytical perspective: {expert['perspective']}
        
        Analyze this {ticker} chart image and provide your expert opinion in the following JSON format:
        {{
            "sentiment": "bullish|bearish|neutral",
            "price_target_low": <number>,
            "price_target_high": <number>, 
            "confidence": <1-100>,
            "time_horizon": "1-3 months|3-6 months|6-12 months",
            "key_reasoning": "<2-3 sentence explanation of your analysis>"
        }}
        
        Base your analysis on your area of expertise and provide realistic price targets.
        Consider current market conditions and your specialized knowledge.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Please analyze this {ticker} chart from your expert perspective."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            # Parse JSON response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            json_str = content[start_idx:end_idx] if start_idx != -1 and end_idx != 0 else content
            
            analysis_data = json.loads(json_str)
            
            return ExpertPrediction(
                expert_name=expert['name'],
                sentiment=analysis_data['sentiment'],
                price_target_low=float(analysis_data['price_target_low']),
                price_target_high=float(analysis_data['price_target_high']),
                confidence=int(analysis_data['confidence']),
                time_horizon=analysis_data['time_horizon'],
                key_reasoning=analysis_data['key_reasoning']
            )
            
        except Exception as e:
            self.logger.error(f"Error getting analysis from {expert['name']}: {e}")
            return None
    
    def conduct_expert_debate(self, image_url: str, ticker: str) -> List[ExpertPrediction]:
        """Conduct multi-agent analysis with all experts."""
        predictions = []
        
        for expert_key in self.experts.keys():
            self.logger.info(f"Getting analysis from {self.experts[expert_key]['name']}")
            prediction = self._get_expert_analysis(expert_key, image_url, ticker)
            
            if prediction:
                predictions.append(prediction)
            
            # Small delay to avoid rate limiting
            time.sleep(1)
        
        return predictions
    
    def generate_consensus(self, predictions: List[ExpertPrediction]) -> Tuple[str, Tuple[float, float]]:
        """Generate consensus sentiment and price range from expert predictions."""
        if not predictions:
            return "neutral", (0.0, 0.0)
        
        # Weighted consensus based on confidence
        total_weight = sum(p.confidence for p in predictions)
        
        sentiment_scores = {"bullish": 0, "neutral": 0, "bearish": 0}
        price_lows = []
        price_highs = []
        
        for pred in predictions:
            weight = pred.confidence / total_weight
            sentiment_scores[pred.sentiment] += weight
            price_lows.append(pred.price_target_low)
            price_highs.append(pred.price_target_high)
        
        # Determine consensus sentiment
        consensus_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        
        # Calculate consensus price range
        consensus_low = sum(price_lows) / len(price_lows)
        consensus_high = sum(price_highs) / len(price_highs)
        
        return consensus_sentiment, (consensus_low, consensus_high)


class TradingViewMultiAgentAnalyzer:
    """Main analyzer class that orchestrates the complete analysis process."""
    
    def __init__(self, screenshot_dir: str = "./screenshots"):
        """
        Initialize the analyzer with all required services.
        
        Args:
            screenshot_dir: Directory for storing screenshots
        """
        self.capture_service = TradingViewCaptureService(screenshot_dir)
        self.upload_service = ImageUploadService()
        self.analysis_engine = MultiAgentAnalysisEngine()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def analyze_ticker(self, ticker: str, cleanup_files: bool = True) -> Optional[AnalysisResult]:
        """
        Perform complete multi-agent analysis for a given ticker.
        
        Args:
            ticker: Stock symbol to analyze
            cleanup_files: Whether to delete local files after analysis
            
        Returns:
            AnalysisResult object with complete analysis data
        """
        try:
            # Step 1: Capture chart
            self.logger.info(f"Starting analysis for {ticker}")
            chart_path = self.capture_service.capture_chart(ticker)
            
            if not chart_path:
                self.logger.error(f"Failed to capture chart for {ticker}")
                return None
            
            # Step 2: Upload image
            image_url = self.upload_service.upload_image(chart_path)
            
            if not image_url:
                self.logger.error(f"Failed to upload image for {ticker}")
                return None
            
            # Step 3: Conduct multi-agent analysis
            expert_predictions = self.analysis_engine.conduct_expert_debate(image_url, ticker)
            
            if not expert_predictions:
                self.logger.error(f"Failed to get expert predictions for {ticker}")
                return None
            
            # Step 4: Generate consensus
            consensus_sentiment, consensus_range = self.analysis_engine.generate_consensus(expert_predictions)
            
            # Step 5: Create result object
            result = AnalysisResult(
                ticker=ticker,
                current_price=None,  # Could be enhanced to fetch current price
                expert_predictions=expert_predictions,
                consensus_sentiment=consensus_sentiment,
                consensus_target_range=consensus_range,
                image_url=image_url,
                analysis_timestamp=datetime.now()
            )
            
            # Cleanup local files if requested
            if cleanup_files and os.path.exists(chart_path):
                os.remove(chart_path)
                self.logger.info(f"Cleaned up local file: {chart_path}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    def generate_markdown_report(self, result: AnalysisResult) -> str:
        """
        Generate a GitHub markdown table report from analysis results.
        
        Args:
            result: AnalysisResult object
            
        Returns:
            Formatted markdown report
        """
        report = f"""# Multi-Agent Analysis Report: {result.ticker}

**Analysis Timestamp:** {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Chart Image:** [View Chart]({result.image_url})

## Expert Predictions

| Expert | Specialty | Sentiment | Price Target Range | Confidence | Time Horizon | Key Reasoning |
|--------|-----------|-----------|-------------------|------------|--------------|---------------|
"""
        
        for pred in result.expert_predictions:
            sentiment_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}[pred.sentiment]
            report += f"| {pred.expert_name} | {pred.expert_name.split()[-1]} Analysis | {sentiment_emoji} {pred.sentiment.title()} | ${pred.price_target_low:.2f} - ${pred.price_target_high:.2f} | {pred.confidence}% | {pred.time_horizon} | {pred.key_reasoning} |\n"
        
        consensus_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}[result.consensus_sentiment]
        
        report += f"""
## Consensus Analysis

**Overall Sentiment:** {consensus_emoji} {result.consensus_sentiment.title()}  
**Consensus Price Range:** ${result.consensus_target_range[0]:.2f} - ${result.consensus_target_range[1]:.2f}

## Investment Recommendations

Based on the multi-agent analysis, here are the consolidated investment perspectives:

### Short-term (1-3 months)
- **Sentiment:** Monitor for {result.consensus_sentiment} signals
- **Risk Level:** Assess based on consensus confidence levels

### Medium-term (3-6 months)  
- **Target Range:** ${result.consensus_target_range[0]:.2f} - ${result.consensus_target_range[1]:.2f}
- **Strategy:** Consider expert reasoning for position sizing

### Long-term (6+ months)
- **Outlook:** Review fundamental and technical alignment
- **Monitoring:** Track key indicators mentioned by experts

---
*This analysis is generated by AI experts and should not be considered as financial advice. Always conduct your own research and consult with financial professionals before making investment decisions.*
"""
        
        return report


def main():
    """Example usage of the TradingView Multi-Agent Analyzer."""
    # Initialize analyzer
    analyzer = TradingViewMultiAgentAnalyzer()
    
    # Analyze a ticker
    ticker = "SPY"
    result = analyzer.analyze_ticker(ticker)
    
    if result:
        # Generate and print markdown report
        report = analyzer.generate_markdown_report(result)
        print(report)
        
        # Optionally save to file
        with open(f"{ticker}_analysis_report.md", "w") as f:
            f.write(report)
        print(f"\nReport saved to {ticker}_analysis_report.md")
    else:
        print(f"Analysis failed for {ticker}")


if __name__ == "__main__":
    main()