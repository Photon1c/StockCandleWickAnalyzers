import os
import time
import base64
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from PIL import Image

# Optional imports guarded at runtime for environments without these packages
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from webdriver_manager.chrome import ChromeDriverManager
except Exception:  # pragma: no cover - optional runtime dependency
    webdriver = None
    Service = None
    Options = None
    ChromeDriverManager = None

try:
    from imagekitio import ImageKit
except Exception:  # pragma: no cover - optional runtime dependency
    ImageKit = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional runtime dependency
    OpenAI = None


load_dotenv()


@dataclass
class Prediction:
    """Structured prediction produced by an expert persona.

    Attributes:
        expert_name: Persona display name.
        sentiment: One of {"bullish", "bearish", "neutral"}.
        confidence: Integer 0-100.
        price_low: Lower bound of expected price range.
        price_high: Upper bound of expected price range.
        horizon_days: Investment time horizon in days.
        rationale: Short, high-signal rationale.
    """

    expert_name: str
    sentiment: str
    confidence: int
    price_low: float
    price_high: float
    horizon_days: int
    rationale: str


class ChartAnalyzer:
    """End-to-end pipeline to capture a TradingView chart, optionally upload it, and run AI analysis.

    This class provides the following capabilities:
    - Capture TradingView chart screenshots with Selenium (headless by default)
    - Convert screenshots to valid PNG files
    - Upload images to ImageKit (if credentials provided) or fallback to inline base64 for OpenAI vision
    - Run single-agent or multi-agent (debate-style) analysis via OpenAI vision
    - Produce a GitHub Markdown table of expert predictions and an ensemble row
    """

    def __init__(
        self,
        screenshot_dir: Optional[str] = None,
        imagekit_private_key: Optional[str] = None,
        imagekit_public_key: Optional[str] = None,
        imagekit_url_endpoint: Optional[str] = None,
        openai_model: Optional[str] = None,
        headless: bool = True,
        pause_seconds: int = 5,
    ) -> None:
        self.screenshot_dir = (
            screenshot_dir
            or os.getenv("SCREENSHOT_DIR")
            or "./screenshots"
        )
        os.makedirs(self.screenshot_dir, exist_ok=True)

        self.imagekit_private_key = imagekit_private_key or os.getenv("IMAGEKIT_PRIVATE_KEY")
        self.imagekit_public_key = imagekit_public_key or os.getenv("IMAGEKIT_PUBLIC_KEY")
        self.imagekit_url_endpoint = imagekit_url_endpoint or os.getenv("IMAGEKIT_URL_ENDPOINT")

        self.openai_model = openai_model or os.getenv("OPENAI_VISION_MODEL") or "gpt-4o"
        self.headless = headless
        self.pause_seconds = pause_seconds

        self._openai_client = None
        if OpenAI is not None:
            try:
                self._openai_client = OpenAI()
            except Exception:
                self._openai_client = None

        self._imagekit_client = None
        if ImageKit is not None and self.imagekit_private_key and self.imagekit_public_key and self.imagekit_url_endpoint:
            try:
                self._imagekit_client = ImageKit(
                    private_key=self.imagekit_private_key,
                    public_key=self.imagekit_public_key,
                    url_endpoint=self.imagekit_url_endpoint,
                )
            except Exception:
                self._imagekit_client = None

    # -------------------------- Selenium / Capture -------------------------- #

    def _build_chrome_options(self) -> Optional[Options]:
        """Construct Chrome options for headless, consistent screenshots.

        Returns:
            Selenium Options instance or None if selenium is unavailable.
        """
        if Options is None:
            return None
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        return chrome_options

    def capture_tradingview_chart(self, symbol: str) -> Optional[str]:
        """Capture a TradingView chart as a proper PNG image.

        Args:
            symbol: Ticker symbol, e.g., "SPY" or "NASDAQ:TSLA".

        Returns:
            Absolute path to the saved PNG file, or None on failure.
        """
        if webdriver is None or Service is None or ChromeDriverManager is None:
            print("Selenium or ChromeDriver is not available in this environment.")
            return None

        raw_path = os.path.join(self.screenshot_dir, f"{symbol}_chart_raw.png")
        png_path = os.path.join(self.screenshot_dir, f"{symbol}_chart.png")

        try:
            chrome_options = self._build_chrome_options()
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)

            tradingview_url = f"https://www.tradingview.com/chart/?symbol={symbol}"
            driver.get(tradingview_url)
            time.sleep(self.pause_seconds)

            driver.save_screenshot(raw_path)
            driver.quit()

            # Convert to a valid PNG (strip alpha to avoid some viewers' issues)
            with Image.open(raw_path) as img:
                rgb = img.convert("RGB")
                rgb.save(png_path, "PNG")

            if os.path.exists(raw_path):
                os.remove(raw_path)

            return png_path if os.path.exists(png_path) else None
        except Exception as exc:  # pragma: no cover - runtime integration
            try:
                # Best-effort cleanup
                if os.path.exists(raw_path):
                    os.remove(raw_path)
            except Exception:
                pass
            print(f"Error capturing TradingView chart: {exc}")
            return None

    # -------------------------- Image Prep / Upload ------------------------- #

    def _upload_to_imagekit(self, image_path: str) -> Optional[str]:
        """Upload an image to ImageKit.

        Returns:
            Public HTTPS URL or None.
        """
        if self._imagekit_client is None:
            return None
        try:
            with open(image_path, "rb") as img_file:
                upload = self._imagekit_client.upload(
                    file=img_file, file_name=os.path.basename(image_path)
                )
            return getattr(upload, "url", None)
        except Exception as exc:  # pragma: no cover - external service
            print(f"Error uploading image to ImageKit: {exc}")
            return None

    def _image_to_data_url(self, image_path: str) -> str:
        """Encode a PNG image as a data URL string suitable for OpenAI vision.

        Args:
            image_path: Absolute path to local PNG file.

        Returns:
            Data URL, e.g., "data:image/png;base64,..."
        """
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def prepare_image_reference(self, image_path: str) -> Optional[str]:
        """Return a URL or data URL reference for use with OpenAI vision.

        Preference order:
        1) ImageKit public URL (if configured)
        2) Inline base64 data URL
        """
        if not os.path.exists(image_path):
            return None
        # Try upload first if configured
        url = self._upload_to_imagekit(image_path)
        if url:
            return url
        # Fallback to inline data URL
        return self._image_to_data_url(image_path)

    # ------------------------------- OpenAI -------------------------------- #

    def _ensure_openai(self) -> bool:
        if self._openai_client is None:
            print("OpenAI client is not available or not configured.")
            return False
        return True

    def _vision_message(self, image_ref: str, user_prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": "You are a stock market expert analyzing candlestick charts."
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_ref}},
                ],
            },
        ]

    def analyze_chart_single(self, image_ref: str, token_limit: int = 450) -> Optional[str]:
        """Run a single-agent OpenAI vision analysis and return the raw text.

        Args:
            image_ref: URL or data URL to the chart image.
            token_limit: Soft cap for the response length.
        """
        if not self._ensure_openai():
            return None

        prompt = (
            "Analyze the stock chart in this image and provide insights on directional "
            "momentum sentiment, trends, volatility, and possible price action in a neat "
            "table report. Limit your response to ~" + str(token_limit) + " tokens and "
            "indicate the perceived sentiment."
        )

        try:
            response = self._openai_client.chat.completions.create(
                model=self.openai_model,
                messages=self._vision_message(image_ref, prompt),
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover - external service
            print(f"Error generating single-agent analysis: {exc}")
            return None

    # ------------------------ Multi-Agent (Debate) -------------------------- #

    def _expert_personas(self) -> List[Tuple[str, str]]:
        """Return (name, persona) tuples for expert agents."""
        return [
            (
                "Technical Analyst",
                (
                    "You are a postdoctoral technical analyst specializing in price action, "
                    "candlestick patterns, and trend-following indicators."
                ),
            ),
            (
                "Macro Strategist",
                (
                    "You are a postdoctoral macro strategist focusing on rates, liquidity, "
                    "and cross-asset flows impacting equities."
                ),
            ),
            (
                "Quant Researcher",
                (
                    "You are a postdoctoral quantitative researcher emphasizing volatility, "
                    "risk premia, and regime detection."
                ),
            ),
            (
                "Risk Manager",
                (
                    "You are a postdoctoral risk manager focusing on drawdowns, tail risks, "
                    "and position sizing across horizons."
                ),
            ),
        ]

    def _persona_messages(self, persona_name: str, persona_desc: str, image_ref: str) -> List[Dict[str, Any]]:
        """Build messages instructing an expert persona to emit a compact JSON prediction."""
        system_prompt = (
            f"You are {persona_name}. {persona_desc} "
            "Respond ONLY with a strict JSON object using these keys: "
            "sentiment (bullish|bearish|neutral), confidence (0-100), "
            "price_low (float), price_high (float), horizon_days (int), "
            "rationale (short, high-signal)."
        )
        user_prompt = (
            "Given the chart image, provide your prediction range and horizon. "
            "If uncertain, widen the range and lower confidence."
        )
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_ref}},
                ],
            },
        ]

    def _call_openai_json(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Call OpenAI and robustly parse a top-level JSON object from the response."""
        if not self._ensure_openai():
            return None
        try:
            response = self._openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.7,
            )
            text = response.choices[0].message.content.strip()
            # Best-effort: extract JSON
            json_str = text
            # Trim code fences if present
            if json_str.startswith("```"):
                json_str = json_str.strip("`\n ")
                # Might start with json\n{...}
                if json_str.lower().startswith("json\n"):
                    json_str = json_str[5:]
            return json.loads(json_str)
        except Exception as exc:  # pragma: no cover - external service / parsing
            print(f"OpenAI JSON parse error: {exc}")
            return None

    def _to_prediction(self, expert_name: str, data: Dict[str, Any]) -> Optional[Prediction]:
        try:
            sentiment = str(data.get("sentiment", "neutral")).lower()
            confidence = int(data.get("confidence", 50))
            price_low = float(data.get("price_low"))
            price_high = float(data.get("price_high"))
            horizon_days = int(data.get("horizon_days", 30))
            rationale = str(data.get("rationale", ""))
            if price_high < price_low:
                price_low, price_high = price_high, price_low
            confidence = max(0, min(100, confidence))
            return Prediction(
                expert_name=expert_name,
                sentiment=sentiment,
                confidence=confidence,
                price_low=price_low,
                price_high=price_high,
                horizon_days=horizon_days,
                rationale=rationale.strip(),
            )
        except Exception as exc:
            print(f"Invalid prediction data for {expert_name}: {exc}")
            return None

    def run_debate(self, image_ref: str) -> List[Prediction]:
        """Run independent expert predictions and return the list of structured outputs."""
        predictions: List[Prediction] = []
        for name, desc in self._expert_personas():
            data = self._call_openai_json(self._persona_messages(name, desc, image_ref))
            if not data:
                continue
            pred = self._to_prediction(name, data)
            if pred:
                predictions.append(pred)
        return predictions

    # -------------------------- Aggregation / Output ------------------------ #

    def _ensemble(self, preds: List[Prediction]) -> Optional[Prediction]:
        if not preds:
            return None
        low = sum(p.price_low for p in preds) / len(preds)
        high = sum(p.price_high for p in preds) / len(preds)
        horizon = int(round(sum(p.horizon_days for p in preds) / len(preds)))
        confidence = int(round(sum(p.confidence for p in preds) / len(preds)))
        # Majority sentiment (tie -> neutral)
        counts: Dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}
        for p in preds:
            if p.sentiment in counts:
                counts[p.sentiment] += 1
            else:
                counts["neutral"] += 1
        sentiment = max(counts.items(), key=lambda kv: kv[1])[0]
        if len({v for v in counts.values() if v == max(counts.values())}) > 1:
            sentiment = "neutral"

        rationale = "Ensemble average of expert ranges and horizons; majority-vote sentiment."
        return Prediction(
            expert_name="Ensemble",
            sentiment=sentiment,
            confidence=confidence,
            price_low=low,
            price_high=high,
            horizon_days=horizon,
            rationale=rationale,
        )

    def to_markdown_table(self, preds: List[Prediction]) -> str:
        """Render predictions as a GitHub Markdown table, including an ensemble row if available."""
        if not preds:
            return "No predictions available."
        ens = self._ensemble(preds)

        header = (
            "| Expert | Sentiment | Confidence | Price Range | Horizon (days) | Key Rationale |\n"
            "|---|---|---:|---|---:|---|\n"
        )
        def row(p: Prediction) -> str:
            price_range = f"{p.price_low:.2f} - {p.price_high:.2f}"
            return (
                f"| {p.expert_name} | {p.sentiment} | {p.confidence}% | "
                f"{price_range} | {p.horizon_days} | {p.rationale.replace('|', '/')} |"
            )

        lines = [header] + [row(p) for p in preds]
        if ens:
            lines.append(row(ens))
        return "\n".join(lines)

    # ------------------------------ Orchestration --------------------------- #

    def analyze_symbol(
        self,
        symbol: str,
        use_existing_png: Optional[str] = None,
        multi_agent: bool = True,
    ) -> Optional[str]:
        """Full pipeline for a given ticker symbol.

        Args:
            symbol: Ticker to analyze (e.g., "SPY").
            use_existing_png: If provided, skip capture and use this local PNG path.
            multi_agent: If True, run multi-agent debate producing a Markdown table; otherwise return single-agent text.

        Returns:
            Markdown string (for multi-agent) or plain text (for single-agent), or None on failure.
        """
        # 1) Capture a chart or use provided path
        png_path = use_existing_png or self.capture_tradingview_chart(symbol)
        if not png_path or not os.path.exists(png_path):
            print("Chart capture failed or file missing.")
            return None

        # 2) Build a reference for OpenAI
        image_ref = self.prepare_image_reference(png_path)
        if not image_ref:
            print("Unable to prepare image reference for OpenAI.")
            return None

        # 3) Run analysis
        if multi_agent:
            preds = self.run_debate(image_ref)
            if not preds:
                print("Multi-agent debate produced no predictions.")
                return None
            return self.to_markdown_table(preds)
        else:
            return self.analyze_chart_single(image_ref)


def main() -> None:  # pragma: no cover - CLI convenience
    import argparse

    parser = argparse.ArgumentParser(description="TradingView chart analyzer with multi-agent debate.")
    parser.add_argument("symbol", type=str, nargs="?", default=os.getenv("TICKER") or "SPY", help="Ticker symbol, e.g., SPY")
    parser.add_argument("--single", action="store_true", help="Run single-agent analysis instead of multi-agent debate")
    parser.add_argument("--use-png", type=str, default=None, help="Use an existing local PNG file (skip capture)")
    parser.add_argument("--no-headless", action="store_true", help="Disable headless Chrome for debugging")
    parser.add_argument("--pause", type=int, default=int(os.getenv("CAPTURE_PAUSE_SECONDS") or 5), help="Seconds to pause after loading chart")

    args = parser.parse_args()

    analyzer = ChartAnalyzer(headless=not args.no_headless, pause_seconds=args.pause)
    result = analyzer.analyze_symbol(
        symbol=args.symbol,
        use_existing_png=args.use_png,
        multi_agent=not args.single,
    )

    if result:
        print("\n=== Analysis Output ===\n")
        print(result)
    else:
        print("Analysis failed.")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()