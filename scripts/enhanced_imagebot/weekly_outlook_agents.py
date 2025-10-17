"""
Weekly Outlook Report Generator (v1)

Purpose:
- Capture a TradingView chart (optional), ask 4 distinct agents for a 1–2 sentence
  weekly outlook, and render a clean markdown report with a consensus snapshot.

Highlights:
- Stable OpenAI client creation that avoids httpx 'proxies' issues (handled in helper).
- Minimal, model-agnostic prompts with a concise output schema.
- Report includes consensus emoji, sentiment breakdown, average confidence,
  and dominant keywords in addition to the per-agent table.

Usage:
  python enhanced_imagebot/weekly_outlook_agents.py \
    --ticker SPY --model gpt-5 --output-dir ./reports --screenshot-dir ./screenshots

Notes:
- This v1 is neutral with no gamma-context lens. See v2 for a gamma-aware rendition.
"""

import os
import json
import time
import base64
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import httpx


# ---------------------------
# Configuration
# ---------------------------
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
DEFAULT_TICKER = os.getenv("TICKER", "SPY")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "enhanced_imagebot/reports")
DEFAULT_SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "./screenshots")
DEFAULT_HEADLESS = os.getenv("HEADLESS", "true").lower() in {"1", "true", "yes"}
DEFAULT_CAPTURE = os.getenv("CAPTURE_CHART", "true").lower() in {"1", "true", "yes"}

# Easily customizable agents (exactly four personas)
AGENTS: List[Dict[str, str]] = [
    {
        "name": "Macro Strategist",
        "persona": "Top-down macro analyst focusing on rates, inflation trends, and liquidity cycles.",
        "style": "Direct, concise, risk-aware",
    },
    {
        "name": "Technical Specialist",
        "persona": "Chart-focused technician emphasizing price action, momentum, and levels.",
        "style": "Pragmatic, momentum-sensitive",
    },
    {
        "name": "Quant Analyst",
        "persona": "Data-driven quant emphasizing volatility regimes and probabilistic outcomes.",
        "style": "Objective, uncertainty-calibrated",
    },
    {
        "name": "Flow Trader",
        "persona": "Order-flow oriented trader watching positioning, options gamma, and flows.",
        "style": "Tactical, flow-aware",
    },
]


# ---------------------------
# Utilities
# ---------------------------

def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # Use PNG as we convert to PNG always
    return f"data:image/png;base64,{b64}"
# ---------------------------
# Lightweight summarizers (for the report)
# ---------------------------

def summarize_keywords(agent_results: List[Dict[str, object]], top_k: int = 5) -> List[str]:
    counts: Dict[str, int] = {}
    for a in agent_results:
        kws = a.get("keywords") or []
        if not isinstance(kws, list):
            kws = [str(kws)]
        for k in kws:
            key = str(k).strip().lower()
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]


def confidence_stats(agent_results: List[Dict[str, object]]) -> Tuple[int, int, int]:
    vals: List[int] = []
    for a in agent_results:
        try:
            vals.append(int(a.get("confidence", 0)))
        except Exception:
            pass
    if not vals:
        return (0, 0, 0)
    avg = round(sum(vals) / len(vals))
    return (avg, min(vals), max(vals))



# ---------------------------
# Screenshot Capture
# ---------------------------

def capture_tradingview_chart(symbol: str, screenshot_dir: str, headless: bool = True) -> Optional[str]:
    """Captures a TradingView chart screenshot and converts it to a proper PNG file.

    Returns the final PNG path on success, or None on failure.
    """
    # Lazy imports so the script can run without these deps if capture is disabled
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from webdriver_manager.chrome import ChromeDriverManager
        from PIL import Image
    except Exception as e:
        print(f"Capture skipped: required dependency not available ({e}).")
        return None

    try:
        print("Initializing WebDriver…")
        def build_options(use_new_headless: bool) -> Options:
            o = Options()
            o.add_argument("--window-size=1920x1080")
            o.add_argument("--disable-blink-features=AutomationControlled")
            o.add_argument("--disable-gpu")
            o.add_argument("--no-sandbox")
            o.add_argument("--disable-dev-shm-usage")
            if headless:
                o.add_argument("--headless=new" if use_new_headless else "--headless")
            return o

        # Use Selenium Manager (no external webdriver-manager) and fall back to legacy headless
        try:
            driver = webdriver.Chrome(options=build_options(True))
        except Exception:
            driver = webdriver.Chrome(options=build_options(False))

        try:
            tradingview_url = f"https://www.tradingview.com/chart/?symbol={symbol}"
            print(f"Opening TradingView URL: {tradingview_url}")
            driver.get(tradingview_url)
            time.sleep(5)

            raw_screenshot_path = os.path.join(screenshot_dir, f"{symbol}_chart_raw.png")
            final_screenshot_path = os.path.join(screenshot_dir, f"{symbol}_chart.png")

            print("Taking screenshot…")
            driver.save_screenshot(raw_screenshot_path)
        finally:
            driver.quit()

        print("Converting screenshot to a valid PNG format…")
        img = Image.open(raw_screenshot_path)
        img = img.convert("RGB")
        img.save(final_screenshot_path, "PNG")
        os.remove(raw_screenshot_path)

        if os.path.exists(final_screenshot_path):
            print(f"Chart successfully saved: {final_screenshot_path}")
            return final_screenshot_path
        else:
            print("Screenshot failed, file not found!")
            return None

    except Exception as e:
        print(f"Error capturing TradingView chart: {e}")
        return None


# ---------------------------
# Image Upload (optional)
# ---------------------------

def upload_image_to_imagekit(image_path: str) -> Optional[str]:
    """Uploads an image to ImageKit and returns the HTTPS URL. Returns None if unavailable.

    Falls back to None if credentials are missing or upload fails.
    """
    try:
        from imagekitio import ImageKit  # lazy import
    except Exception:
        return None

    try:
        private_key = os.getenv("IMAGEKIT_PRIVATE_KEY")
        public_key = os.getenv("IMAGEKIT_PUBLIC_KEY")
        url_endpoint = os.getenv("IMAGEKIT_URL_ENDPOINT")
        if not (private_key and public_key and url_endpoint):
            return None

        if not os.path.exists(image_path):
            return None

        print(f"Uploading {image_path} to ImageKit…")
        imagekit = ImageKit(
            private_key=private_key,
            public_key=public_key,
            url_endpoint=url_endpoint,
        )
        with open(image_path, "rb") as img_file:
            upload = imagekit.upload(file=img_file, file_name=os.path.basename(image_path))
        return upload.url
    except Exception as e:
        print(f"Error uploading image: {e}")
        return None


# ---------------------------
# OpenAI Client (fixes httpx proxies error)
# ---------------------------

def build_openai_client():
    """Create an OpenAI client with a custom httpx.Client to avoid the 'proxies' kwarg issue.

    By providing our own client, we prevent the OpenAI SDK from constructing an httpx.Client
    with an unsupported 'proxies' argument (seen with httpx>=0.28).

    Returns the OpenAI client instance, or None if the SDK is unavailable.
    """
    try:
        from openai import OpenAI  # local import to avoid hard dependency at import-time
    except Exception:
        return None

    http_client = httpx.Client(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
        follow_redirects=True,
        # trust_env=True enables environment proxy variables without passing 'proxies='
        trust_env=True,
    )
    return OpenAI(http_client=http_client)


# Attempt chat completion with model-specific token parameter compatibility
def _complete_json(
    client,
    *,
    model: str,
    system_prompt: str,
    user_content: List[object],
    temperature: float,
    max_new_tokens: int,
) -> Tuple[str, object]:
    """Return (raw_text, raw_response). Tries Responses API first, then Chat Completions."""
    def json_schema_format():
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "weekly_outlook",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string", "enum": ["Bullish", "Bearish", "Neutral"]},
                        "outlook": {"type": "string"},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "integer", "minimum": 0, "maximum": 100}
                    },
                    "required": ["sentiment", "outlook", "keywords", "confidence"],
                    "additionalProperties": False
                }
            }
        }

    def extract_text(resp) -> str:
        try:
            t = getattr(resp, "output_text", None)
            if t:
                return t
        except Exception:
            pass
        try:
            out = getattr(resp, "output", None) or []
            chunks = []
            for item in out:
                content = getattr(item, "content", None) or []
                for c in content:
                    txt = getattr(c, "text", None)
                    if txt:
                        chunks.append(txt)
            if chunks:
                return "\n".join(chunks)
        except Exception:
            pass
        try:
            return resp.choices[0].message.content  # chat fallthrough
        except Exception:
            return ""
    # Try Responses API
    try:
        if hasattr(client, "responses"):
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format=json_schema_format(),
                max_output_tokens=max_new_tokens,
                temperature=temperature,
            )
            raw_text = extract_text(resp)
            if raw_text:
                return raw_text, resp
    except Exception:
        pass

    # Fallback: Chat Completions
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=json_schema_format(),
            max_completion_tokens=max_new_tokens,
            temperature=temperature,
        )
        raw_text = extract_text(resp)
        return raw_text, resp
    except Exception as e:
        # Last resort: try without response_format
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
            )
            raw_text = extract_text(resp)
            return raw_text, resp
        except Exception:
            return "", e


# ---------------------------
# Agent Summaries
# ---------------------------

def _extract_json_block(text: str) -> str:
    """Attempt to extract a JSON object from a model response, stripping code fences if present."""
    if not text:
        return "{}"
    text = text.strip()
    # Remove fenced code blocks if present
    if text.startswith("```)"):
        # unlikely typo guard; handled below
        pass
    if text.startswith("```"):
        text = text.strip("`\n ")
        # If a language hint is present on first line, drop it
        first_newline = text.find("\n")
        if first_newline != -1:
            maybe_lang = text[:first_newline].strip()
            if all(ch.isalpha() or ch in {"-", "#"} for ch in maybe_lang) and len(maybe_lang) <= 20:
                text = text[first_newline + 1 :]
    # Extract between first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def call_agent_summary(client, model: str, agent: Dict[str, str], image_url_or_data: Optional[str], ticker: str) -> Dict[str, object]:
    """Ask one agent for a weekly outlook; return structured JSON-like dict.

    The agent returns: sentiment (Bullish/Bearish/Neutral), outlook (1–2 sentences),
    keywords (list of 3-5), confidence (0-100).
    """
    # If OpenAI client is unavailable, return a neutral placeholder
    if client is None:
        return {
            "name": agent["name"],
            "sentiment": "Neutral",
            "outlook": (
                f"Model unavailable; expect range-bound trade unless a strong catalyst tilts the tape. "
                f"Use intraday momentum and liquidity cues on {ticker} to lean with strength and fade stalls."
            ),
            "keywords": ["range", "momentum", "liquidity", "fade", "catalyst"],
            "confidence": 50,
        }

    system_prompt = (
        f"You are {agent['name']}, {agent['persona']}. "
        f"Write in a {agent['style']} tone."
    )

    user_content: List[object] = [
        {
            "type": "text",
            "text": (
                "Provide a WEEKLY outlook for {ticker} in ONE or TWO sentences. "
                "Return ONLY valid JSON with keys: sentiment ('Bullish'|'Bearish'|'Neutral'), "
                "outlook (string), keywords (array of 3-5 short lowercase keywords), confidence (0-100)."
            ).format(ticker=ticker),
        }
    ]

    if image_url_or_data:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url_or_data},
        })

    try:
        raw_content, _resp_obj = _complete_json(
            client,
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=0.8,
            max_new_tokens=220,
        )
        
        raw = raw_content or "{}"
        raw_json = _extract_json_block(raw)
        try:
            data = json.loads(raw_json)
        except Exception:
            data = {}

        # Normalize with defaults; if outlook missing, salvage first sentence from raw
        sentiment = str(data.get("sentiment", "Neutral")).strip().title()
        outlook = str(data.get("outlook", "")).strip()
        if not outlook:
            if not (raw_content and raw_content.strip()):
                outlook = (
                    f"Weekly outlook unavailable from model; lean neutral unless momentum and liquidity tilt {ticker}. "
                    f"Use pullbacks to join strength and fade failed pushes."
                )
            else:
                s = raw.strip()
                for sep in [". ", "! ", "? "]:
                    idx = s.find(sep)
                    if idx != -1:
                        outlook = (s[: idx + 1]).strip()
                        break
                if not outlook:
                    outlook = "Outlook unavailable; model returned non-JSON content."

        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        keywords = [str(k).strip().lower() for k in keywords][:5]
        confidence = int(max(0, min(100, int(data.get("confidence", 60)))))
        return {
            "name": agent["name"],
            "sentiment": sentiment,
            "outlook": outlook,
            "keywords": keywords if keywords else ["analysis", "weekly"],
            "confidence": confidence,
            "_debug": {"raw": raw_content or "", "parsed": data},
        }
    except Exception as e:
        # Fallback minimal structure
        return {
            "name": agent["name"],
            "sentiment": "Neutral",
            "outlook": f"Unable to generate a detailed outlook due to an error: {e}",
            "keywords": ["uncertain", "error", "retry"],
            "confidence": 40,
        }


def sentiment_to_color_emoji(sentiment: str) -> str:
    s = (sentiment or "").strip().lower()
    if s == "bullish":
        return "🟢 Bullish"
    if s == "bearish":
        return "🔴 Bearish"
    return "🟡 Neutral"


def compute_consensus(sentiments: List[str]) -> str:
    counts = {"bullish": 0, "neutral": 0, "bearish": 0}
    for s in sentiments:
        key = (s or "").strip().lower()
        if key in counts:
            counts[key] += 1
    if counts["bullish"] > max(counts["neutral"], counts["bearish"]):
        return "Bullish"
    if counts["bearish"] > max(counts["neutral"], counts["bullish"]):
        return "Bearish"
    return "Neutral"


# ---------------------------
# Report Generation
# ---------------------------

def generate_markdown_report(
    ticker: str,
    model: str,
    image_ref: Optional[str],
    agent_results: List[Dict[str, object]],
    output_dir: str,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    date_slug = datetime.now(timezone.utc).strftime("%Y%m%d")
    ensure_dirs(output_dir)

    # Consensus
    consensus = compute_consensus([str(a.get("sentiment", "Neutral")) for a in agent_results])
    consensus_label = sentiment_to_color_emoji(consensus)

    # Build table rows
    lines: List[str] = []
    lines.append(f"### {ticker} Weekly Outlook")
    lines.append("")
    lines.append(f"- **Generated**: {ts}")
    lines.append(f"- **Model**: `{model}`")
    if image_ref:
        lines.append(f"- **Chart Image**: {image_ref}")
    lines.append(f"- **Consensus**: {consensus_label}")
    lines.append("")

    # GitHub-flavored markdown table
    lines.append("| Agent | Sentiment | Outlook (1–2 sentences) | Keywords | Confidence |")
    lines.append("|---|---|---|---|---|")
    for a in agent_results:
        name = str(a.get("name", "Agent"))
        sent = sentiment_to_color_emoji(str(a.get("sentiment", "Neutral")))
        outlook = str(a.get("outlook", ""))
        kw_list = [f"`{k}`" for k in (a.get("keywords") or [])]
        keywords = ", ".join(kw_list) if kw_list else "`n/a`"
        conf = f"{int(a.get('confidence', 0))}%"
        lines.append(f"| {name} | {sent} | {outlook} | {keywords} | {conf} |")

    # Consensus snapshot (extra context)
    sentiments = [str(a.get("sentiment", "Neutral")).strip().lower() for a in agent_results]
    bull = sentiments.count("bullish")
    bear = sentiments.count("bearish")
    neut = sentiments.count("neutral")
    avg_conf, min_conf, max_conf = confidence_stats(agent_results)
    top_kws = summarize_keywords(agent_results)

    lines.append("")
    lines.append("#### Consensus Snapshot")
    lines.append(f"- **Sentiment Breakdown**: 🟢 {bull} | 🟡 {neut} | 🔴 {bear}")
    lines.append(f"- **Average Confidence**: {avg_conf}% (range {min_conf}–{max_conf}%)")
    if top_kws:
        lines.append(f"- **Dominant Keywords**: {', '.join(f'`{k}`' for k in top_kws)}")

    # Per-agent quick briefs
    def first_sentence(s: str) -> str:
        s = (s or "").strip()
        for sep in [". ", "! ", "? "]:
            idx = s.find(sep)
            if idx != -1:
                return (s[: idx + 1]).strip()
        return s

    lines.append("")
    lines.append("#### Per-Agent Briefs")
    for a in agent_results:
        name = str(a.get("name", "Agent"))
        sent = sentiment_to_color_emoji(str(a.get("sentiment", "Neutral")))
        brief = first_sentence(str(a.get("outlook", "")))
        lines.append(f"- **{name}** — {sent}: {brief}")

    content = "\n".join(lines) + "\n"

    out_path = os.path.join(output_dir, f"{ticker}_weekly_outlook_{date_slug}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    return out_path


# ---------------------------
# Main Orchestration
# ---------------------------

def run(ticker: str, model: str, output_dir: str, screenshot_dir: str, capture_chart: bool, headless: bool) -> Tuple[Optional[str], Optional[str]]:
    load_dotenv()

    ensure_dirs(output_dir, screenshot_dir)

    image_path: Optional[str] = None
    image_url_or_data: Optional[str] = None
    image_link_for_report: Optional[str] = None

    if capture_chart:
        print("\n=== Capturing TradingView Chart ===")
        image_path = capture_tradingview_chart(ticker, screenshot_dir=screenshot_dir, headless=headless)

    if image_path:
        # Try to upload; if unavailable, fallback to data URL
        print("\nAttempting image upload…")
        uploaded_url = upload_image_to_imagekit(image_path)
        if uploaded_url:
            image_url_or_data = uploaded_url
            image_link_for_report = uploaded_url
            # clean up local file after successful upload
            try:
                os.remove(image_path)
            except Exception:
                pass
        else:
            print("ImageKit upload unavailable; embedding as data URL.")
            image_url_or_data = to_data_url(image_path)
            image_link_for_report = "(embedded image)"
            # keep local file so the user can inspect

    # Build OpenAI client with custom httpx.Client to avoid 'proxies' kwarg error
    client = build_openai_client()

    print("\n=== Generating Weekly Outlooks (4 agents) ===")
    agent_results: List[Dict[str, object]] = []
    for agent in AGENTS:
        result = call_agent_summary(client, model, agent, image_url_or_data, ticker)
        agent_results.append(result)

    print("\n=== Writing Markdown Report ===")
    report_path = generate_markdown_report(
        ticker=ticker,
        model=model,
        image_ref=image_link_for_report,
        agent_results=agent_results,
        output_dir=output_dir,
    )

    print(f"\nReport saved: {report_path}")
    return report_path, image_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly outlook generator with 4 customizable agents and markdown report.")
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--screenshot-dir", default=DEFAULT_SCREENSHOT_DIR)
    parser.add_argument("--headless", action="store_true" if not DEFAULT_HEADLESS else "store_false",
                        help=("Run Chrome in headless mode (default)" if DEFAULT_HEADLESS else "Run Chrome with a visible window"))
    parser.add_argument("--no-screenshot", action="store_true", help="Skip capturing TradingView chart")

    args = parser.parse_args()

    # Determine effective headless and capture flags based on defaults and CLI
    effective_headless = DEFAULT_HEADLESS if not args.headless else (not DEFAULT_HEADLESS)
    effective_capture = DEFAULT_CAPTURE and (not args.no_screenshot)

    run(
        ticker=args.ticker,
        model=args.model,
        output_dir=args.output_dir,
        screenshot_dir=args.screenshot_dir,
        capture_chart=effective_capture,
        headless=effective_headless,
    )