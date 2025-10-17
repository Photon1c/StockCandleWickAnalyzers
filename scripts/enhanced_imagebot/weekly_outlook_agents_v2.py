"""
Weekly Outlook Report Generator (v2, Gamma-Aware)

Purpose:
- Same fundamentals as v1, but agents are instructed to use a "gamma-hill" mental model
  (bowl/ridge/plateau) to bias toward decisive Bullish/Bearish stances and reduce Neutral.

Key Differences vs v1:
- Adds `--gamma-state` hint (positive | negative | neutral | unknown) and surfaces this
  context in both prompting and the final report.
- Uses a slightly more opinionated prompt and fallback logic to avoid bland neutrality.

Usage:
  python enhanced_imagebot/weekly_outlook_agents_v2.py \
    --ticker SPY --model gpt-5 --gamma-state positive \
    --output-dir ./reports --screenshot-dir ./screenshots
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
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # Changed from gpt-5 to gpt-4o (actual model)
DEFAULT_TICKER = os.getenv("TICKER", "SPY")
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "enhanced_imagebot/reports")
DEFAULT_SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "./screenshots")
DEFAULT_HEADLESS = os.getenv("HEADLESS", "true").lower() in {"1", "true", "yes"}
DEFAULT_CAPTURE = os.getenv("CAPTURE_CHART", "true").lower() in {"1", "true", "yes"}
# Add optional gamma context from env
DEFAULT_GAMMA_STATE = os.getenv("GAMMA_STATE", "").strip().lower()

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

# Normalize gamma state helper
def normalize_gamma_state(s: Optional[str]) -> str:
    s = (s or "").strip().lower()
    if s in {"pos", "+", "positive", "long", "bullish"}: return "positive"
    if s in {"neg", "-", "negative", "short", "bearish"}: return "negative"
    if s in {"neutral", "flat", "zero"}: return "neutral"
    return "unknown"


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
            return resp.choices[0].message.content
        except Exception:
            return ""
    # Try Responses API first (if available in newer models)
    # Note: Skip for gpt-4o as it uses Chat Completions API
    if model not in ["gpt-4o", "gpt-4", "gpt-4o-mini"]:
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
                    print(f"✅ Response API succeeded ({len(raw_text)} chars)")
                    return raw_text, resp
        except Exception:
            pass  # Silently fall through to Chat Completions

    # Try Chat Completions with JSON mode
    try:
        print(f"Trying Chat Completions API with model: {model}")
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
        print(f"✅ Chat Completions succeeded ({len(raw_text) if raw_text else 0} chars)")
        return raw_text, resp
    except Exception as e:
        print(f"⚠️ JSON mode failed: {e}, trying without JSON schema...")
        # Fallback without strict JSON schema
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt + "\n\nIMPORTANT: Return ONLY valid JSON with no additional text."},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=max_new_tokens,
                temperature=temperature,
            )
            raw_text = extract_text(resp)
            print(f"✅ Fallback mode succeeded ({len(raw_text) if raw_text else 0} chars)")
            return raw_text, resp
        except Exception as e2:
            print(f"❌ All API attempts failed: {e2}")
            return "", e2


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


def call_agent_summary(client, model: str, agent: Dict[str, str], image_url_or_data: Optional[str], ticker: str, gamma_state: str) -> Dict[str, object]:
    """Ask one agent for a weekly outlook; return structured JSON-like dict.

    The agent returns: sentiment (Bullish/Bearish/Neutral), outlook (1–2 sentences),
    keywords (list of 3-5), confidence (0-100).
    """
    gamma_norm = normalize_gamma_state(gamma_state)

    # If OpenAI client is unavailable, craft a decisive, analogy-driven fallback
    if client is None:
        if gamma_norm == "positive":
            sentiment = "Bullish"
            outlook = (
                "Gamma looks like a bowl— the price-ball tends to roll back toward the valley; "
                "expect dips to find 'brakes' with a gentle lean higher unless a strong shock tips the slope."
            )
            keywords = ["bowl", "mean-reversion", "dampening", "pullback-buying", "brakes"]
            confidence = 66
        elif gamma_norm == "negative":
            sentiment = "Bearish"
            outlook = (
                "Gamma resembles a ridge— the price-ball accelerates away on pushes; "
                "momentum can expand and range 'rails' give way faster, so manage risk for trend bursts."
            )
            keywords = ["ridge", "acceleration", "momentum", "expansion", "trend"]
            confidence = 66
        elif gamma_norm == "neutral":
            sentiment = "Neutral"
            outlook = (
                "Flat plateau— the price-ball sits pinned near a magnet; "
                "expect rangey chop until the slope tilts, then follow the first real roll."
            )
            keywords = ["plateau", "pinning", "magnet", "range", "tilt"]
            confidence = 55
        else:
            # Unknown: light persona-based tilt to reduce excessive neutrality
            sentiment = "Bearish" if agent.get("name") == "Flow Trader" else "Bullish"
            outlook = (
                "Slope unclear— trade it like a shallow hill: let the first tilt decide; "
                "fade minor rolls if they stall, ride with acceleration if the ball commits."
            )
            keywords = ["tilt", "acceleration", "fade", "commit", "hill"]
            confidence = 60

        return {
            "name": agent["name"],
            "sentiment": sentiment,
            "outlook": outlook,
            "keywords": keywords,
            "confidence": confidence,
        }

    system_prompt = (
        f"You are {agent['name']}, {agent['persona']}. Write in a {agent['style']} tone.\n"
        "Use the 'gamma hill' mental model: "
        "positive gamma = a bowl/valley where the ball (price) rolls back toward center (mean-reversion, damping); "
        "negative gamma = an inverted hill/ridge where the ball accelerates away (trend, expansion); "
        "neutral/pinned = a flat plateau near heavy OI where the ball barely moves (pinning).\n"
        "Decide sentiment with bias toward Bullish/Bearish and use Neutral only if pinning is truly dominant. "
        "Offer 1–2 practical sentences with an actionable angle (what to watch: slope, pinning, likely path or trigger). "
        "Avoid specific levels since strikes aren’t provided; use plain-language cues like 'magnet', 'ridge', 'valley', 'acceleration', 'brakes'."
    )

    gamma_hint = (
        "Gamma context hint: 'positive' = bowl/mean-reversion, 'negative' = ridge/trend, 'neutral' = flat/pin. "
        f"Context for {ticker}: {gamma_norm}."
    )

    user_content: List[object] = [
        {
            "type": "text",
            "text": (
                "Provide a WEEKLY outlook for {ticker} in ONE or TWO sentences using the gamma-hill analogy. "
                "Return ONLY valid JSON with keys: sentiment ('Bullish'|'Bearish'|'Neutral'), "
                "outlook (string), keywords (array of 3-5 short lowercase keywords), confidence (0-100). "
                "Favor Bullish or Bearish unless the hill is truly flat/pinned; Neutral must be rare and justified. "
                "{gamma_hint}"
            ).format(ticker=ticker, gamma_hint=gamma_hint),
        }
    ]

    if image_url_or_data:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url_or_data},
        })

    try:
        raw, _resp_obj = _complete_json(
            client,
            model=model,
            system_prompt=system_prompt,
            user_content=user_content,
            temperature=0.9,
            max_new_tokens=240,
        )
        
        # Debug logging
        print(f"\n[{agent['name']}] Raw response length: {len(raw) if raw else 0}")
        
        raw = raw or "{}"
        raw_json = _extract_json_block(raw)
        
        # Try to parse JSON
        data = {}
        try:
            data = json.loads(raw_json)
            print(f"[{agent['name']}] ✅ JSON parsed successfully")
        except json.JSONDecodeError as e:
            print(f"[{agent['name']}] ⚠️ JSON parse error: {e}")
            print(f"[{agent['name']}] Raw JSON block: {raw_json[:200]}...")
            # Try to extract from raw text if JSON parsing fails
            if raw:
                # Look for sentiment keywords in text
                raw_lower = raw.lower()
                if "bullish" in raw_lower:
                    data["sentiment"] = "Bullish"
                elif "bearish" in raw_lower:
                    data["sentiment"] = "Bearish"
                else:
                    data["sentiment"] = "Neutral"
                # Use the raw text as outlook
                data["outlook"] = raw[:200].strip()
        
        # Normalize with defaults
        sentiment = str(data.get("sentiment", "Neutral")).strip().title()
        outlook = str(data.get("outlook", "")).strip()
        
        if not outlook or outlook == "Outlook unavailable; model returned non-JSON content.":
            # Try to extract meaningful text from raw response
            if raw and len(raw.strip()) > 10:
                # Clean up the raw response
                clean_text = raw.strip()
                # Remove JSON artifacts
                clean_text = clean_text.replace("```json", "").replace("```", "").strip()
                # Take first 1-2 sentences
                sentences = []
                for sep in [". ", "! ", "? "]:
                    parts = clean_text.split(sep)
                    if len(parts) > 1:
                        sentences.append(parts[0].strip() + sep.strip())
                        if len(parts) > 2:
                            sentences.append(parts[1].strip() + sep.strip())
                        break
                outlook = " ".join(sentences[:2]) if sentences else clean_text[:200]
            
            if not outlook:
                outlook = f"Analysis in progress for {ticker} (model response formatting issue)."
        
        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = [str(keywords)]
        keywords = [str(k).strip().lower() for k in keywords if k][:5]
        if not keywords:
            keywords = ["weekly", "outlook", ticker.lower()]
        
        confidence = int(max(0, min(100, int(data.get("confidence", 66)))))

        # Encourage decisive stance if Neutral
        if sentiment.lower() == "neutral":
            if gamma_norm == "positive":
                sentiment = "Bullish"
            elif gamma_norm == "negative":
                sentiment = "Bearish"
            else:
                # Light persona-based tilt
                if agent.get("name") == "Flow Trader":
                    sentiment = "Bearish"
                else:
                    sentiment = "Bullish"

        result = {
            "name": agent["name"],
            "sentiment": sentiment,
            "outlook": outlook,
            "keywords": keywords if keywords else ["gamma", "weekly"],
            "confidence": confidence,
        }
        # Encourage decisive stance if Neutral
        if result["sentiment"].lower() == "neutral":
            if gamma_norm == "positive":
                result["sentiment"] = "Bullish"
            elif gamma_norm == "negative":
                result["sentiment"] = "Bearish"
        return result
    except Exception as e:
        # Fallback minimal structure
        fallback_sent = "Bullish" if normalize_gamma_state(gamma_state) != "negative" else "Bearish"
        return {
            "name": agent["name"],
            "sentiment": fallback_sent,
            "outlook": f"Unable to generate a detailed outlook due to an error: {e}",
            "keywords": ["uncertain", "gamma-hill", "fallback"],
            "confidence": 45,
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


# Extra: share v1's quick summarizers for a richer report
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
# Report Generation
# ---------------------------

def generate_markdown_report(
    ticker: str,
    model: str,
    image_ref: Optional[str],
    agent_results: List[Dict[str, object]],
    output_dir: str,
    gamma_context: str,
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
    if gamma_context and gamma_context != "unknown":
        lines.append(f"- **Gamma Context**: `{gamma_context}`")
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

def run(ticker: str, model: str, output_dir: str, screenshot_dir: str, capture_chart: bool, headless: bool, gamma_state: str) -> Tuple[Optional[str], Optional[str]]:
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

    gamma_norm = normalize_gamma_state(gamma_state)

    print("\n=== Generating Weekly Outlooks (4 agents) ===")
    agent_results: List[Dict[str, object]] = []
    for agent in AGENTS:
        result = call_agent_summary(client, model, agent, image_url_or_data, ticker, gamma_norm)
        agent_results.append(result)

    print("\n=== Writing Markdown Report ===")
    report_path = generate_markdown_report(
        ticker=ticker,
        model=model,
        image_ref=image_link_for_report,
        agent_results=agent_results,
        output_dir=output_dir,
        gamma_context=gamma_norm,
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
    parser.add_argument("--gamma-state", default=DEFAULT_GAMMA_STATE, help="Gamma context hint: positive | negative | neutral | unknown")

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
        gamma_state=args.gamma_state,
    )
