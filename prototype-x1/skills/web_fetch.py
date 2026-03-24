"""
Web fetch skill — lets ARIA fetch and read web pages.

Auto-registered as: web_fetch
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("x1.web_fetch")

# Max chars to return (keep context window manageable)
_MAX_CHARS = 4000


def _strip_html(html: str) -> str:
    """Rough HTML → plain text. Keeps it lightweight — no extra deps."""
    # Remove script/style blocks
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Decode common entities
    for entity, char in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        text = text.replace(entity, char)
    return text


def skill_web_fetch(url: str) -> str:
    """Fetch a web page and return its text content.

    Args:
        url: The URL to fetch (must start with http:// or https://).

    Returns:
        Plain text content of the page, truncated to ~4000 chars.
    """
    import urllib.request
    import urllib.error

    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; ARIA-Bot/1.0)",
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json",
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(200_000)  # cap download at 200KB

            # Detect encoding
            charset = "utf-8"
            if "charset=" in content_type:
                charset = content_type.split("charset=")[-1].split(";")[0].strip()

            text = raw.decode(charset, errors="replace")

            # If HTML, strip tags
            if "html" in content_type.lower():
                text = _strip_html(text)

            # Truncate
            if len(text) > _MAX_CHARS:
                text = text[:_MAX_CHARS] + f"\n\n[…truncated — {len(text)} total chars]"

            log.info("web_fetch OK: %s (%d chars)", url[:80], len(text))
            return text

    except urllib.error.HTTPError as e:
        return f"[WEB_FETCH ERROR] HTTP {e.code}: {e.reason} — {url}"
    except urllib.error.URLError as e:
        return f"[WEB_FETCH ERROR] {e.reason} — {url}"
    except Exception as e:
        return f"[WEB_FETCH ERROR] {e}"

