import asyncio
import csv
import json
import random
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, TimeoutError as PWTimeoutError


# -----------------------------
# Helpers
# -----------------------------

def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def slugify(s: str, max_len: int = 140) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", s)
    return s[:max_len].strip("_")


def load_urls(path: str) -> List[str]:
    urls: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)
    return urls


def looks_like_block_or_challenge(html: str, final_url: str) -> Tuple[bool, str]:
    low = (html or "").lower()
    url_low = (final_url or "").lower()

    patterns = [
        ("captcha", "contains 'captcha'"),
        ("cloudflare", "contains 'cloudflare'"),
        ("attention required", "contains 'attention required'"),
        ("verify you are human", "contains 'verify you are human'"),
        ("checking your browser", "contains 'checking your browser'"),
        ("access denied", "contains 'access denied'"),
    ]
    for needle, why in patterns:
        if needle in low:
            return True, why

    if any(x in url_low for x in ["captcha", "challenge", "blocked", "verify"]):
        return True, "final url looks like a challenge/captcha"

    if len(low) < 1500:
        return True, "html too short (possible interstitial or load failure)"

    return False, ""


def default_headers() -> Dict[str, str]:
    return {
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Upgrade-Insecure-Requests": "1",
    }


async def polite_delay(cfg: Dict[str, Any]) -> None:
    mn = float(cfg.get("min_delay_sec", 0.0))
    mx = float(cfg.get("max_delay_sec", 0.0))
    if mx > 0:
        await asyncio.sleep(random.uniform(mn, mx))


def parse_proxy(proxy_url: str) -> Dict[str, str]:
    """
    Playwright proxy dict: {"server": "...", "username": "...", "password": "..."}
    Supported:
      - http://user:pass@host:port
      - http://host:port
    """
    from urllib.parse import urlparse
    u = urlparse(proxy_url)
    if not u.scheme or not u.hostname or not u.port:
        raise ValueError(f"Bad proxy format: {proxy_url}")
    out = {"server": f"{u.scheme}://{u.hostname}:{u.port}"}
    if u.username:
        out["username"] = u.username
    if u.password:
        out["password"] = u.password
    return out


# -----------------------------
# Result model
# -----------------------------

@dataclass
class LoadResult:
    url: str
    final_url: str
    timestamp_utc: str

    ok: bool
    status: str  #OK/FAIL/BLOCK_SUSPECTED
    http_status: Optional[int]
    load_ms: int

    title: Optional[str]
    content_length: int

    attempt: int
    proxy: Optional[str]
    error: Optional[str]
    block_reason: Optional[str]

    saved_html_path: Optional[str]
    saved_screenshot_path: Optional[str]


# -----------------------------
# Browser pool (by proxy)
# -----------------------------

class BrowserPool:
    """
    держим браузеры по прокси, чтобы не запускать новый Chromium на каждый URL.
    """
    def __init__(self, playwright, headless: bool):
        self.playwright = playwright
        self.headless = headless
        self._browsers: Dict[str, List[Browser]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    async def get_browser(self, proxy_url: Optional[str]) -> Browser:
        key = proxy_url or "__direct__"
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            if key not in self._browsers:
                self._browsers[key] = []

            if self._browsers[key]:
                return self._browsers[key].pop()

            launch_args: Dict[str, Any] = {"headless": self.headless}
            if proxy_url:
                launch_args["proxy"] = parse_proxy(proxy_url)

            browser = await self.playwright.chromium.launch(**launch_args)
            return browser

    async def release_browser(self, proxy_url: Optional[str], browser: Browser) -> None:
        key = proxy_url or "__direct__"
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()

        async with self._locks[key]:
            self._browsers.setdefault(key, []).append(browser)

    async def close_all(self) -> None:
        for key, lst in self._browsers.items():
            for b in lst:
                try:
                    await b.close()
                except Exception:
                    pass
        self._browsers.clear()


# -----------------------------
# Loading logic
# -----------------------------

async def new_context(browser: Browser) -> BrowserContext:
    return await browser.new_context(
        viewport={"width": 1366, "height": 768},
        extra_http_headers=default_headers(),
        locale="en-US",
    )


async def fetch_once(
    cfg: Dict[str, Any],
    pool: BrowserPool,
    url: str,
    attempt: int,
    out_dir: Path,
    proxy_url: Optional[str],
) -> LoadResult:
    t0 = time.time()
    ts = utc_iso()

    html_path = None
    shot_path = None

    http_status: Optional[int] = None
    final_url = url
    title: Optional[str] = None
    error: Optional[str] = None
    block_reason: Optional[str] = None

    nav_timeout = int(cfg.get("nav_timeout_ms", 45000))
    total_timeout = int(cfg.get("total_timeout_ms", 60000))

    browser = await pool.get_browser(proxy_url)
    context = None
    page: Optional[Page] = None

    try:
        context = await new_context(browser)
        page = await context.new_page()

        async def _do():
            nonlocal http_status, final_url, title, block_reason, html_path, shot_path

            resp: Optional[Response] = await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=nav_timeout,
            )
            if resp is not None:
                http_status = resp.status

            await asyncio.sleep(0.4)

            final_url = page.url
            title = await page.title()
            html = await page.content()

            suspected, why = looks_like_block_or_challenge(html, final_url)
            if suspected:
                block_reason = why

            if (not resp) or (http_status and http_status >= 400) or suspected:
                safe_id = slugify(url)
                if cfg.get("save_html_on_fail", True):
                    html_path = str(out_dir / "html" / f"{safe_id}__a{attempt}.html")
                    with open(html_path, "w", encoding="utf-8") as f:
                        f.write(html)
                if cfg.get("save_screenshot_on_fail", True):
                    shot_path = str(out_dir / "screens" / f"{safe_id}__a{attempt}.png")
                    try:
                        await page.screenshot(path=shot_path, full_page=True)
                    except Exception:
                        shot_path = None

            ok = (not suspected) and (http_status is None or http_status < 400)
            status = "OK" if ok else ("BLOCK_SUSPECTED" if suspected else "FAIL")
            return html, ok, status

        try:
            html, ok, status = await asyncio.wait_for(_do(), timeout=total_timeout / 1000.0)
        except asyncio.TimeoutError:
            ok = False
            status = "FAIL"
            error = f"TOTAL_TIMEOUT_{total_timeout}ms"
            html = ""
        except PWTimeoutError as e:
            ok = False
            status = "FAIL"
            error = f"NAV_TIMEOUT_{nav_timeout}ms: {str(e)}"
            html = ""
        except Exception as e:
            ok = False
            status = "FAIL"
            error = f"EXCEPTION: {type(e).__name__}: {e}"
            html = ""

        load_ms = int((time.time() - t0) * 1000)
        return LoadResult(
            url=url,
            final_url=final_url,
            timestamp_utc=ts,
            ok=ok,
            status=status,
            http_status=http_status,
            load_ms=load_ms,
            title=title,
            content_length=len(html),
            attempt=attempt,
            proxy=proxy_url,
            error=error,
            block_reason=block_reason,
            saved_html_path=html_path,
            saved_screenshot_path=shot_path,
        )

    finally:
        try:
            if page is not None:
                await page.close()
        except Exception:
            pass
        try:
            if context is not None:
                await context.close()
        except Exception:
            pass
        await pool.release_browser(proxy_url, browser)


async def fetch_with_retries(
    cfg: Dict[str, Any],
    pool: BrowserPool,
    url: str,
    out_dir: Path,
    _proxy_url_unused: Optional[str] = None,
) -> LoadResult:
    max_retries = int(cfg.get("max_retries", 3))
    base = float(cfg.get("retry_backoff_base_sec", 2.0))
    jitter = float(cfg.get("retry_jitter_sec", 0.7))

    last: Optional[LoadResult] = None
    for attempt in range(1, max_retries + 1):
        await polite_delay(cfg)

        proxy_url = choose_proxy_for_task(cfg)

        res = await fetch_once(cfg, pool, url, attempt, out_dir, proxy_url)
        last = res
        if res.ok:
            return res

        sleep_s = base * (2 ** (attempt - 1)) + random.uniform(0, jitter)
        await asyncio.sleep(min(sleep_s, 30.0))

    return last


def choose_proxy_for_task(cfg: Dict[str, Any]) -> Optional[str]:
    proxies_cfg = cfg.get("proxies", {}) or {}
    if not proxies_cfg.get("enabled", False):
        return None
    lst = proxies_cfg.get("list", []) or []
    if not lst:
        return None
    return random.choice(lst) if proxies_cfg.get("random_proxy_per_task", True) else lst[0]


# -----------------------------
# Reporting
# -----------------------------

def write_csv(path: Path, rows: List[LoadResult]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))


def write_jsonl(path: Path, rows: List[LoadResult]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


# -----------------------------
# Main
# -----------------------------

async def main():
    cfg = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
    out_dir = Path(cfg.get("out_dir", "out"))
    ensure_dir(out_dir / "html")
    ensure_dir(out_dir / "screens")

    urls = load_urls("urls.txt")
    limit = int(cfg.get("limit_urls", 0))
    if limit > 0:
        urls = urls[:limit]

    results: List[LoadResult] = []

    async with async_playwright() as p:
        pool = BrowserPool(p, headless=bool(cfg.get("headless", True)))
        sem = asyncio.Semaphore(int(cfg.get("concurrency", 4)))

        async def worker(u: str) -> None:
            async with sem:
                proxy = choose_proxy_for_task(cfg)
                res = await fetch_with_retries(cfg, pool, u, out_dir, proxy)
                results.append(res)
                print(f"[{res.status}] {u} ({res.load_ms}ms) proxy={res.proxy}")

        await asyncio.gather(*(worker(u) for u in urls))
        await pool.close_all()

    results.sort(key=lambda r: (r.status != "OK", r.url))

    write_csv(out_dir / "report.csv", results)
    write_jsonl(out_dir / "report.jsonl", results)

    ok = sum(1 for r in results if r.ok)
    total = len(results)
    blocked = sum(1 for r in results if r.status == "BLOCK_SUSPECTED")
    failed = sum(1 for r in results if r.status == "FAIL")
    print(f"\nDone. OK={ok}/{total}, BLOCK_SUSPECTED={blocked}, FAIL={failed}")
    print(f"Reports: {out_dir / 'report.csv'} and {out_dir / 'report.jsonl'}")


if __name__ == "__main__":
    asyncio.run(main())