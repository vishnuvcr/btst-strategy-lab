from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
import yaml

NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"


def _nse_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
            "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.8",
            "Referer": "https://www.nseindia.com/",
        }
    )
    return s


def load_nse_equity_universe() -> pd.DataFrame:
    last_error = None
    for attempt in range(3):
        try:
            r = _nse_session().get(NSE_EQUITY_URL, timeout=30)
            r.raise_for_status()
            df = pd.read_csv(io.BytesIO(r.content))
            df.columns = [str(c).strip().upper() for c in df.columns]
            if "SYMBOL" not in df.columns:
                raise RuntimeError(f"NSE security master missing SYMBOL column: {df.columns.tolist()}")
            if "SERIES" in df.columns:
                df = df[df["SERIES"].astype(str).str.upper().eq("EQ")]
            df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
            df = df[df["SYMBOL"].str.len().between(1, 30)].drop_duplicates("SYMBOL")
            if df.empty:
                raise RuntimeError("NSE equity universe is empty")
            return df
        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download NSE equity universe: {last_error}")


def _write_yahoo_frame(symbol: str, frame: pd.DataFrame, destination: Path) -> bool:
    if frame is None or frame.empty:
        return False
    x = frame.copy()
    if isinstance(x.columns, pd.MultiIndex):
        x.columns = [c[0] if isinstance(c, tuple) else c for c in x.columns]
    x = x.reset_index()
    x.columns = [str(c).strip().lower().replace(" ", "_") for c in x.columns]
    if "date" not in x.columns:
        x = x.rename(columns={x.columns[0]: "date"})
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(x.columns):
        return False
    x["symbol"] = symbol
    if "adj_close" not in x.columns:
        x["adj_close"] = x["close"]
    cols = ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]
    x = x[cols].dropna(subset=["date", "open", "high", "low", "close"])
    if x.empty:
        return False
    x.to_csv(destination / f"{symbol}.csv", index=False)
    return True


def download(cfg: dict, destination: str = "data/nse_all_daily") -> None:
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=True)
    universe = load_nse_equity_universe()
    max_symbols = int(cfg["data"].get("max_symbols", 5000))
    symbols = universe["SYMBOL"].head(max_symbols).tolist()
    start = cfg["data"].get("start_date", "2015-01-01")
    end = cfg["data"].get("end_date")
    tickers = [f"{s}.NS" for s in symbols]

    used = 0
    failed = []
    batch_size = int(cfg["data"].get("yahoo_batch_size", 50))
    print(f"NSE EQ universe symbols discovered: {len(symbols)}")
    for i in range(0, len(tickers), batch_size):
        batch_symbols = symbols[i : i + batch_size]
        batch_tickers = tickers[i : i + batch_size]
        try:
            data = yf.download(
                batch_tickers,
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            for symbol, ticker in zip(batch_symbols, batch_tickers):
                try:
                    if isinstance(data.columns, pd.MultiIndex):
                        if ticker not in data.columns.get_level_values(0):
                            failed.append(symbol)
                            continue
                        frame = data[ticker]
                    else:
                        frame = data
                    if _write_yahoo_frame(symbol, frame, out):
                        used += 1
                    else:
                        failed.append(symbol)
                except Exception:
                    failed.append(symbol)
        except Exception as exc:
            failed.extend(batch_symbols)
            print(f"Yahoo batch failed for {batch_symbols[0]}..{batch_symbols[-1]}: {exc}")
        print(f"Yahoo progress: {min(i + batch_size, len(symbols))}/{len(symbols)}; usable files={used}")
        time.sleep(0.5)

    if used == 0:
        raise RuntimeError("Yahoo Finance returned no usable NSE daily OHLCV files")
    print(f"NSE-wide daily acquisition complete: universe={len(symbols)}, usable_files={used}, failed={len(failed)}")


def main() -> None:
    with open("config/swing.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    download(cfg)


if __name__ == "__main__":
    main()
