#!/usr/bin/env python3
import os
import json
from datetime import date, timedelta, datetime, timezone

import pandas as pd
import yfinance as yf

TICKERS = os.getenv("TICKERS", "XLK,XLY,XLP,XLF,XLV,XLI,XLB,XLE,XLRE,XLU,XLC,SPY").split(",")
YEARS = int(os.getenv("YEARS", "5"))
OUT_DIR = os.getenv("OUT_DIR", "dist")

def fetch_prices(tickers, start, end):
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    frames = []
    for t in tickers:
        sub = df[t].reset_index().rename(columns=str.lower)
        if sub.empty:
            continue
        sub["ticker"] = t
        sub = sub[["date", "ticker", "open", "high", "low", "close", "adj close", "volume"]]
        sub = sub.rename(columns={"adj close": "adj_close"})
        frames.append(sub)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "ticker"])
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["ingested_at"] = datetime.now(timezone.utc)
    out["source"] = "yfinance"
    return out.sort_values(["ticker", "date"])

def validate(df: pd.DataFrame):
    assert not df.empty, "No rows fetched"
    assert df["date"].max() <= date.today(), "Future dates detected"
    assert df[["open","high","low","close","adj_close","volume"]].isna().sum().sum() == 0, "Nulls in price fields"
    bad_price = df[(df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["high"] < df["open"]) | (df["high"] < df["close"])]
    assert bad_price.empty, "Price bounds violated (low/high vs open/close)"
    assert (df["volume"] >= 0).all(), "Negative volumes"
    assert df.duplicated(subset=["date","ticker"]).sum() == 0, "Duplicate (date,ticker)"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    start = (date.today() - timedelta(days=365*YEARS)).isoformat()
    end = date.today().isoformat()

    df = fetch_prices(TICKERS, start, end)
    validate(df)

    # Write artifacts
    parquet_path = os.path.join(OUT_DIR, "prices_5y.parquet")
    csv_gz_path  = os.path.join(OUT_DIR, "prices_5y.csv.gz")
    meta_path    = os.path.join(OUT_DIR, "metadata.json")

    # Deterministic ordering for stable diffs
    df = df.sort_values(["ticker", "date"])
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_gz_path, index=False, compression="gzip")

    meta = {
        "tickers": TICKERS,
        "rows": int(len(df)),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "years": YEARS,
        "schema": {c: str(t) for c, t in df.dtypes.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {parquet_path}, {csv_gz_path}, {meta_path}")

if __name__ == "__main__":
    main()
