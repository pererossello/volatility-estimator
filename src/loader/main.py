from datetime import datetime, timezone

import numpy as np
import yfinance as yf
import pandas as pd


def get_option_chain(ticker="AAPL", clean=True):
 
    t = yf.Ticker(ticker)   # object with stock info and methods 

    # Spot price of the stock
    spot = t.fast_info["last_price"]

    expiries = t.options # list of expiry dates for options
    if not expiries:
        chain = pd.DataFrame()

    rows, asof = [], datetime.now(timezone.utc)

    for exp in expiries:
        try:
            oc = t.option_chain(exp)
        except Exception:
            continue

        for side, df in (("call", oc.calls), ("put", oc.puts)):
            if df is None or df.empty: 
                continue
            tmp = df.copy()
            tmp["otype"] = side
            tmp["expiry"] = pd.Timestamp(exp, tz="UTC")
            rows.append(tmp)

        if not rows:
            chain = pd.DataFrame()   

    chain = pd.concat(rows, ignore_index=True)
    chain = chain.rename(columns={"strike": "K"})

    chain['spot'] = spot
    chain["moneyness"] = chain["K"] / spot
    chain["mid"] = (chain["bid"] + chain["ask"]) / 2.0
    chain["spread"] = chain["ask"] - chain["bid"]
    chain["asof"] = pd.Timestamp(asof)
    chain["T"] = (chain["expiry"] - chain["asof"]).dt.total_seconds() / (365.0 * 24 * 3600)

    if clean:
        print('Cleaning...')
        # Keep tiny premiums so OTM calls survive; just avoid total dust
        chain = chain[chain["mid"] > 0.05]
        # No same-day expiry
        chain = chain[chain["T"] > 1/365]

    print("Rows:", len(chain))

    return chain

