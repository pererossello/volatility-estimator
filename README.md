# Volatility Estimator (Work In Progress)

Experimental toy project exploring implied ("intrinsic") volatility estimation for European vanilla options using:

* The Black–Scholes pricing model implemented in **JAX**
* **Automatic differentiation (autodiff)** to obtain Vega (gradient of price wrt volatility) instead of deriving / coding it manually
* A simple **Newton–Raphson root finder** to back out the implied volatility from a target market option price

> Status: Early WIP. Interfaces and folder structure will likely change. Expect rough edges, missing validation, and limited error handling.

---

## Why JAX?
Traditionally, implied volatility solvers either:
1. Hard‑code the analytic Vega expression, or
2. Use finite differences (slower / noisier), or
3. Rely on library greeks.

Here we let **JAX autodiff** compute the derivative of the pricing function w.r.t. `sigma` directly:

```python
loss_grad = jax.grad(loss, argnums=4)  # derivative wrt sigma_guess
```

Benefits:
* Fewer manual formulas to mis‑implement
* Easy extensibility when the pricing model changes
* 64‑bit precision enabled for numerical stability (`jax_enable_x64`)

---

## Repository Structure

```
src/
	model/main.py        # black_scholes + Newton-Raphson implied vol solver
	loader/main.py       # yfinance option chain retrieval & light cleaning
notebooks/
	infer_sigma.ipynb    # Minimal demonstration of pricing + inversion
```

---

## Quick Start

### 1. Create & activate a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install dependencies
```bash
pip install jax jaxlib yfinance pandas numpy matplotlib
```

> Note: On GPU/TPU you may wish to install a platform‑specific `jaxlib` wheel. See the official JAX installation guide if needed.

### 3. Run the demonstration notebook
Open `notebooks/infer_sigma.ipynb` in VS Code / Jupyter and execute the cells. It will:
1. Define a synthetic option scenario
2. Compute the theoretical Black–Scholes price with a chosen "true" volatility
3. Invert that price via Newton–Raphson to recover the implied volatility using autodiff for the gradient

### 4. Minimal pure-Python example
```python
from src.model.main import black_scholes, NR_for_sigma

S, K, T, r, q = 100, 110, 1.0, 0.05, 0.0
true_sigma = 0.20
otype = "call"

market_price = black_scholes(S, K, T, r, true_sigma, q, otype)
implied = NR_for_sigma(S, K, T, r, market_price, sigma_guess=0.30, q=q, otype=otype, tol=1e-7)

print("Market price:", market_price)
print("Recovered implied volatility:", implied)
```

---

## Option Chain Loader
`src/loader/main.py` contains a thin wrapper around `yfinance` to pull the full option chain for a given equity ticker, compute basic derived columns (mid price, moneyness, time to expiry), and filter illiquid / immediate‑expiry entries. This will later be integrated with the implied volatility solver to build a surface.

---

## Numerical Method (Newton–Raphson)
The implied volatility solves for `sigma` such that:

```
BlackScholes(S, K, T, r, sigma, q, otype) = market_price
```

We define a loss `f(sigma) = model_price - market_price` and iterate:

```
sigma_{n+1} = sigma_n - f(sigma_n) / f'(sigma_n)
```

Where `f'(sigma_n)` (Vega) is obtained automatically via JAX. Convergence is typically fast provided the initial guess is reasonable and the option is not extremely deep ITM/OTM with very short maturity.

### Current Simplifications
* No explicit guard for extremely low Vega (could cause large steps)
* No bracketing / fallback method (e.g., bisection or Brent) if Newton diverges
* Assumes continuous compounding and constant rates/dividends

---

## References
* Black & Scholes (1973) – The Pricing of Options and Corporate Liabilities
* JAX Documentation – https://github.com/google/jax
* Option pricing lecture notes / standard texts (Hull, etc.)

---

## Disclaimer
This repository is for learning and experimentation. Not investment advice. Use at your own risk.

