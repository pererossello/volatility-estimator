import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm

jax.config.update("jax_enable_x64", True)



def black_scholes(S, K, T, r, sigma, q=0.0, otype="call"):
    """
    Black-Scholes formula for European call and put options.
    """
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)

    if otype == "call":
        call = S * jnp.exp(-q * T) * jnorm.cdf(d1) - K * jnp.exp(-r * T) * jnorm.cdf(d2)
        return call
    else:
        put = K * jnp.exp(-r * T) * jnorm.cdf(-d2) - S * jnp.exp(-q * T) * jnorm.cdf(
            -d1
        )
        return put

def loss(S, K, T, r, sigma_guess, market_price, q=0., otype="call"):
    model_price = black_scholes(S, K, T, r, sigma_guess, q, otype)
    return model_price - market_price
loss_grad = jax.grad(loss, argnums=4)

def NR_for_sigma(
    S,
    K,
    T,
    r,
    market_price,
    sigma_guess=0.3,
    q=0.0,
    otype="call",
    tol=1e-7,
    max_iter=100,
    verbose=False,
):
    """
    Newton-Raphson method to find the implied volatility.
    """
    sigma = sigma_guess
    for i in range(max_iter):
        loss_value = loss(S, K, T, r, sigma, market_price, q, otype)

        if verbose:
            print(f"Iter {i}: sigma={sigma}, loss={loss_value}")

        if jnp.abs(loss_value) < tol:
            break

        if jnp.isnan(loss_value) or jnp.isinf(loss_value):
            raise ValueError(f"Loss function returned NaN or Inf at iteration {i}")

        grad_value = loss_grad(S, K, T, r, sigma, market_price, q, otype)
        sigma -= loss_value / grad_value

    return sigma 
