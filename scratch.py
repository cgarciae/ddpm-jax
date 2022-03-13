import jax
import jax.numpy as jnp

x = 1.0
t = 0.5


@jax.jit
def f(x, t):
    return jax.lax.cond(t > 0, lambda: x, lambda: -x)


y = f(x, t)

print(y)
