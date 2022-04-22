import dataclasses
import typing as tp
from functools import partial
from inspect import isfunction

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import treeo as to
from einop import einop
from elegy.modules.flax_module import ModuleState

A = tp.TypeVar("A")
M = tp.TypeVar("M", bound="nn.Module")
ArrayFn = tp.Callable[[jnp.ndarray], jnp.ndarray]


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


def conv_padding(*args: int) -> tp.List[tp.Tuple[int, int]]:
    return [(p, p) for p in args]


@dataclasses.dataclass
class Residual(nn.Module):
    fn: nn.Module

    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


@dataclasses.dataclass
class SinusoidalEmb(nn.Module):
    dim: int

    def __call__(self, time: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


def Upsample(dim: int):
    return nn.ConvTranspose(dim, [4, 4], strides=[2, 2], padding=conv_padding(2, 2))


def Downsample(dim: int):
    return nn.Conv(dim, [4, 4], strides=[2, 2], padding=conv_padding(1, 1))


@dataclasses.dataclass
class PreNorm(nn.Module):
    fn: ArrayFn

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm()(x)
        return self.fn(x)


# building block modules


@dataclasses.dataclass
class Sequential(nn.Module):
    modules: tp.Tuple[ArrayFn, ...]

    @classmethod
    def new(cls, *modules):
        return cls(modules)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for module in self.modules:
            x = module(x)
        return x


@dataclasses.dataclass
class Identity(nn.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


@dataclasses.dataclass
class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    dim: int
    dim_out: int
    time_emb_dim: tp.Optional[int] = None
    mult: int = 2
    norm: bool = True

    def setup(self):
        self.mlp = (
            Sequential.new(nn.gelu, nn.Dense(self.dim))
            if self.time_emb_dim is not None
            else None
        )

        self.ds_conv = nn.Conv(
            self.dim, [7, 7], padding=conv_padding(3, 3), feature_group_count=self.dim
        )

        self.net = Sequential.new(
            nn.LayerNorm() if self.norm else Identity(),
            nn.Conv(self.dim_out * self.mult, [3, 3], padding=conv_padding(1, 1)),
            nn.gelu,
            nn.Conv(self.dim_out, [3, 3], padding=conv_padding(1, 1)),
        )

        self.res_conv = (
            nn.Conv(self.dim_out, [1, 1], padding=conv_padding(0, 0))
            if self.dim != self.dim_out
            else Identity()
        )

    def __call__(self, x: jnp.ndarray, time_emb: tp.Optional[jnp.ndarray] = None):
        h = self.ds_conv(x)

        if self.mlp is not None:
            assert time_emb is not None, "time emb must be passed in"
            condition = self.mlp(time_emb)
            h = h + einop(condition, "b c -> b 1 1 c")

        h = self.net(h)
        return h + self.res_conv(x)


@dataclasses.dataclass
class LinearAttention(nn.Module):
    dim: int
    heads: int = 4
    dim_head: int = 32

    def setup(self):
        hidden_dim = self.dim_head * self.heads
        self.to_qkv = nn.Conv(hidden_dim * 3, [1, 1], use_bias=False)
        self.to_out = nn.Conv(self.dim, [1, 1])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, h, w, c = x.shape
        qkv = jnp.split(self.to_qkv(x), 3, axis=-1)
        q, k, v = map(
            lambda t: einop(t, "b x y (h c) -> b (x y) h c", h=self.heads),
            qkv,
        )

        scale = self.dim_head**-0.5
        q = q * scale

        k = nn.softmax(k, axis=-1)
        context = jnp.einsum("b n h d , b n h e -> b h d e", k, v)

        out = jnp.einsum("b h d e, b n h d -> b n h e ", context, q)
        out = einop(out, "b (x y) h c -> b x y (h c)", h=self.heads, x=h, y=w)
        return self.to_out(out)


def Resize(
    sample_shape: tp.Sequence[int],
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
    antialias: bool = True,
    precision=jax.lax.Precision.HIGHEST,
) -> ArrayFn:
    def _resize(x: jnp.ndarray) -> jnp.ndarray:
        batch_dims = len(x.shape) - len(sample_shape) - 1  # 1 = channel dim
        shape = (*x.shape[:batch_dims], *sample_shape, x.shape[-1])
        output = jax.image.resize(
            x,
            shape=shape,
            method=method,
            antialias=antialias,
            precision=precision,
        )

        return output

    return _resize


@dataclasses.dataclass
class EMA(tp.Generic[A], to.Tree, to.Immutable):
    mu: float = to.static(0.999)
    params: tp.Optional[A] = to.node(None)

    def init(self, params: A) -> "EMA":
        return self.replace(params=params)

    def update(self, new_params: A) -> tp.Tuple[A, "EMA"]:
        if self.params is None:
            raise ValueError("params must be initialized")

        params = jax.tree_map(self._ema, self.params, new_params)
        ema = self.replace(params=params)
        return params, ema

    def _ema(self, params, new_params):
        return self.mu * params + (1.0 - self.mu) * new_params


@dataclasses.dataclass
class UNet(nn.Module):
    dim: int
    out_dim: tp.Optional[int] = None
    dim_mults: tp.Tuple[int, ...] = (1, 2, 4, 8)
    channels: int = 3
    with_time_emb: bool = True

    def setup(self):
        dims = [self.channels, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if self.with_time_emb:
            time_dim = self.dim
            self.time_mlp = Sequential.new(
                SinusoidalEmb(self.dim),
                nn.Dense(self.dim * 4),
                nn.gelu,
                nn.Dense(self.dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        downs = []
        ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            downs.append(
                [
                    ConvNextBlock(
                        dim_in, dim_out, time_emb_dim=time_dim, norm=ind != 0
                    ),
                    ConvNextBlock(dim_out, dim_out, time_emb_dim=time_dim),
                    Residual(PreNorm(LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else Identity(),
                ]
            )

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(LinearAttention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            ups.append(
                [
                    ConvNextBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                    ConvNextBlock(dim_in, dim_in, time_emb_dim=time_dim),
                    Residual(PreNorm(LinearAttention(dim_in))),
                    Upsample(dim_in) if not is_last else Identity(),
                ]
            )

        self.downs = downs
        self.ups = ups

        out_dim = default(self.out_dim, self.channels)
        self.final_conv = Sequential.new(
            ConvNextBlock(self.dim, self.dim),
            nn.Conv(out_dim, [1, 1]),
        )

    def __call__(self, x: jnp.ndarray, time: jnp.ndarray) -> jnp.ndarray:
        t = self.time_mlp(time) if self.time_mlp is not None else None

        hs = []

        for convnext, convnext2, attn, downsample in self.downs:
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            hs.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for (convnext, convnext2, attn, upsample), h in zip(self.ups, reversed(hs)):
            x = jnp.concatenate([x, h], axis=-1)
            x = convnext(x, t)
            x = convnext2(x, t)
            x = attn(x)
            x = upsample(x)
            x

        return self.final_conv(x)


class GaussianDiffusion(to.Tree, to.Immutable):
    beta: jnp.ndarray = to.node()
    alpha: jnp.ndarray = to.node()
    alpha_prod: jnp.ndarray = to.node()

    def __init__(self, beta: np.ndarray):
        self.beta = jnp.asarray(beta)
        self.alpha = 1.0 - self.beta
        self.alpha_prod = jnp.cumprod(self.alpha)

    def forward_sample(
        self, key: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray
    ) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
        def _sample_fn(key: jnp.ndarray, x: jnp.ndarray, t: jnp.ndarray):
            alpha_prod_t: jnp.ndarray = self.alpha_prod[t]
            assert alpha_prod_t.shape == ()
            noise = jax.random.normal(key, shape=x.shape)
            xt = jnp.sqrt(alpha_prod_t) * x + jnp.sqrt(1.0 - alpha_prod_t) * noise

            return xt, noise

        key = jax.random.split(key, len(x))
        return jax.vmap(_sample_fn)(key, x, t)

    def reverse_sample(
        self,
        key: jnp.ndarray,
        model: ModuleState[M],
        x: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:

        noise_pred = model.apply(None, x, t)[0]

        beta_t: jnp.ndarray = self.beta[t][:, None, None, None]
        alpha_t: jnp.ndarray = self.alpha[t][:, None, None, None]
        alpha_prod_t: jnp.ndarray = self.alpha_prod[t][:, None, None, None]

        noise = jnp.where(
            t[:, None, None, None] > 0,
            jnp.sqrt(beta_t) * jax.random.normal(self.key(), shape=x.shape),
            jnp.zeros_like(x),
        )

        weighted_noise_pred = beta_t / jnp.sqrt(1.0 - alpha_prod_t) * noise_pred
        x_next = (1.0 / jnp.sqrt(alpha_t)) * (x - weighted_noise_pred) + noise

        return x_next

    def reverse_sample_vmap(
        self,
        key: jnp.ndarray,
        model: ModuleState[M],
        x: jnp.ndarray,
        t: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        NOTE: Here we use vmap to "simplify" the logic, by using per-sample calculations
        variables like `beta_t` are scalars which as a result broadcast for free.
        """

        def _sample_fn(
            key: jnp.ndarray,
            noise_pred: jnp.ndarray,
            x: jnp.ndarray,
            t: jnp.ndarray,
        ):

            beta_t: jnp.ndarray = self.beta[t]
            alpha_t: jnp.ndarray = self.alpha[t]
            alpha_prod_t: jnp.ndarray = self.alpha_prod[t]

            assert t.shape == ()
            assert beta_t.shape == ()

            noise = jax.lax.cond(
                t > 0,
                lambda: jnp.sqrt(beta_t) * jax.random.normal(key, shape=x.shape),
                lambda: jnp.zeros_like(x),
            )

            weighted_noise_pred = beta_t / jnp.sqrt(1.0 - alpha_prod_t) * noise_pred
            x_next = (1.0 / jnp.sqrt(alpha_t)) * (x - weighted_noise_pred) + noise

            return x_next

        noise_pred = model.apply(None, x, t)[0]
        key = jax.random.split(key, len(x))
        return jax.vmap(_sample_fn)(key, noise_pred, x, t)

    @partial(jax.jit, static_argnums=(3,), device=jax.devices()[0])
    def reverse_sample_loop(
        self,
        key: jnp.ndarray,
        model: ModuleState[M],
        sample_shape: tp.Tuple[int, ...],
    ) -> jnp.ndarray:
        def scan_fn(x: jnp.ndarray, keys_ts: tp.Tuple[jnp.ndarray, jnp.ndarray]):
            key, t = keys_ts
            x = self.reverse_sample_vmap(key, model, x, t)
            return x, x

        n_steps = len(self.beta)
        key, normal_key = jax.random.split(key)
        x0 = jax.random.normal(normal_key, shape=sample_shape)

        t = jnp.arange(n_steps)[::-1]
        t = einop(t, "... ->  ... batch", batch=sample_shape[0])
        keys = jax.random.split(key, n_steps)

        return jax.lax.scan(scan_fn, x0, (keys, t))[1]
