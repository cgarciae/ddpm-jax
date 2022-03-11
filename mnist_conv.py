import typing as tp
from functools import partial

import datasets
import einops
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import treex as tx
import typer
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm

from helper_plot import hdr_plot_style, make_animation, plot_gradients

hdr_plot_style()


class KeySeq(tx.KeySeq):
    def split(self, n: int = 2) -> "KeySeq":
        key = self()
        return self.__class__(jax.random.split(key, n))

    def advance(self) -> "KeySeq":
        key = self()
        return self.__class__(key)


def get_data():

    ds = datasets.load.load_dataset("mnist")
    ds.set_format("numpy")
    x = np.stack(ds["train"]["image"])[..., None]
    x = x.astype(np.float32)

    # normalize -1 to 1
    x = x / 255.0
    x = (x - 0.5) * 2.0

    return x


def make_beta_schedule(schedule="linear", n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = np.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = np.linspace(start**0.5, end**0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = np.linspace(-6, 6, n_timesteps)
        betas = jax.nn.sigmoid(betas) * (end - start) + start
    elif schedule == "squared":
        betas = np.linspace(0, 1, n_timesteps)
        betas = betas**2 * (end - start) + start
    elif schedule == "cubed":
        betas = np.linspace(0, 1, n_timesteps)
        betas = betas**3 * (end - start) + start
    else:
        raise ValueError(f"Unknown schedule {schedule}")
    return np.asarray(betas)


def simple_cond(
    pred: jnp.ndarray, vtrue: jnp.ndarray, vfalse: jnp.ndarray
) -> jnp.ndarray:
    pred = pred.astype(np.float32)
    return pred * vtrue + (1.0 - pred) * vfalse


class Sampler(tx.Treex):
    beta: jnp.ndarray = tx.node()
    alpha: jnp.ndarray = tx.node()
    alpha_prod: jnp.ndarray = tx.node()
    next_key: KeySeq = tx.node()

    def __init__(self, beta: np.ndarray, seed: int = 42) -> None:
        self.beta = jnp.asarray(beta)
        self.alphas = 1.0 - self.beta
        self.alpha_prod = jnp.cumprod(self.alphas)
        self.next_key = KeySeq(seed)

    def forward_sample(
        self, x: jnp.ndarray, t: jnp.ndarray
    ) -> tp.Tuple[jnp.ndarray, jnp.ndarray]:
        def _sample_fn(next_key: KeySeq, x: jnp.ndarray, t: jnp.ndarray):
            alpha_prod_t = self.alpha_prod[t]
            assert alpha_prod_t.shape == ()
            noise = jax.random.normal(next_key(), shape=x.shape)
            xt = jnp.sqrt(alpha_prod_t) * x + jnp.sqrt(1.0 - alpha_prod_t) * noise

            return xt, noise

        return jax.vmap(_sample_fn)(self.next_key.split(len(x)), x, t)

    def reverse_sample(
        self, model: "ConditionalModel", x: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:

        noise_pred = model(x, t)

        beta_t = self.beta[t][:, None]
        alpha_t = self.alphas[t][:, None]
        alpha_prod_t = self.alpha_prod[t][:, None]

        noise = jnp.where(
            t[:, None] > 0,
            jnp.sqrt(beta_t) * jax.random.normal(self.next_key(), shape=x.shape),
            jnp.zeros_like(x),
        )

        weighted_noise_pred = beta_t / jnp.sqrt(1.0 - alpha_prod_t) * noise_pred
        x_tm1 = (1.0 / jnp.sqrt(alpha_t)) * (x - weighted_noise_pred) + noise

        return x_tm1

    def reverse_sample_vmap(
        self, model: "ConditionalModel", x: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        def _sample_fn(
            next_key: KeySeq,
            x: jnp.ndarray,
            t: jnp.ndarray,
        ):
            noise_pred = model(x[None], t[None])[0]

            beta_t = self.beta[t]
            alpha_t = self.alphas[t]
            alpha_prod_t = self.alpha_prod[t]

            assert t.shape == ()
            assert beta_t.shape == ()

            noise = jax.lax.cond(
                t > 0,
                lambda _: jnp.sqrt(beta_t)
                * jax.random.normal(next_key(), shape=x.shape),
                lambda _: jnp.zeros_like(x),
                None,
            )

            weighted_noise_pred = beta_t / jnp.sqrt(1.0 - alpha_prod_t) * noise_pred
            x_tm1 = (1.0 / jnp.sqrt(alpha_t)) * (x - weighted_noise_pred) + noise

            return x_tm1

        return jax.vmap(_sample_fn)(self.next_key.split(len(x)), x, t)

    @partial(jax.jit, static_argnums=(2, 3))
    def reverse_sample_loop(
        self,
        model: "ConditionalModel",
        sample_shape: tp.Tuple[int, ...],
    ) -> jnp.ndarray:
        def scan_fn(state: tp.Tuple[jnp.ndarray, Sampler], t: jnp.ndarray):
            x, sampler = state
            x = sampler.reverse_sample_vmap(model, x, t)
            return (x, sampler), x

        x0 = jax.random.normal(self.next_key(), shape=sample_shape)
        t = jnp.arange(len(self.beta))[::-1]
        t = einops.repeat(t, "... ->  ... batch", batch=sample_shape[0])

        return jax.lax.scan(scan_fn, (x0, self), t)[1]


def noise_estimation_loss(
    model: "ConditionalModel",
    sampler: Sampler,
    x: jnp.ndarray,
    t: jnp.ndarray,
):

    # model input
    xt, noise = sampler.forward_sample(x, t)

    noise_pred = model(xt, t)

    return ((noise - noise_pred) ** 2).mean()


noise_estimation_loss_jit = jax.jit(noise_estimation_loss)


class EMA(tx.Treex):
    params: tx.Module = tx.node()

    def __init__(self, params, mu=0.999):
        self.mu = mu
        self.params = params

    def update(self, new_params):
        self.params = jax.tree_map(self.ema, self.params, new_params)
        return self.params

    def ema(self, params, new_params):
        return self.mu * params + (1.0 - self.mu) * new_params


class ConditionalLinear(tx.Module):
    def __init__(self, num_out, n_steps):
        self.num_out = num_out
        self.conv = tx.Conv(num_out, [5, 5])
        self.embed = tx.Embed(n_steps, num_out)

    def __call__(self, x, t):
        return self.conv(x) + self.embed(t)[:, None, None, :]


class ConditionalModel(tx.Module):
    def __init__(self, timesteps: int):
        self.timesteps = timesteps

    @tx.compact
    def __call__(self, x, t):
        x = ConditionalLinear(32, self.timesteps)(x, t)
        x = jax.nn.softplus(x)

        x = ConditionalLinear(32, self.timesteps)(x, t)
        x = jax.nn.softplus(x)

        x = ConditionalLinear(32, self.timesteps)(x, t)
        x = jax.nn.softplus(x)

        x = ConditionalLinear(32, self.timesteps)(x, t)
        x = jax.nn.softplus(x)

        x = ConditionalLinear(32, self.timesteps)(x, t)
        x = jax.nn.softplus(x)

        return tx.Conv(1, [5, 5])(x)


def plot_trajectory(
    x_seq: np.ndarray,
    n_plots: int = 10,
    max_timestep: tp.Optional[int] = None,
    reverse: bool = False,
):
    fig, axs = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3))
    idxs = np.linspace(0, len(x_seq) - 1, n_plots).astype(np.int32)
    ts = np.linspace(
        0,
        max_timestep - 1 if max_timestep else len(x_seq) - 1,
        n_plots,
    ).astype(np.int32)

    if reverse:
        idxs = idxs[::-1]
        ts = ts[::-1]

    for plot_i, (i, t) in enumerate(zip(idxs, ts)):
        cur_x = x_seq[i]
        axs[plot_i].imshow(cur_x.squeeze(), cmap="gray")
        axs[plot_i].set_axis_off()
        axs[plot_i].set_title("$q(\mathbf{x}_{" + str(t) + "})$")


@jax.jit
def train_step(
    model: ConditionalModel,
    optimizer: tx.Optimizer,
    sampler: Sampler,
    ema: EMA,
    x: jnp.ndarray,
    t: jnp.ndarray,
):
    grads = jax.grad(noise_estimation_loss)(model, sampler, x, t)
    model = optimizer.update(grads, model)
    model = ema.update(model)
    return model, optimizer, sampler, ema


@jax.jit
def get_gradient(model: ConditionalModel, x: jnp.ndarray):
    ts = jnp.arange(model.timesteps // 4, dtype=jnp.int32)

    @jax.vmap
    def f(t):
        t = jnp.zeros(x.shape[0], dtype=jnp.int32) + t
        return -model(x, t)

    return f(ts).mean(axis=0)


def main(
    steps: int = 100_000,
    timesteps: int = 1_000,
    batch_size: int = 128,
    n_samples: int = 8_000,
    viz: bool = True,
    plot_steps: int = 10_000,
    schedule: str = "squared",
    tpu: bool = False,
):
    if tpu:
        import jax.tools.colab_tpu

        jax.tools.colab_tpu.setup_tpu()
    X = get_data()

    # plot 8 random mnist images using subplots
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        ax[i // 4, i % 4].imshow(X[i].squeeze(), cmap="gray")
        ax[i // 4, i % 4].set_axis_off()

    # Noise schedule
    beta = make_beta_schedule(
        schedule=schedule, n_timesteps=timesteps, start=1e-5, end=1e-2
    )
    plt.figure(figsize=(16, 12))
    plt.scatter(np.arange(timesteps), beta)

    sampler = Sampler(beta, seed=42)

    x_seq = sampler.forward_sample(
        einops.repeat(X[0], "... -> batch ...", batch=timesteps),
        np.linspace(0, timesteps - 1, timesteps).astype(np.int32),
    )[0]

    plot_trajectory(x_seq, n_plots=5, max_timestep=timesteps)

    if viz:
        plt.show()

    t_init = np.random.randint(0, timesteps, size=batch_size)
    model = ConditionalModel(timesteps).init(42, (X[:batch_size], t_init))
    print(model.tabulate((X[:batch_size], t_init)))
    optimizer = tx.Optimizer(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(3e-3),
        )
    ).init(model)
    ema = EMA(model, mu=0.9)

    for i in tqdm(range(steps + 1)):
        batch_idxs = np.random.choice(len(X), size=batch_size, replace=False)
        x_batch = X[batch_idxs]
        t = np.random.randint(0, timesteps, size=batch_size)

        model, optimizer, sampler, ema = train_step(
            model, optimizer, sampler, ema, x_batch, t
        )

        if i % plot_steps == 0:
            # print loss
            print(noise_estimation_loss_jit(model, sampler, x_batch, t))

            if viz:
                x_seq = sampler.reverse_sample_loop(model, (1, 28, 28, 1))
                print("Plotting...")

                # print(x_seq[-1])

                plot_trajectory(x_seq[::-1][:, 0], n_plots=7, reverse=True)

                # anim = make_animation(
                #     x_seq,
                #     model=model,
                #     data=X,
                #     forward=get_gradient,
                #     step_size=1,
                #     interval=10,
                #     xlim_scale=0.7,
                #     ylim_scale=0.7,
                # )

                plt.show()

                # anim.save("sample.gif", writer="imagemagick")

    return locals()

if __name__ == "__main__":
    typer.run(main)
