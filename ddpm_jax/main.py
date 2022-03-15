import typing as tp
from functools import partial

import datasets
import flax_tools as ft
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import PIL.Image
import tensorflow as tf
import typer
from einop import einop
from sklearn.datasets import make_swiss_roll
from tqdm import tqdm

from ddpm_jax.helper_plot import hdr_plot_style, make_animation, plot_gradients
from ddpm_jax.models import EMA, GaussianDiffusion, Resize, UNet

hdr_plot_style()

Model = ft.ModuleManager[UNet]


def to_uint8_image(x: np.ndarray) -> np.ndarray:
    x = (x / 2.0 + 0.5) * 255
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def get_data(batch_size: int, image_shape: tp.Tuple[int, int], channels: int):

    hgds = datasets.load.load_dataset("cgarciae/cartoonset", split="train")

    ds = tf.data.Dataset.from_generator(
        lambda: hgds,
        output_signature={
            "img_bytes": tf.TensorSpec(shape=(), dtype=tf.string),
        },
    )

    def process_fn(sample):
        x: tf.Tensor = tf.image.decode_png(sample["img_bytes"], channels=channels)

        # maybe resize the image
        if x.shape != image_shape:
            x = tf.image.resize(x, image_shape, antialias=True)

        # normalize -1 to 1
        x = x / 255.0
        x = (x - 0.5) * 2.0

        return {"img": x}

    ds = ds.map(process_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.repeat()
    ds = ds.shuffle(seed=42, buffer_size=1_000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    ds = ds.as_numpy_iterator()

    return ds


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


@partial(jax.jit, device=jax.devices()[0])
def noise_estimation_loss(
    params: tp.Optional[tp.Any],
    model: "Model",
    difussion_process: GaussianDiffusion,
    key: jnp.ndarray,
    x: jnp.ndarray,
    t: jnp.ndarray,
):
    preds: jnp.ndarray
    key_sample, key_model = jax.random.split(key, 2)

    if params is not None:
        model = model.update(params=params)

    # model input
    xt, noise = difussion_process.forward_sample(key_sample, x, t)

    preds, model = model(key_model, xt, t)

    return jnp.mean((noise - preds) ** 2)


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


@partial(jax.jit, device=jax.devices()[0])
def train_step(
    key: jnp.ndarray,
    model: Model,
    optimizer: ft.Optimizer,
    difussion_process: GaussianDiffusion,
    ema: EMA,
    x: jnp.ndarray,
    t: jnp.ndarray,
):
    print("JITTING")
    params = model["params"]
    grads = jax.grad(noise_estimation_loss)(params, model, difussion_process, key, x, t)
    params, optimizer = optimizer.update(grads, params)
    params, ema = ema.update(params)
    model = model.update(params=params)
    return model, optimizer, difussion_process, ema


@jax.jit
def get_gradient(model: Model, x: jnp.ndarray):
    ts = jnp.arange(model.timesteps // 4, dtype=jnp.int32)

    @jax.vmap
    def f(t):
        t = jnp.zeros(x.shape[0], dtype=jnp.int32) + t
        return -model(x, t)

    return f(ts).mean(axis=0)


def plot_data(
    key: jnp.ndarray,
    X: np.ndarray,
    timesteps: int,
    beta: np.ndarray,
    difussion_process: GaussianDiffusion,
):
    x_seq = difussion_process.forward_sample(
        key,
        einop(X[0], "... -> batch ...", batch=timesteps),
        np.linspace(0, timesteps - 1, timesteps).astype(np.int32),
    )[0]
    x_seq = to_uint8_image(x_seq)

    plot_trajectory(x_seq, n_plots=5, max_timestep=timesteps)

    # plot 8 random mnist images using subplots
    fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        xi = to_uint8_image(X[i])
        ax[i // 4, i % 4].imshow(xi.squeeze(), cmap="gray")
        ax[i // 4, i % 4].set_axis_off()

    plt.figure(figsize=(16, 12))
    plt.scatter(np.arange(timesteps), beta)


def main(
    steps: int = 100_000,
    timesteps: int = 1_000,
    batch_size: int = 32,
    viz: bool = True,
    plot_steps: int = 10_000,
    schedule: str = "squared",
    tpu: bool = False,
    image_shape: tp.List[int] = (64, 64),
    n_channels: int = 3,
):
    if tpu:
        from jax.tools import colab_tpu

        colab_tpu.setup_tpu()

    ds = get_data(batch_size, image_shape, n_channels)
    ds = iter(ds)

    key = ft.Key(42)
    beta = make_beta_schedule(
        schedule=schedule, n_timesteps=timesteps, start=1e-5, end=1e-2
    )

    time_sample = np.random.randint(0, timesteps, size=batch_size)
    x_sample = next(ds)["img"]

    model = ft.ModuleManager.new(UNet(dim=32, channels=n_channels)).init(
        42, x_sample, time_sample
    )
    optimizer = ft.Optimizer(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(3e-3),
        )
    ).init(model["params"])
    difussion_process = GaussianDiffusion.new(beta)
    ema = EMA(model["params"], mu=0.9)

    if viz:
        plot_data(
            key=key,
            X=x_sample,
            timesteps=timesteps,
            beta=beta,
            difussion_process=difussion_process,
        )
        plt.show()

    for i, batch in tqdm(zip(range(steps + 1), ds), total=steps + 1):
        x_batch = batch["img"]
        t_batch = np.random.randint(0, timesteps, size=batch_size)

        key, key_step = jax.random.split(key)
        model, optimizer, difussion_process, ema = train_step(
            key_step, model, optimizer, difussion_process, ema, x_batch, t_batch
        )

        if i % plot_steps == 0:
            # print loss
            print(
                noise_estimation_loss(
                    None, model, difussion_process, key, x_batch, t_batch
                )
            )

            if viz:
                x_seq = difussion_process.reverse_sample_loop(
                    key, model, (1, *image_shape, n_channels)
                )
                x_seq = to_uint8_image(x_seq)
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
