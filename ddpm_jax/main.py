import typing as tp
from functools import partial
from pickletools import optimize

import datasets
import jax
import jax.numpy as jnp
import jax_metrics as jm
import matplotlib.pyplot as plt
import numpy as np
import optax
import tensorflow as tf
import treex as tx
import typer
from einop import einop
from elegy.model.model_full import Model as Trainer
from elegy.modules.core_module import CoreModule
from elegy.modules.flax_module import ModuleState
from tqdm import tqdm

from ddpm_jax.helper_plot import hdr_plot_style, make_animation, plot_gradients
from ddpm_jax.models import EMA, ArrayFn, GaussianDiffusion, Resize, UNet

hdr_plot_style()

DEVICE_0 = jax.devices()[0]

Model = ModuleState[UNet]
DM = tp.TypeVar("DM", bound="DiffusionModel")
Logs = tp.Dict[str, jnp.ndarray]


def to_uint8_image(x: np.ndarray) -> np.ndarray:
    x = (x / 2.0 + 0.5) * 255
    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def get_data(
    dataset_params: tp.Tuple[str, ...],
    dataset_type: str,
    batch_size: int,
    image_shape: tp.Tuple[int, int],
    channels: int,
    timesteps: int,
):
    assert dataset_type in {"image", "bytes"}

    hfds = datasets.load.load_dataset(*dataset_params, split="train", cache_dir="data")

    img_key = [key for key in hfds.features.keys() if key.startswith("im")][0]

    if dataset_type == "image":
        remove_columns = [key for key in hfds.features.keys() if key != "img"]

        def map_fn(sample):
            img = np.asarray(sample[img_key])

            if len(img.shape) == 2:
                img = img[..., None]

            return {"img": img}

        hfds = hfds.map(map_fn, remove_columns=remove_columns)
        ds = tf.data.Dataset.from_generator(
            lambda: hfds,
            output_signature={
                "img": tf.TensorSpec(shape=(None, None, channels), dtype=tf.uint8),
            },
        )
    else:
        ds = tf.data.Dataset.from_generator(
            lambda: hfds,
            output_signature={
                "img_bytes": tf.TensorSpec(shape=(), dtype=tf.string),
            },
        )

    def process_fn(sample):
        x: tf.Tensor

        if dataset_type == "image":
            x = sample["img"]
        else:
            x = tf.image.decode_png(sample["img_bytes"], channels=channels)

        # resize the image
        x = tf.image.resize(x, image_shape, antialias=True)

        # normalize -1 to 1
        x = x / 255.0
        x = (x - 0.5) * 2.0

        t = tf.random.uniform(shape=(), minval=0, maxval=timesteps, dtype=tf.int32)

        return x, t

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


def plot_data(
    key: jnp.ndarray,
    X: np.ndarray,
    timesteps: int,
    beta: np.ndarray,
    diffusion: GaussianDiffusion,
):
    x_seq = diffusion.forward_sample(
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


class DiffusionModel(CoreModule):
    # static
    image_shape: tp.Tuple[int, ...]
    n_channels: int
    timesteps: int
    viz: bool

    # nodes
    key: tp.Optional[jnp.ndarray] = tx.node()
    model: Model = tx.node()
    optimizer: tx.Optimizer = tx.node()
    diffusion: GaussianDiffusion = tx.node()
    metrics: jm.LossesAndMetrics = tx.node()
    ema: EMA = tx.node()

    def __init__(
        self,
        *,
        beta: np.ndarray,
        timesteps: int = 1000,
        dims: int = 32,
        dim_mults: tp.Sequence[int] = (1, 2, 4, 8),
        image_shape: tp.Sequence[int] = (64, 64),
        use_gradient_clipping: bool = True,
        loss_type: str = "mse",
        train_lr: float = 3e-3,
        n_channels: int = 3,
        ema_decay: float = 0.9,
        viz: bool = False,
    ):
        self.key = None
        self.model = Model(
            UNet(
                dim=dims,
                dim_mults=tuple(dim_mults),
                channels=n_channels,
            )
        )
        self.optimizer = tx.Optimizer(
            optax.chain(
                *([optax.clip_by_global_norm(1.0)] if use_gradient_clipping else []),
                optax.adamw(train_lr),
            )
        )
        self.diffusion = GaussianDiffusion(beta)
        self.metrics = jm.LossesAndMetrics(
            losses=jm.losses.MeanSquaredError()
            if loss_type == "mse"
            else jm.losses.MeanAbsoluteError()
        )
        self.ema = EMA(ema_decay)

        self.image_shape = tuple(image_shape)
        self.n_channels = n_channels
        self.timesteps = timesteps
        self.viz = viz

    # @partial(jax.jit, device=DEVICE_0)
    @tx.toplevel_mutable
    def init_step(
        self, key: jnp.ndarray, batch: tp.Tuple[jnp.ndarray, jnp.ndarray]
    ) -> "DiffusionModel":
        print("INIT_STEP")
        x, t = batch

        if self.viz:
            plot_data(
                key,
                x,
                self.timesteps,
                self.diffusion.beta,
                self.diffusion,
            )
            plt.show()

        model_key, self.key = jax.random.split(key)
        self.model = self.model.init(model_key, x, t)
        self.optimizer = self.optimizer.init(self.model["params"])
        self.ema = self.ema.init(self.model["params"])
        return self

    @partial(jax.jit, device=DEVICE_0)
    @tx.toplevel_mutable
    def reset_step(self: "DiffusionModel") -> "DiffusionModel":
        self.metrics = self.metrics.reset()
        return self

    @partial(jax.jit, device=DEVICE_0)
    @tx.toplevel_mutable
    def noise_estimation_loss(
        self: "DiffusionModel",
        params: tp.Optional[tp.Any],
        key: jnp.ndarray,
        x: jnp.ndarray,
        t: jnp.ndarray,
    ) -> tp.Tuple[jnp.ndarray, "DiffusionModel"]:
        noise_pred: jnp.ndarray
        key_sample, key_model = jax.random.split(key, 2)

        if params is not None:
            self.model = self.model.update(params=params)

        # model input
        xt, noise = self.diffusion.forward_sample(key_sample, x, t)

        noise_pred, self.model = self.model.apply(key_model, xt, t)

        loss, self.metrics = self.metrics.loss_and_update(
            preds=noise_pred, target=noise
        )

        return loss, self

    @partial(jax.jit, device=DEVICE_0)
    @tx.toplevel_mutable
    def train_step(
        self: "DiffusionModel",
        batch: tp.Tuple[jnp.ndarray, jnp.ndarray],
        batch_idx: int,
        epoch_idx: int,
    ) -> tp.Tuple[Logs, "DiffusionModel"]:
        print("TRAIN_STEP")
        x, t = batch

        assert self.key is not None

        params = self.model["params"]
        loss_key, self.key = jax.random.split(self.key)

        grads, self = jax.grad(self.noise_estimation_loss, has_aux=True)(
            params, loss_key, x, t
        )

        params, self.optimizer = self.optimizer.update(grads, params)
        params, self.ema = self.ema.update(params)
        self.model = self.model.update(params=params)

        logs = self.metrics.compute_logs()

        return logs, self

    def on_epoch_end(self: DM, epoch: int, logs: tp.Optional[Logs] = None) -> DM:

        if self.viz:
            assert self.key is not None

            print("Generating Sample...")
            x_seq = self.diffusion.reverse_sample_loop(
                self.key, self.model, (1, *self.image_shape, self.n_channels)
            )
            x_seq = to_uint8_image(x_seq)
            print("Plotting...")

            plot_trajectory(x_seq[::-1][:, 0], n_plots=7, reverse=True)

            plt.show()

        return self


def main(
    step_per_epoch: int = 500,
    total_steps: int = 100_000,
    timesteps: int = 1_000,
    batch_size: int = 32,
    viz: bool = True,
    schedule: str = "squared",
    tpu: bool = False,
    image_shape: tp.List[int] = (64, 64),
    n_channels: int = 3,
    dataset_params: tp.List[str] = ("cgarciae/cartoonset", "10k"),
    dataset_type: str = "bytes",
    dims: int = 32,
    ema_decay: float = 0.9,
    train_lr: float = 3e-3,
    loss_type: str = "mse",
    use_gradient_clipping: bool = True,
    dim_mults: tp.List[int] = (1, 2, 4, 8),
):
    epochs = total_steps // step_per_epoch

    if tpu:
        from jax.tools import colab_tpu

        colab_tpu.setup_tpu()

    ds = get_data(
        dataset_params=tuple(dataset_params),
        dataset_type=dataset_type,
        batch_size=batch_size,
        image_shape=tuple(image_shape),
        channels=n_channels,
        timesteps=timesteps,
    )

    beta = make_beta_schedule(
        schedule=schedule, n_timesteps=timesteps, start=1e-5, end=1e-2
    )

    module = DiffusionModel(
        beta=beta,
        timesteps=timesteps,
        dims=dims,
        dim_mults=dim_mults,
        image_shape=tuple(image_shape),
        use_gradient_clipping=use_gradient_clipping,
        loss_type=loss_type,
        train_lr=train_lr,
        n_channels=n_channels,
        ema_decay=ema_decay,
        viz=viz,
    )

    trainer = Trainer(module)

    trainer.fit(
        ds,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=step_per_epoch,
    )

    return locals()


if __name__ == "__main__":
    typer.run(main)
