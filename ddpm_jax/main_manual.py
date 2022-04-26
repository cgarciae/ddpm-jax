import enum
import typing as tp
from base64 import b64encode
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
import wandb
from einop import einop
from elegy.model.model_full import Model as Trainer
from elegy.modules.core_module import CoreModule
from elegy.modules.flax_module import ModuleState
from matplotlib import animation
from tqdm import tqdm

from ddpm_jax.helper_plot import hdr_plot_style, make_animation, plot_gradients
from ddpm_jax.models import EMA, ArrayFn, GaussianDiffusion, Resize, UNet

hdr_plot_style()

DEVICE_0 = jax.devices()[0]

print(jax.devices())

Model = ModuleState[UNet]
DM = tp.TypeVar("DM", bound="DiffusionModel")
Logs = tp.Dict[str, jnp.ndarray]


class VizType(enum.Enum):
    LOCAL = enum.auto()
    WANDB = enum.auto()


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
    elif dataset_type == "bytes":
        ds = tf.data.Dataset.from_generator(
            lambda: hfds,
            output_signature={
                "img_bytes": tf.TensorSpec(shape=(), dtype=tf.string),
            },
        )
    else:
        raise ValueError(f"Unknown dataset type {dataset_type}")

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


def format_for_wandb_video(
    xs: np.ndarray, step_size=10, padding: int = 100
):
    xs = np.asarray(xs)
    x_last = xs[-1]

    # apply step size
    xs = xs[::step_size]

    # add end padding
    xpad = einop(x_last, "... -> padding ...", padding=padding)
    xs = np.concatenate([xpad, xs], axis=0)

    # swap axes
    xs = einop(xs, "t h w c -> t c h w")

    return xs


def plot_trajectory(
    xs: np.ndarray,
    n_plots: int = 10,
    max_timestep: tp.Optional[int] = None,
    reverse: bool = False,
    as_animation: bool = True,
    interval: int = 10,
    repeat_delay: int = 1000,
    step_size: int = 10,
):
    xs = xs[::step_size]

    if as_animation:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from uuid import uuid4

        from IPython import get_ipython
        from IPython.display import HTML, Image, Markdown, display

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        imshow_anim = axs[0].imshow(xs[0])
        imshow_static = axs[1].imshow(xs[-1])

        for ax in axs:
            ax.axis("off")

        xpad = einop(xs[-1], "... -> pad ...", pad=len(xs))
        xs = np.concatenate([xs, xpad], axis=0)

        N = len(xs)

        def animate(i):
            imshow_anim.set_array(xs[i])
            return [imshow_anim, imshow_static]

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=lambda: animate(0),
            frames=np.linspace(0, N - 1, N, dtype=int),
            interval=interval,
            repeat_delay=repeat_delay,
            blit=True,
        )

        if get_ipython():
            with TemporaryDirectory() as tmpdir:
                img_name = Path(tmpdir) / f"{uuid4()}.gif"
                anim.save(str(img_name), writer="pillow", fps=60)
                image_bytes = b64encode(img_name.read_bytes()).decode("utf-8")

            display(HTML(f"""<img src='data:image/gif;base64,{image_bytes}'>"""))
            del anim
            plt.close()
        else:
            plt.show()
    else:
        idxs = np.linspace(0, len(xs) - 1, n_plots).astype(np.int32)
        ts = np.linspace(
            0,
            max_timestep - 1 if max_timestep else len(xs) - 1,
            n_plots,
        ).astype(np.int32)

        if reverse:
            idxs = idxs[::-1]
            ts = ts[::-1]

        fig, axs = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3))

        for plot_i, (i, t) in enumerate(zip(idxs, ts)):
            axs[plot_i].imshow(xs[i].squeeze())
            axs[plot_i].set_axis_off()
            axs[plot_i].set_title("$q(\mathbf{x}_{" + str(t) + "})$")


def forward_sample(
    key: jnp.ndarray,
    X: np.ndarray,
    timesteps: int,
    diffusion: GaussianDiffusion,
):
    xs = diffusion.forward_sample(
        key,
        einop(X[0], "... -> batch ...", batch=timesteps),
        np.linspace(0, timesteps - 1, timesteps).astype(np.int32),
    )[0]
    xs = np.asarray(xs)
    xs = to_uint8_image(xs)
    return xs


def plot_data(
    X: np.ndarray,
    xs: np.ndarray,
    timesteps: int,
    beta: np.ndarray,
):

    # plot 8 random mnist images using subplots
    sammples_fig, ax = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        xi = to_uint8_image(X[i])
        ax[i // 4, i % 4].imshow(xi.squeeze(), cmap="gray")
        ax[i // 4, i % 4].set_axis_off()

    beta_fig = plt.figure(figsize=(16, 12))
    plt.scatter(np.arange(timesteps), beta)

    return sammples_fig, beta_fig


class DiffusionModel(CoreModule):
    # static
    image_shape: tp.Tuple[int, ...]
    n_channels: int
    timesteps: int
    viz: tp.Optional[VizType]

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
        loss_type: str = "mae",
        train_lr: float = 3e-3,
        n_channels: int = 3,
        ema_decay: float = 0.9,
        viz: tp.Optional[VizType] = None,
    ):
        if loss_type == "mse":
            loss = jm.losses.MeanSquaredError()
        elif loss_type == "mae":
            loss = jm.losses.MeanAbsoluteError()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
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
        self.metrics = jm.LossesAndMetrics(losses=loss)
        self.ema = EMA(ema_decay)

        self.image_shape = tuple(image_shape)
        self.n_channels = n_channels
        self.timesteps = timesteps
        self.viz = viz

    # @partial(jax.jit) #) #) #, donate_argnums=0)
    @tx.toplevel_mutable
    def init_step(
        self, key: jnp.ndarray, batch: tp.Tuple[jnp.ndarray, jnp.ndarray]
    ) -> "DiffusionModel":
        print("INIT_STEP")
        x, t = batch

        if self.viz is not None:
            xs = forward_sample(key, x, self.timesteps, self.diffusion)

            if self.viz == VizType.LOCAL:
                plot_data(
                    x,
                    xs,
                    self.timesteps,
                    self.diffusion,
                )
                plot_trajectory(xs, max_timestep=self.timesteps)
                plt.show()
            elif self.viz == VizType.WANDB:
                xs = format_for_wandb_video(xs)
                wandb.log(
                    {
                        # "real_samples": samples_fig,
                        # "betas": beta_fig,
                        "forward_diffusion": wandb.Video(xs, fps=60),
                    }
                )
            else:
                raise ValueError(f"Unknown viz type: {self.viz}")

        model_key, self.key = jax.random.split(key)
        self.model = self.model.init(model_key, x, t)
        self.optimizer = self.optimizer.init(self.model["params"])
        self.ema = self.ema.init(self.model["params"])
        return self

    @partial(jax.jit)  # , donate_argnums=0)
    @tx.toplevel_mutable
    def reset_step(self: "DiffusionModel") -> "DiffusionModel":
        self.metrics = self.metrics.reset()
        return self

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

    @partial(jax.jit)  # , donate_argnums=0)
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

    def on_epoch_end(
        self: DM, epoch: int, logs: tp.Optional[tp.Dict[str, tp.Any]] = None
    ) -> DM:

        if self.viz:
            assert self.key is not None
            assert logs is not None

            logs = logs.copy()

            print("Generating Sample...")
            xs = self.diffusion.reverse_sample_loop(
                self.key, self.model, (1, *self.image_shape, self.n_channels)
            )
            xs = xs[:, 0]
            xs = to_uint8_image(xs)
            print("Plotting...")

            if self.viz == VizType.LOCAL:
                plot_trajectory(xs, n_plots=7)
                plt.show()

            elif self.viz == VizType.WANDB:
                # log video to wandb
                xs = format_for_wandb_video(xs)
                logs["reverse_diffusion"] = wandb.Video(xs, fps=60)
                wandb.log(logs)
            else:
                raise ValueError(f"Unknown viz type: {self.viz}")

        return self


def main(
    steps_per_epoch: int = 500,
    total_steps: int = 100_000,
    timesteps: int = 1_000,
    batch_size: int = 32,
    viz: tp.Optional[str] = "wandb",
    schedule: str = "squared",
    tpu: bool = False,
    image_shape: tp.List[int] = (64, 64),
    n_channels: int = 3,
    dataset_params: tp.List[str] = ("cgarciae/cartoonset", "10k"),
    dataset_type: str = "bytes",
    dims: int = 64,
    ema_decay: float = 0.9,
    train_lr: float = 3e-3,
    loss_type: str = "mae",
    use_gradient_clipping: bool = True,
    dim_mults: tp.List[int] = (1, 2, 4, 8),
    verbose: int = 1,
):

    epochs = total_steps // steps_per_epoch

    if tpu:
        from jax.tools import colab_tpu

        colab_tpu.setup_tpu()

    if viz is not None:
        _viz = VizType[viz.upper()]
    else:
        _viz = None

    if _viz == VizType.WANDB:
        wandb.init(project="ddpm-jax")

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
        viz=_viz,
    )

    # trainer = Trainer(module)

    # trainer.fit(
    #     ds,
    #     epochs=epochs,
    #     batch_size=batch_size,
    #     steps_per_epoch=steps_per_epoch,
    #     verbose=verbose,
    # )

    ds_iterator = iter(ds)
    logs = {}

    batch = next(ds_iterator)
    module = module.init_step(jax.random.PRNGKey(0), batch)

    for epoch in range(epochs):
        
        module = module.reset_step()

        for step in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}", unit="batches"):
            batch = next(ds_iterator)

            logs, module = module.train_step(batch, step, epoch)

        module = module.on_epoch_end(epoch, logs)


    return locals()


if __name__ == "__main__":
    typer.run(main)
