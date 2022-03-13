import flax_tools as ft
import jax
import numpy as np
from ddpm_jax.models import UNet
import functools


def eval_shape(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        return jax.eval_shape(lambda: method(self, *args, **kwargs))

    return wrapper


class TestUNet:
    @eval_shape
    def test_unet(self):
        model = ft.ModuleManager.new(UNet(dim=64, dim_mults=(1, 2, 4, 8)))

        x = np.random.uniform(size=(2, 64, 64, 3))
        t = np.random.randint(0, 100, size=(2,))

        model = model.init(42, x, t)

        y, model = model(42, x, t)

        assert y.shape == x.shape
