import ddpm_jax.main
import importlib

importlib.reload(ddpm_jax.main)

main_locals = ddpm_jax.main.main(
    steps_per_epoch=500,
    dataset_params=("cgarciae/cartoonset", "100k"),
    image_shape=(64, 64),
    dims=64,
)

locals().update(main_locals)
