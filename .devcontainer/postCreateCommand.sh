

poetry install
poetry install
poetry run pip install --upgrade \
    'flax>=0.4.0' \
    'jax[cuda11_cudnn805]>=0.3.0' \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html

poetry run pip install --upgrade \
    'flax<0.4.0' \
    'jax[cuda11_cudnn805]<0.3.0' \
    -f https://storage.googleapis.com/jax-releases/jax_releases.html
