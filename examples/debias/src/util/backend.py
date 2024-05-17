import jax

def jax_backends():
  backends = []
  for backend in ['cpu', 'gpu', 'tpu']:
    try:
      jax.devices(backend)
    except RuntimeError:
      pass
    else:
      backends.append(backend)
  return backends
