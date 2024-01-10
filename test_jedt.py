from f110_gym.envs.utils import edt as jedt
from scipy.ndimage import distance_transform_edt as edt
from f110_gym.envs.track import Track
import numpy as np
import jax.numpy as jnp
from f110_gym.envs.utils import JaxEnum

class Test(JaxEnum):
    A: 1
    B: 2

t = Test()

assert False
track = Track.from_track_name("Spielberg")
map_img_real = track.occupancy_map

map_img = 255 * np.ones((5, 5))
map_img[2, 3] = 0

dt = edt(map_img)
jdt = jedt(jnp.array(map_img))

print(map_img)
print('------')
print(dt)
print('------')
print(jdt)
assert np.allclose(dt, jdt)
print(f"Transforms equal: {np.allclose(dt, jdt)}.")

# benchmark
import timeit

print(timeit.timeit("edt(map_img)", globals=globals()))
print(timeit.timeit("jedt(jnp.array(map_img))", globals=globals()))