import jax
from pyuff_ustb import Uff

from vbeam.beamformers import get_das_beamformer
from vbeam.data_importers import import_pyuff
from vbeam.util.download import cached_download
import matplotlib.pyplot as plt

data_url = "http://www.ustb.no/datasets/PICMUS_carotid_cross.uff"
uff = Uff(cached_download(data_url))
channel_data = uff.read("/channel_data")
scan = uff.read("/scan")

setup = import_pyuff(channel_data, scan, frames=0)
#setup.scan = setup.scan.resize(x=200, z=400)
beamformer = jax.jit(get_das_beamformer(setup))

result = beamformer(**setup.data)
plt.imshow(result.T, aspect="auto", cmap="gray", vmin=-60)
plt.colorbar()
plt.show(block=True)