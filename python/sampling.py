import herwig_python

import sys
sys.path.insert(0, '/home/finn/.pyenv/versions/3.10.16/envs/madnis/lib/python3.10/site-packages')
sys.path.append('/mnt/data-slow/herwig/python/madnis')

#import Sampling.python.nf_old as nf_old
import flat
import vg

# Singleton instance
python_sampler_instance = flat.FlatSampler()
#nf.MadnisSampler()
#flat.FlatSampler()
#vg.VegasSampler()
def train(python_sampler, n_dims, diagram_dimension):
    python_sampler_instance.setup_base(python_sampler, n_dims, diagram_dimension)
    return python_sampler_instance.train()

def load(python_sampler, n_dims):
    python_sampler_instance.setup_base(python_sampler, n_dims)
    return python_sampler_instance.load()