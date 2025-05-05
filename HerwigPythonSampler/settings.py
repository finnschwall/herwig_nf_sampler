from decouple import Config, RepositoryEnv
import os

if os.path.exists('config.ini'):
    config = Config(RepositoryEnv('config.ini'))
else:
    from decouple import config

INITIAL_POINTS = config('INITIAL_POINTS', default=20000, cast=int)
TRAINING_EPOCHS = config('TRAINING_EPOCHS', default=20, cast=int)

CHANNEL_SELECTION_DIM = config('CHANNEL_SELECTION_DIM', default=1, cast=int)
"""0 for LEP, 1 for LHC (usually)"""

BACKEND = config('BACKEND', default='madnis')


CHANNEL_DROP_THRESHOLD = config('CHANNEL_DROP_THRESHOLD', default=0.05, cast=float)
"""percentage of expected cross section (1/n_channels) after which the channel is dropped
"""

USE_CUDA = config('USE_CUDA', default=True, cast=bool)
"""GPU acceleration. Strongly recommended if available."""
