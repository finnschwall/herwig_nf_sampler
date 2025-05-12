from decouple import Config, RepositoryEnv
import os

if os.path.exists('config.ini'):
    config = Config(RepositoryEnv('config.ini'))
else:
    from decouple import config

INITIAL_POINTS = config('INITIAL_POINTS', default=150000, cast=int)
TRAINING_EPOCHS = config('TRAINING_EPOCHS', default=13, cast=int)

BATCH_SIZE = config('BATCH_SIZE', default=16384, cast=int)

LEARNING_RATE = config('LEARNING_RATE', default=6e-4, cast=float)#default=3e-4)

ALWAYS_RETRAIN = config('ALWAYS_RETRAIN', default=True, cast=bool)
"""If True, retrain the model even if it already exists. This is useful for debugging purposes."""

SPLIT_BY_CHANNELS = config('SPLIT_BY_CHANNELS', default=False, cast=bool)

BACKEND = config('BACKEND', default='madnis')


CHANNEL_DROP_THRESHOLD = config('CHANNEL_DROP_THRESHOLD', default=0.05, cast=float)
"""percentage of expected cross section (1/n_channels) after which the channel is dropped
"""

USE_CUDA = config('USE_CUDA', default=True, cast=bool)
"""GPU acceleration. Strongly recommended if available."""

COLLECT_TRAINING_INTEGRATION_METRICS = config('COLLECT_TRAINING_INTEGRATION_METRICS', default=False, cast=bool)
"""Collect training integration metrics. This will slow down the training process significantly.
    Use only for debugging purposes.
"""

