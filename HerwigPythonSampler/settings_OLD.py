#Settings file used for systematic runs
from decouple import Config, RepositoryEnv
import os


if os.path.exists('config.ini'):
    config = Config(RepositoryEnv('config.ini'))
else:
    from decouple import config



INITIAL_POINTS = config('INITIAL_POINTS', default=5e6, cast=int)
TRAINING_EPOCHS = config('TRAINING_EPOCHS', default=5, cast=int)
ZERO_POINT_REMOVAL = config('ZERO_POINT_REMOVAL', default=0.3, cast=float)
OPTIMIZER = config('OPTIMIZER', default='adam', cast=str)
"""adam, adamw, rmsprop, sgd"""

NORMALIZE_CROSS_SECTIONS = config('NORMALIZE_CROSS_SECTIONS', default=False, cast=bool)


BATCH_SIZE = config('BATCH_SIZE', default=32768, cast=int)

LEARNING_RATE = config('LEARNING_RATE', default=6e-4, cast=float)#default=3e-4)

ALWAYS_RETRAIN = config('ALWAYS_RETRAIN', default=False, cast=bool)
"""If True, retrain the model even if it already exists. This is useful for debugging purposes."""

SPLIT_BY_CHANNELS = config('SPLIT_BY_CHANNELS', default=False, cast=bool)

NUM_COUPLING_BLOCKS = config('NUM_COUPLING_BLOCKS', default=16, cast=int)
COUPLING_CONSTRUCTOR = config('COUPLING_CONSTRUCTOR', default='log', cast=str)
"""log, exchange, random"""
UNITS_PER_SUBNET = config('UNITS_PER_SUBNET', default=32, cast=int)
SUBNET_LAYERS = config('SUBNET_LAYERS', default=3, cast=int)

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

LIVE_TRAINING_PLOT = config('LIVE_TRAINING_PLOT', default=False, cast=bool)

SEED = config('SEED', default=42, cast=int)

PLOT_DIMS = config('PLOT_DIMS', default=True, cast=bool)
"""Plot the phase space distribution"""

MULTI_STAGE_TRAINING = config('MULTI_STAGE_TRAINING', default=False, cast=bool)
PER_STAGE_EPOCHS = config('PER_STAGE_EPOCHS', default=2, cast=int)
PER_STAGE_POINTS = config('PER_STAGE_POINTS', default=1.67e6, cast=int)
NUM_STAGES = config('NUM_STAGES', default=3, cast=int)
SKIP_PARAM = config('SKIP_PARAM', default=1, cast=int)



FINAL_SAMPLE_SIZE = config('FINAL_SAMPLE_SIZE', default=500000, cast=int)

