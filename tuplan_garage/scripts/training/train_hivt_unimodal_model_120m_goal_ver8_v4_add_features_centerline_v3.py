from tutorials.utils.tutorial_utils import setup_notebook
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map

# (Optional) Increase notebook width for all embedded cells to display properly
from IPython.core.display import display, HTML
display(HTML("<style>.output_result { max-width:100% !important; }</style>"))
display(HTML("<style>.container { width:100% !important; }</style>"))

# Useful imports
import os
from pathlib import Path
import tempfile

import hydra
from tqdm import tqdm

import pickle
from pathlib import Path
import ray
import shutil

import gc
import itertools
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from omegaconf import DictConfig

from nuplan.common.utils.distributed_scenario_filter import DistributedMode, DistributedScenarioFilter
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.abstract_scenario_builder import AbstractScenarioBuilder, RepartitionStrategy
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.scenario_building_builder import build_scenario_builder
from nuplan.planning.script.builders.scenario_filter_builder import build_scenario_filter
from nuplan.planning.training.experiments.cache_metadata_entry import (CacheMetadataEntry,
                                                                        CacheResult,
                                                                        save_cache_metadata)
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import chunk_list, worker_map

logger = logging.getLogger(__name__)

# Location of path with all training configs
# NUPLAN_ROOT_ = os.environ['NUPLAN_DEVKIT_ROOT']
CONFIG_PATH = '../../../nuplan-devkit/nuplan/planning/script/config/training'
CONFIG_NAME = 'default_training'

# Create a temporary directory to store the cache and experiment artifact
EXPERIMENT = 'training_hivt_py' #JY
TRAINING = 'train_hivt_unimodal_model_120m_goal_ver8_v4_add_features_centerline_v3' #JY
RESULT_SAVE_DIR = f'/home/workspace/dataset/nuplan/exp/exp/{EXPERIMENT}/'  # optionally replace with persistent dir
CACHE_PATH = '/home/workspace/cache/hivt_120m_goal_ver10_0629' 
MINISET_TOKEN_LIST = '/home/mnt/sda/nuplan/dataset/cache/mini_set_token_list_final.pickle'
# LOG_DIR = SAVE_DIR / EXPERIMENT / JOB_NAME

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    'seed=0',
    'py_func=train',
    f'+training={TRAINING}',  #JY # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
    f'group={str(RESULT_SAVE_DIR)}',
    f'cache.cache_path={str(CACHE_PATH)}',
    f'cache.use_cache_without_dataset=True',
    f'experiment_name={EXPERIMENT}',
    f'job_name={TRAINING}',
    'scenario_builder=nuplan',  # use nuplan mini database
    'worker=sequential',
     # 'scenario_filter.limit_total_scenarios=500',  # Choose 500 scenarios to train with
    'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
    'lightning.trainer.params.max_epochs=100',
    'data_loader.params.batch_size=32',
    'data_loader.params.num_workers=0',
    'optimizer.lr=5e-4',
    'lr_scheduler=multistep_lr',
    'lr_scheduler.milestones=[50,75]',
    'lr_scheduler.gamma=0.1'
])
 
from nuplan.planning.script.run_training import main

main(cfg)