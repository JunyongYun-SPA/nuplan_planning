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
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name', type=str, required=False, default='train_hivt_unimodal_model_BEV_feature_baseline_v2')
parser.add_argument('--version', type=str, default='1')
parser.add_argument('--hidden_dim', type=int, required=False, default=64) 
parser.add_argument('--batch_size', type=int, required=False, default=32)
parser.add_argument('--resolution', type=int, required=False, default=100)
parser.add_argument('--map_channel_dimension', type=int, required=False, default=4)
parser.add_argument('--iterative_centerline', type=bool, required=False, default=False) 
parser.add_argument('--occupied_area', type=bool, required=False, default=False) 
parser.add_argument('--pos_emb', type=str, required=False, default='learnable') 
parser.add_argument('--agent_temporal_pos_emb', type=bool, required=False, default=False) 
parser.add_argument('--additional_state', type=bool, required=False, default=False) 
parser.add_argument('--multimodal', type=bool, required=False, default=False) 
parser.add_argument('--cnn', type=str, required=False, default='default') 
args = parser.parse_args()

logger = logging.getLogger(__name__)

# Location of path with all training configs
# NUPLAN_ROOT_ = os.environ['NUPLAN_DEVKIT_ROOT']
CONFIG_PATH = '../../../nuplan-devkit/nuplan/planning/script/config/training'
CONFIG_NAME = 'default_training'

# Create a temporary directory to store the cache and experiment artifact
EXPERIMENT = 'training_hivt_py' #JY
TRAINING = args.name #JY
RESULT_SAVE_DIR = f'/home/workspace/dataset/nuplan/exp/exp/{EXPERIMENT}/'  # optionally replace with persistent dir
CACHE_PATH = '/home/workspace/dataset/cache/hivt_occupancy_v4' 
MINISET_TOKEN_LIST = '/home/mnt/sda/nuplan/dataset/cache/mini_set_token_list_final.pickle'
TRAIN_START_WITH_CHECK_POINT = "False"
CHECK_POINT_PATH = "/home/workspace/dataset/nuplan/exp/exp/training_hivt_py/training_hivt_py/train_hivt_unimodal_model_BEV_feature/2024.07.07.23.32.02/checkpoints/last_77.ckpt"

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
    f'experiment_name={EXPERIMENT}_v{args.version}',
    f'job_name={TRAINING}',
    'scenario_builder=nuplan',  # use nuplan mini database
    'worker=sequential',
    f'lightning.trainer.checkpoint.resume_training={TRAIN_START_WITH_CHECK_POINT}',
    f'+checkpoint_path={CHECK_POINT_PATH}',
     # 'scenario_filter.limit_total_scenarios=500',  # Choose 500 scenarios to train with
    'lightning.trainer.params.accelerator=ddp_spawn',  # ddp is not allowed in interactive environment, using ddp_spawn instead - this can bottleneck the data pipeline, it is recommended to run training outside the notebook
    'lightning.trainer.params.max_epochs=150',
    'data_loader.params.batch_size=32',
    'data_loader.params.num_workers=0',
    'optimizer.lr=5e-4',
    'lr_scheduler=multistep_lr',
    'lr_scheduler.milestones=[50,75,100]',
    'lr_scheduler.gamma=0.1',
    f'model.hidden_dim={args.hidden_dim}',
    f'model.batch_size={args.batch_size}',
    f'model.resolution={args.resolution}',
    f'model.map_channel_dimension={args.map_channel_dimension}',
    f'model.iterative_centerline={args.iterative_centerline}',
    f'model.occupied_area={args.occupied_area}',
    f'model.pos_emb={args.pos_emb}',
    f'model.agent_temporal_pos_emb={args.agent_temporal_pos_emb}',
    f'model.additional_state={args.additional_state}',
    f'model.multimodal={args.multimodal}',
    f'model.cnn={args.cnn}',
])
 
from nuplan.planning.script.run_training import main

main(cfg)