from tutorials.utils.tutorial_utils import setup_notebook
from nuplan.planning.script.builders.scenario_builder import build_scenarios
from nuplan.planning.script.builders.worker_pool_builder import build_worker
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.utils.multithreading.worker_utils import WorkerPool, worker_map
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
CONFIG_PATH = '../../../../nuplan-devkit/nuplan/planning/script/config/simulation'
CONFIG_NAME = 'default_simulation'

# Create a temporary directory to store the cache and experiment artifact
SPLIT='reduced_val14_split'
CHALLENGE = 'closed_loop_nonreactive_agents' # open_loop_boxes, closed_loop_nonreactive_agents, closed_loop_reactive_agents
CHECKPOINT='/home/workspace/dataset/nuplan/exp/exp/training_hivt_unimodal_model_120m_v3_goal_ver3/training_hivt_unimodal_model_120m_v3_goal_ver3/training_hivt_unimodal_model_120m_v3_goal_ver3/2024.03.17.03.38.11/checkpoints/last.ckpt'
PLANNER = 'hivt_unimodal_120m_v3_goal_planner_ver3'
# LOG_DIR = SAVE_DIR / EXPERIMENT / JOB_NAME

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize(config_path=CONFIG_PATH)

# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    f'+simulation={CHALLENGE}',  #JY # raster model that consumes ego, agents and map raster layers and regresses the ego's trajectory
    f'planner={PLANNER}',
    'worker=ray_distributed',
    f'planner.pdm_open_planner.checkpoint_path={CHECKPOINT}',
    f'scenario_filter={SPLIT}',
    f'scenario_builder=nuplan'])
 
from nuplan.planning.script.run_simulation import main

# cfg.scenario_filter.limit_total_scenarios = 1


main(cfg)