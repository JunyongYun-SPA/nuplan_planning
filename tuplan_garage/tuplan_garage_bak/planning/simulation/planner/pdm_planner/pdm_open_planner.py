import gc
import logging
import warnings
from typing import Type, cast

import torch
from nuplan.planning.simulation.observation.observation_type import (DetectionsTracks, Observation)
from nuplan.planning.simulation.planner.abstract_planner import (PlannerInitialization, PlannerInput)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (transform_predictions_to_states)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (InterpolatedTrajectory)
from nuplan.planning.training.modeling.lightning_module_wrapper import (LightningModuleWrapper)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.utils.serialization.scene import Trajectory

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import (AbstractPDMPlanner)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (get_drivable_area_map)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_feature_utils import (create_pdm_feature)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


# BH가 튜토리얼에서 추가한 Import들
import time
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder

from nuplan.planning.scenario_builder.scenario_utils import sample_indices_with_time_horizon
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneOnRouteStatusData,
    LaneSegmentConnections,
    LaneSegmentCoords,
    LaneSegmentGroupings,
    LaneSegmentTrafficLightData,
    get_neighbor_vector_map,
    get_on_route_status,
    get_traffic_light_encoding,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features_from_tensor,
    compute_yaw_rate_from_state_tensors,
    convert_absolute_quantities_to_relative,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    sampled_tracked_objects_to_tensor_list,
)
###


warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class PDMOpenPlanner(AbstractPDMPlanner):
    """PDM-Open planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(self,
                model: TorchModuleWrapper,
                checkpoint_path: str,
                map_radius: float):
        """
        Constructor for PDMOpenPlanner
        :param model: torch model
        :param checkpoint_path: path to checkpoint for model as string
        :param map_radius: radius around ego to consider
        """
        super(PDMOpenPlanner, self).__init__(map_radius)

        self._device = "cpu"

        self._model = LightningModuleWrapper.load_from_checkpoint(checkpoint_path, model=model,map_location=self._device).model

        self._model.eval()
        torch.set_grad_enabled(False)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        self._initialization = initialization # BH
        gc.collect()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput, tmp_scenario) -> AbstractTrajectory:
        """Inherited, see superclass."""

        gc.disable()
        ego_state, _ = current_input.history.current_state
        goal_ego_state = self._initialization.mission_goal
        goal_x = goal_ego_state.x
        goal_y = goal_ego_state.y
        route_roadblock_ids = self._initialization.route_roadblock_ids
        
        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state) 

        # Update/Create drivable area polygon map
        # BH: Visual_package 만들기
        # BH_vis_data_pack = {}
        # BH_vis_data_pack["route_seq"] = route_roadblock_ids
        # BH_vis_data_pack["goal_point"] = [goal_x, goal_y]
        # BH_vis_data_pack["ego_location"] = [ego_state.center.point.x, ego_state.center.point.y]
        # BH_vis_data_pack["current_iteration"] = self._iteration
        # BH_vis_data_pack["scenario_token"] = tmp_scenario.scenario.token
        # print(self._iteration)
        
        self._drivable_area_map = get_drivable_area_map(self._map_api, ego_state, self._, self._initialization, current_input, tmp_scenario._scenario.token)

        # Create centerline
        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))
        
        # feature building & model forward
        pdm_feature = create_pdm_feature(self._model, current_input, self._centerline, None, self._device)
        start = time.time()
        predictions = self._model.forward({"pdm_features": pdm_feature})
        end = time.time()
        #print(f"{end - start:.5f} sec")
        # convert to absolute
        trajectory_data = cast(Trajectory, predictions["trajectory"]).data
        trajectory = trajectory_data.cpu().detach().numpy()[0]

        trajectory = InterpolatedTrajectory(transform_predictions_to_states(trajectory,
                                                                            current_input.history.ego_states,
                                                                            self._model.trajectory_sampling.time_horizon,
                                                                            self._model.trajectory_sampling.step_time))

        # 튜토리얼 LaneGCN Implementation, BH
        # features = self.BH_tutorial_build_features(current_input, self._initialization)  
        ##

        self._iteration += 1
        return trajectory

    # BH가 nuplan 튜토리얼에서 가져옴
    def BH_tutorial_build_features(self, current_input: PlannerInput, initialization: PlannerInitialization) -> FeaturesType:
        """
        Makes a single inference on a Pytorch/Torchscript model.
        :param current_input: Iteration specific inputs for building the feature.
        :param initialization: Additional data require for building the feature.
        :return: dictionary of FeaturesType types.
        """
        ## BH 같은 동작 디버깅용 수정

        tutorial_vecmap = VectorMapFeatureBuilder(radius=self._map_radius)
        TrajectorySampling_ = TrajectorySampling
        TrajectorySampling_.interval_length = 0.5
        TrajectorySampling_.num_poses = 4
        TrajectorySampling_.step_time = 0.5
        TrajectorySampling_.time_horizon = 1.5
        tutorial_agents = AgentsFeatureBuilder(trajectory_sampling=TrajectorySampling_)
        
        feature_builders = [tutorial_vecmap, tutorial_agents]
        
        features = {
            builder.get_feature_unique_name(): builder.get_features_from_simulation(current_input, initialization)
            for builder in feature_builders
        }

        features = {name: feature.to_feature_tensor() for name, feature in features.items()}
        features = {name: feature.to_device(self._device) for name, feature in features.items()}
        features = {name: feature.collate([feature]) for name, feature in features.items()}
        return features
    