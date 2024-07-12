import gc
import logging
import warnings
from typing import Type, cast

import torch
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    transform_predictions_to_states,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.lightning_module_wrapper import (
    LightningModuleWrapper,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.utils.serialization.scene import Trajectory

from tuplan_garage.planning.simulation.planner.pdm_planner.abstract_pdm_planner import (
    AbstractPDMPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation_utils import (
    get_drivable_area_map,
)
# from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_feature_utils import (
#     create_pdm_feature,
# )
from tuplan_garage.planning.simulation.planner.hivt_planner.utils.hivt_50m_goal_feature_utils_v2 import (
    create_hivt_feature,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

# BH 추가 import
from tuplan_garage.planning.training.preprocessing.features.hivt_feature import (
    HiVTFeature, TemporalData #JY
)

import numpy as np
import torch.nn as nn

from tuplan_garage.planning.training.modeling.models.pgp.utils import (
    get_traversal_coordinates,
    smooth_centerline_trajectory,
    waypoints_to_trajectory,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_graph_map_feature_builder_utils import (
    calculate_lane_progress,
    convert_absolute_to_relative_array,
    points_in_polygons,
)

import numpy as np

import torch.nn as nn

from tuplan_garage.planning.training.modeling.models.pgp.utils import (
    get_traversal_coordinates,
    smooth_centerline_trajectory,
    waypoints_to_trajectory,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)

class HivtMultimodal(AbstractPDMPlanner):
    """PDM-Open planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        model: TorchModuleWrapper,
        checkpoint_path: str,
        map_radius: float,
    ):
        """
        Constructor for PDMOpenPlanner
        :param model: torch model
        :param checkpoint_path: path to checkpoint for model as string
        :param map_radius: radius around ego to consider
        """
        super(HivtMultimodal, self).__init__(map_radius)

        self._device = "cpu"

        self._model = LightningModuleWrapper.load_from_checkpoint(
            checkpoint_path,
            model=model,
            map_location=self._device,
        ).model

        self._model.eval()
        torch.set_grad_enabled(False)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization.map_api
        self._initialization = initialization
        self._load_route_dicts(initialization.route_roadblock_ids)
        gc.collect()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(
        self, current_input: PlannerInput, 
        scenario = None
    ) -> AbstractTrajectory:
        """Inherited, see superclass."""

        gc.disable()
        ego_state, _ = current_input.history.current_state

        # Apply route correction on first iteration (ego_state required)
        if self._iteration == 0:
            self._route_roadblock_correction(ego_state)

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # Create centerline
        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))

        # feature building & model forward
        pdm_feature, vis_pack = create_hivt_feature(
            self._model, current_input, self._centerline, None, self._model.device, scenario=scenario, map_api=self._map_api, initialization=self._initialization
        )
        temporal_feature = TemporalData(**pdm_feature)
        self._model.device = 'cpu'
        predictions = self._model.forward({"pdm_features": temporal_feature})
        
        # pi = predictions['pi']
        # pi = pi[pdm_feature['av_index'], :]
        # pi_best_mode = torch.argmax(pi)
        # GT 기준 마지막 포인트가 가장 가까운 Trajectory 선택
        
        
        # if self._iteration >5:
        #     raise ValueError
        # if self._iteration == 0:
        #     print()
        # if scenario.token == '0a3d3adc5ae45cc2':
        #     print()
        # else:
        #     raise ValueError
        y_hat = predictions['trajectory'][:, pdm_feature['av_index'], : , :].detach().cpu()
        route_count_ths = list(range(y_hat.shape[1]))[::-1]
        for route_count_th in route_count_ths:
            y_hat_in_route_mask = []
            for mode_, single_traj in enumerate(y_hat):
                traj_in_route = points_in_polygons(single_traj, temporal_feature["AV_centric_route"])
                all_lane_in_route = []
                for timeStep in range(y_hat.shape[1]):
                    all_lane_in_route.append(traj_in_route[:, timeStep].any())
                all_lane_in_route = np.array(all_lane_in_route)
                if len(np.where(all_lane_in_route==True)[0]) > route_count_th:
                    y_hat_in_route_mask.append(mode_)
            if len(y_hat_in_route_mask) != 0:
                break
        y_hat = y_hat[y_hat_in_route_mask, ...]
        
        # TH로 자르기(초기 0.5부터 시작하여 0.1씩 TH를 낮추기)
        pi = predictions['pi']
        pi_ = pi[temporal_feature['av_index']]
        pi_ = pi_[y_hat_in_route_mask]
        softmax = nn.Softmax(dim=0)
        pi_ = softmax(pi_)

        initial_THs = [0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3]
        for TH_ in initial_THs:
            index_ = torch.where(pi_ > TH_)[0]
            if len(index_) != 0:
                break
        y_hat = y_hat[index_]
        # print()
        
        if len(y_hat) == 0:
            y_hat = predictions['trajectory'][:, pdm_feature['av_index'], : , :].detach().cpu()
        
        # TH 이후 GT가 기준이면 이걸로
        # y_hat_Fs = y_hat[:, -1, :]
        # GT_F = temporal_feature['y'][temporal_feature['av_index'], -1, :].unsqueeze(0)
        # pi_best_mode = torch.argmin(torch.cdist(y_hat_Fs, GT_F))
        # y_hat = y_hat[pi_best_mode, ...]

        # TH 이후 Centerline이 기준이면 이걸로
        y_hat_Fs = y_hat[:, -1, :].type(torch.float)
        GT_F = torch.tensor(temporal_feature['centerline'][-1, :2]).unsqueeze(0).type(torch.float)
        pi_best_mode = torch.argmin(torch.cdist(y_hat_Fs, GT_F))
        y_hat = y_hat[pi_best_mode, ...]
        
        # convert to absolute
        y_hat = y_hat.unsqueeze(0)
        current_ego_vel = np.array([current_input.history.current_state[0]._dynamic_car_state.speed])
        traj_w_heading = waypoints_to_trajectory(y_hat, current_ego_vel).detach().numpy()[0]
        
        
            
        trajectory = InterpolatedTrajectory(
            transform_predictions_to_states(
                traj_w_heading,
                current_input.history.ego_states,
                self._model.trajectory_sampling.time_horizon,
                self._model.trajectory_sampling.step_time,
            )
        )
        

        self._iteration += 1
        return trajectory

