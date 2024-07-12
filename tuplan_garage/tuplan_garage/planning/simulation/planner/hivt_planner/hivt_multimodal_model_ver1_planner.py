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
from tuplan_garage.planning.simulation.planner.hivt_planner.utils.hivt_50m_goal_feature_utils_ver4 import (
    create_hivt_feature,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

# BH 추가 import
from tuplan_garage.planning.training.preprocessing.features.hivt_feature import (
    HiVTFeature, TemporalData #JY
)

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

import time

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
    
    def _vis(self, temporal_feature, vis_pack, predictions, img_path):
        ### visualization
        # visualization
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        cur_t = 11
        hist_t = list(range(0, 11))
        fut_t = list(range(11, 27))
        plt.figure(figsize=(17,17))
        plt.xlim(-150, 150)
        plt.ylim(-150, 150)
        padding_mask = vis_pack["padding_mask"]
        x = vis_pack["x"]
        av_index = temporal_feature["av_index"]
        lane_positions = vis_pack["lane_positions"]
        agent_w = 5
        agent_h = 2
        
        lane_x = lane_positions[:, 0]
        lane_y = lane_positions[:, 1]
        plt.scatter(lane_x, lane_y, s = 25, c="black", alpha=0.2)
        
        for road_block in vis_pack['local_route']:
            x__,y__ = road_block.exterior.xy
            plt.plot(x__,y__)
        
        for node_idx in range(padding_mask.shape[0]):
            if padding_mask[node_idx, cur_t] == True:
                continue
            else:
                valid_hist_t = [t_ for t_ in hist_t if padding_mask[node_idx, t_] == False]
                agent_pos = x[node_idx, valid_hist_t]
                pos_xs = agent_pos[:, 0]
                pos_ys = agent_pos[:, 1]
                pos_yaws = agent_pos[:, 2]
                for index_ in range(pos_xs.shape[0]):
                    pos_x = pos_xs[index_]
                    pos_y = pos_ys[index_]
                    pos_yaw = pos_yaws[index_]
                    if node_idx == av_index:
                        rect = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                                edgecolor='red', facecolor='red', alpha = 1/(pos_xs.shape[0]-index_))
                    else:
                        rect = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                                edgecolor='blue', facecolor='blue', alpha = 1/(pos_xs.shape[0]-index_))
                    ax = plt.gca()
                    t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + ax.transData
                    rect.set_transform(t)
                    ax.add_patch(rect)
                
        valid_fut_t = [t_ - 11 for t_ in fut_t if padding_mask[av_index, t_] == False]
        valid_fut_t_ = [t_ for t_ in fut_t if padding_mask[av_index, t_] == False]
        agent_pos =  predictions[valid_fut_t, :2].clone().detach().cpu()
        pos_xs = agent_pos[:, 0]
        pos_ys = agent_pos[:, 1]
        plt.scatter(pos_xs, pos_ys, s = 30, c="Magenta", edgecolor='black', alpha=0.5)
        agent_pos = x[av_index, valid_fut_t_].clone().detach().cpu()
        pos_xs = agent_pos[:, 0].numpy()
        pos_ys = agent_pos[:, 1].numpy()
        plt.scatter(pos_xs, pos_ys, s = 30, c="limegreen", edgecolor='black', alpha=0.5)
        pos_xs = vis_pack['centerline'][:, 0]
        pos_ys = vis_pack['centerline'][:, 1]
        plt.scatter(pos_xs, pos_ys, s = 30, c="orange", edgecolor='black', alpha=0.2)     
                
        plt.savefig(img_path)
                # pos_yaws = agent_pos[:, 2]
                # for index_ in range(pos_xs.shape[0]):
                #     pos_x = pos_xs[index_]
                #     pos_y = pos_ys[index_]
                #     pos_yaw = pos_yaws[index_]
                #     rect = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                #                                 edgecolor='red', facecolor='green', alpha = 1/(pos_xs.shape[0]-index_))
                #     ax = plt.gca()
                #     t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + ax.transData
                #     rect.set_transform(t)
                #     ax.add_patch(rect)
                # for index_ in range(pos_xs.shape[0]):
                #     pos_x = pos_xs[index_]
                #     pos_y = pos_ys[index_]
                #     pos_yaw = pos_yaws[index_]
                #     if node_idx == av_index:
                #         rect = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                #                 edgecolor='red', facecolor='red', alpha = 1/(pos_xs.shape[0]-index_))
                #     else:
                #         rect = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                #                 edgecolor='blue', facecolor='blue', alpha = 1/(pos_xs.shape[0]-index_))
                #     ax = plt.gca()
                #     t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + ax.transData
                #     rect.set_transform(t)
                #     ax.add_patch(rect)    


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
        start_time_s = time.perf_counter()
        predictions = self._model.forward({"pdm_features": temporal_feature}.copy())
        forward_time = time.perf_counter() - start_time_s
        
        # pi = predictions['pi']
        # pi = pi[pdm_feature['av_index'], :]
        # pi_best_mode = torch.argmax(pi)
        # GT 기준 마지막 포인트가 가장 가까운 Trajectory 선택
        

        y_hat = predictions['trajectory'][:, pdm_feature['av_index'], : , :].detach().cpu()
        # y_hat_multiple = y_hat.clone()
        # route_count_ths = list(range(y_hat.shape[1]))[::-1]
        # for route_count_th in route_count_ths:
        #     y_hat_in_route_mask = []
        #     for mode_, single_traj in enumerate(y_hat):
        #         traj_in_route = points_in_polygons(single_traj, temporal_feature["AV_centric_route"])
        #         all_lane_in_route = []
        #         for timeStep in range(y_hat.shape[1]):
        #             all_lane_in_route.append(traj_in_route[:, timeStep].any())
        #         all_lane_in_route = np.array(all_lane_in_route)
        #         if len(np.where(all_lane_in_route==True)[0]) > route_count_th:
        #             y_hat_in_route_mask.append(mode_)
        #     if len(y_hat_in_route_mask) != 0:
        #         break
        # # y_hat = y_hat[y_hat_in_route_mask, ...]
        
        # # TH로 자르기(초기 0.5부터 시작하여 0.1씩 TH를 낮추기)
        # pi = predictions['pi']
        # pi_ = pi[temporal_feature['av_index']]
        # # pi_ = pi_[y_hat_in_route_mask]
        # softmax = nn.Softmax(dim=0)
        # pi_ = softmax(pi_)

        # initial_THs = [0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3]
        # for TH_ in initial_THs:
        #     index_ = torch.where(pi_ > TH_)[0]
        #     if len(index_) != 0:
        #         break
        # y_hat = y_hat[index_]
        
        # else:
        #     print()
        
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
        
        # if self._iteration % 10 == 0:
        #     img_path = f"/home/workspace/pictures/route_{scenario.token}_{self._iteration}_[{route_count_th}].png"
        #     self._vis(temporal_feature, vis_pack, y_hat, img_path)
        
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
        return trajectory, forward_time
