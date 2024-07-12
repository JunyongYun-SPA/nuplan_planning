import gc
import logging
import warnings
from typing import Type, cast

import torch
import torch.nn as nn
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
from tuplan_garage.planning.simulation.planner.hivt_planner.utils.bh_hivt_120m_goal_feature_utils_ver4 import (
    create_hivt_feature,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

# BH 추가 import
from tuplan_garage.planning.training.preprocessing.features.hivt_feature import (
    HiVTFeature, TemporalData #JY
)
import numpy as np

from tuplan_garage.planning.training.modeling.models.pgp.utils import (
    get_traversal_coordinates,
    smooth_centerline_trajectory,
    waypoints_to_trajectory,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_graph_map_feature_builder_utils import (
    calculate_lane_progress,
    convert_absolute_to_relative_array,
    points_in_polygons,
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
        # self._model = self._model.to(self._device)

        self._model.eval()
        torch.set_grad_enabled(False)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        self.initialization = initialization
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
        plt.scatter(lane_x, lane_y, s = 25, c="black", alpha=0.2, zorder=1)
        
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
                
        valid_fut_t = [t_ for t_ in fut_t if padding_mask[av_index, t_] == False]
        agent_pos = x[av_index, valid_fut_t]
        pos_xs = agent_pos[:, 0].numpy()
        pos_ys = agent_pos[:, 1].numpy()
        pos_yaws = agent_pos[:, 2].numpy()
        plt.scatter(pos_xs, pos_ys, s = 30, c="Magenta", edgecolor='black', alpha=1.0)
        
        pi = predictions['pi']
        pi = pi[temporal_feature['av_index'], :]
        pi_best_mode = torch.argmax(pi)
        
        for i in range(6):
            if i == int(pi_best_mode):
                pos_xs = predictions['trajectory'][i, temporal_feature["av_index"], :,:2].numpy()[:, 0]
                pos_ys = predictions['trajectory'][i, temporal_feature["av_index"], :,:2].numpy()[:, 1]
                plt.scatter(pos_xs, pos_ys, s = 30, c="dodgerblue", edgecolor='black', alpha=1.0, zorder=6)
            else:
                pos_xs = predictions['trajectory'][i, temporal_feature["av_index"], :,:2].numpy()[:, 0]
                pos_ys = predictions['trajectory'][i, temporal_feature["av_index"], :,:2].numpy()[:, 1]
                plt.scatter(pos_xs, pos_ys, s = 30, c="limegreen", edgecolor='black', alpha=1.0, zorder=4)
        pos_xs = vis_pack['centerline'][:, 0]
        pos_ys = vis_pack['centerline'][:, 1]
        plt.scatter(pos_xs, pos_ys, s = 30, c="orange", edgecolor='black', alpha=0.2, zorder=3) 
        
        pos_xs = temporal_feature['y'][temporal_feature["av_index"], :, 0]
        pos_ys = temporal_feature['y'][temporal_feature["av_index"], :, 1]
        plt.scatter(pos_xs, pos_ys, s = 30, c="red", edgecolor='black', alpha=0.2, zorder=5)        
                

        plt.savefig(img_path)
        

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
        
        pdm_feature, vis_pack = create_hivt_feature(
            self._model, current_input, self._centerline, None, self._model.device, scenario=scenario, map_api=self._map_api, initialization = self.initialization
        )
        temporal_feature = TemporalData(**pdm_feature)
        # torch.save(temporal_feature, cache_path)
        self._model.device = 'cpu'
        predictions = self._model.forward({"pdm_features": temporal_feature})
        
        # # 시각화
        # if self._iteration==0:
        #     print()
        #     img_path = f"/home/workspace/pictures/{scenario.token}_{self._iteration}.png"
        #     self._vis(temporal_feature, vis_pack, predictions, img_path)
        # else:
        #     raise ValueError

        # convert to absolute
        y_hat = predictions["trajectory"][:,temporal_feature['av_index'],:, :2]
        pi = predictions['pi']
        pi = pi[temporal_feature['av_index'], :]

        softmax = nn.Softmax(dim=0)
        pi_ = softmax(pi)
        initial_THs = [0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3]
        for TH_ in initial_THs:
            index_ = torch.where(pi_ > TH_)[0]
            if len(index_) != 0:
                break
        y_hat = y_hat[index_]
        pi = pi[index_]
        
        # route내 lane point와 traj 사이의 거리
        try:
            in_route_lanes_index = torch.where(temporal_feature["in_route_lanes"])[0]
            route_positions = temporal_feature["lane_positions"][in_route_lanes_index]
            y_hat_multiple = y_hat.clone()
            mean_dist = []
            y_hat_in_route_mask_mode_index = []
            for mode_, single_traj in enumerate(y_hat):
                dist_map = torch.cdist(route_positions, single_traj)
                dist = torch.mean(torch.min(dist_map, axis = 0)[0])
                mean_dist.append(dist.item())
            y_hat_in_route_mask_mode_index = list(np.argsort(mean_dist))[0]
            y_hat_ = y_hat[y_hat_in_route_mask_mode_index, ...].unsqueeze(0)
            pi_ = pi[y_hat_in_route_mask_mode_index]
        except Exception as e:
            result_pi_index = torch.argmax(pi)
            y_hat_ = y_hat[result_pi_index]
            y_hat_= y_hat_.unsqueeze(0)
            
        current_ego_vel = np.array([current_input.history.current_state[0].dynamic_car_state.speed])
        traj_w_heading = waypoints_to_trajectory(y_hat_, current_ego_vel).detach().numpy()[0]
        
        
            
        trajectory = InterpolatedTrajectory(
            transform_predictions_to_states(
                traj_w_heading,
                current_input.history.ego_states,
                self._model.trajectory_sampling.time_horizon,
                self._model.trajectory_sampling.step_time,
            )
        )
        

        self._iteration += 1
        return trajectory, 0
