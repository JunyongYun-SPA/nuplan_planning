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
        pos_xs = predictions['trajectory'][temporal_feature["av_index"], :,:2].numpy()[:, 0]
        pos_ys = predictions['trajectory'][temporal_feature["av_index"], :,:2].numpy()[:, 1]
        plt.scatter(pos_xs, pos_ys, s = 30, c="limegreen", edgecolor='black', alpha=1.0)
        pos_xs = vis_pack['centerline'][:, 0]
        pos_ys = vis_pack['centerline'][:, 1]
        plt.scatter(pos_xs, pos_ys, s = 30, c="orange", edgecolor='black', alpha=0.2)        
                

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

        # feature building & model forward
        pdm_feature, vis_pack = create_hivt_feature(
            self._model, current_input, self._centerline, None, self._model.device, scenario=scenario, map_api=self._map_api
        )
        temporal_feature = TemporalData(**pdm_feature)
        self._model.device = 'cpu'
        predictions = self._model.forward({"pdm_features": temporal_feature})
        
        # pi = predictions['pi']
        # pi = pi[pdm_feature['av_index'], :]
        # pi_best_mode = torch.argmax(pi) #pi.sort().indices[3:] #torch.argmax(pi)
        # predictions['trajectory'] = predictions["trajectory"][pi_best_mode, ...]
        
        # # reg_mask = ~pdm_feature['padding_mask'][pdm_feature['av_index'], 11:]
        data_goal = pdm_feature['centerline'][:, :2]
        mode_dist_list = []
        softmax = nn.Softmax(dim=0)
        for i in range(6):
            l2_norm = torch.cdist(predictions["trajectory"][i, pdm_feature['av_index'], :, : 2], torch.tensor(data_goal, dtype=torch.float32))
            mode_dist_list.append(torch.mean(l2_norm.min(dim=1).values))
        # best_mode = torch.argmin(torch.tensor(mode_dist_list))
        pi = predictions['pi']
        pi = pi[pdm_feature['av_index'], :]
        
        pi2 = softmax(torch.tensor(mode_dist_list))
        # # l2_norm = torch.norm(predictions["trajectory"][:, pdm_feature['av_index'], -1, : 2] - torch.tensor(data_goal, dtype=torch.float32), p=2, dim=-1)
        # softmax = nn.Softmax(dim=0)
        # l2_norm_ = softmax(l2_norm.min(dim=1).values)
        pi_ = pi + pi2
        best_mode = torch.argmax(pi_)
        # best_mode = torch.mean(l2_norm.min(dim=1).values) #l2_norm.min(dim=1).values.argmin(dim=0) #l2_norm.min(dim=1).values.argmin(dim=0) l2_norm.argmin(dim=0)
        y_hat_best = predictions["trajectory"][best_mode, torch.arange(predictions["trajectory"].shape[1])]
        predictions['trajectory'] = y_hat_best
        
        # l2_norm = torch.norm(predictions["trajectory"][:, pdm_feature['av_index'], -1, : 2] - pdm_feature["y"][pdm_feature['av_index'], -1, :2], p=2, dim=-1)
        # best_mode = l2_norm.argmin(dim=0)
        # y_hat_best = predictions["trajectory"][best_mode, torch.arange(predictions["trajectory"].shape[1])]
        # predictions['trajectory'] = y_hat_best
        
        # pi = predictions['pi']
        # pi = pi[pdm_feature['av_index'], :]
        # pi_best_mode = torch.argmax(pi)
        # GT 기준 마지막 포인트가 가장 가까운 Trajectory 선택
        
        # y_hat = predictions['trajectory'][:, pdm_feature['av_index'], : , :]
        # # TH로 자르기(초기 0.5부터 시작하여 0.1씩 TH를 낮추기)
        # pi = predictions['pi']
        # pi_ = pi[temporal_feature['av_index']]
        # softmax = nn.Softmax(dim=0)
        # pi_ = softmax(pi_)

        # initial_THs = [0.5, 0.4, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3]
        # for TH_ in initial_THs:
        #     index_ = torch.where(pi_ > TH_)[0]
        #     if len(index_) != 0:
        #         break
        # y_hat = y_hat[index_]
        # # print()
        
        # # TH 이후 GT가 기준이면 이걸로
        # y_hat_Fs = y_hat[:, -1, :]
        # GT_F = temporal_feature['y'][temporal_feature['av_index'], -1, :].unsqueeze(0)
        # pi_best_mode = torch.argmin(torch.cdist(y_hat_Fs, GT_F))
        # y_hat = y_hat[pi_best_mode, ...]

        # # TH 이후 Centerline이 기준이면 이걸로
        # y_hat_Fs = y_hat[:, -1, :].type(torch.float)
        # GT_F = torch.tensor(temporal_feature['centerline'][-1, :2]).unsqueeze(0).type(torch.float)
        # pi_best_mode = torch.argmin(torch.cdist(y_hat_Fs, GT_F))
        # y_hat = y_hat[pi_best_mode, ...]
        
        # convert to absolute
        # y_hat = y_hat.unsqueeze(0)
        # current_ego_vel = np.array([current_input.history.current_state[0]._dynamic_car_state.speed])
        # traj_w_heading = waypoints_to_trajectory(y_hat, current_ego_vel).detach().numpy()[0]


        
        # # 시각화
        # if self._iteration==0:
        #     print()
        #     img_path = f"/home/workspace/pictures/{scenario.token}_{self._iteration}.png"
        #     self._vis(temporal_feature, vis_pack, predictions, img_path)
        # else:
        #     raise ValueError
        # /home/workspace/nuplan-devkit/nuplan
        # img_path = f"/home/workspace/nuplan-devkit/docs/images/{scenario.token}_{self._iteration}.png"
        # self._vis(temporal_feature, vis_pack, predictions, img_path)
        

        # convert to absolute
        y_hat = predictions["trajectory"][temporal_feature['av_index'],:, :2].unsqueeze(0)
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