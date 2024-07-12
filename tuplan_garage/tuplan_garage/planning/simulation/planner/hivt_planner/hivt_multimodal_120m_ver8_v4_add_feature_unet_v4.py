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
from tuplan_garage.planning.simulation.planner.hivt_planner.utils.hivt_120m_goal_feature_utils_ver10 import (
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
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_observation import (
    PDMObservation,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
)
from scipy.signal import savgol_filter
import os

from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import DistanceDropEdge, DistanceDropEdgeOtherAgents

warnings.filterwarnings("ignore", category=RuntimeWarning)

logger = logging.getLogger(__name__)


class HivtMultimodal(AbstractPDMPlanner):
    """PDM-Open planner class."""

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        model: TorchModuleWrapper,
        proposal_sampling: TrajectorySampling,
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

        self._device = "cuda"
        model.device = "cuda"
        self._model = LightningModuleWrapper.load_from_checkpoint(
            checkpoint_path,
            model=model,
            map_location=self._device,
        ).model

        
        self._proposal_sampling = proposal_sampling
        # self._model = self._model.to(self._device)

        self._model.eval()
        torch.set_grad_enabled(False)

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._iteration = 0
        self._map_api = initialization.map_api
        self._load_route_dicts(initialization.route_roadblock_ids)
        self.initialization = initialization
        self._observation = PDMObservation(self._model.trajectory_sampling, self._proposal_sampling, map_radius=50, observation_sample_res=1)
        self._scorer = PDMScorer(self._proposal_sampling)

        gc.collect()

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def _vis(self, temporal_feature, vis_pack, predictions, img_path, selected_mode_):
        ### visualization
        # visualization
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        cur_t = 11
        hist_t = list(range(0, 11))
        fut_t = list(range(11, 27))
        fig = plt.figure(figsize=(17,17))
        
        # 첫번째 화면 구성
        plot_fig = fig.add_subplot(1, 1, 1)
        draw_range = 90 #torch.max(torch.max(vis_pack["y_hat_"]), torch.max(vis_pack["GT"])) + 20
        plot_fig.axis(xmin=-draw_range, xmax=draw_range)
        plot_fig.axis(ymin=-draw_range, ymax=draw_range)
        padding_mask = vis_pack["padding_mask"]
        x = vis_pack["x"]
        av_index = temporal_feature["av_index"]
        lane_positions = vis_pack["lane_positions"]
        agent_w = 5
        agent_h = 2
        
        lane_x = lane_positions[:, 0]
        lane_y = lane_positions[:, 1]
        plot_fig.scatter(lane_x, lane_y, s = 25, c="black", alpha=0.2, zorder=1)
        for local_route_ in vis_pack['local_roadblock']:
            x_, y_ = local_route_.exterior.xy
            plot_fig.fill(x_, y_, color='lightgreen', alpha=0.12)
        
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
                        rect1 = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                                edgecolor='red', facecolor='red', alpha = 1/(pos_xs.shape[0]-index_))
                    else:
                        rect1 = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                                edgecolor='blue', facecolor='blue', alpha = 1/(pos_xs.shape[0]-index_))
                    # ax = filtered_by_roadblock.gca()
                    # t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + ax.transData
                    # rect.set_transform(t)
                    # ax.add_patch(rect)
                    # ax = plot_fig.gca()
                    # t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + ax.transData
                    # rect.set_transform(t)
                    # ax.add_patch(rect)
                    # filtered_by_roadblock 객체에 대해 패치 추가
                    t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + plot_fig.transData
                    rect1.set_transform(t)
                    plot_fig.add_patch(rect1)

                    # Nonfiltered_by_roadblock 객체에 대해 패치 추가
                    # t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + Nonfiltered_by_roadblock.transData
                    # rect2.set_transform(t)
                    # Nonfiltered_by_roadblock.add_patch(rect2)
                    
                
        valid_fut_t = [t_ for t_ in fut_t if padding_mask[av_index, t_] == False]

        # multimodal_traj_color = ['royalblue', 'skyblue','purple',"teal", 'brown', 'orange']
        selected_traj_color = ['orange']
        other_traj_color = ['skyblue']
        
        
        GT_pos = x[av_index, valid_fut_t]
        GT_last_p = GT_pos[-1, :2].norm().item()
        pi_pos = vis_pack["y_hat_"][vis_pack["pi_best_mode"].item(), ...]
        pi_last_p = pi_pos[-1, :2].norm().item()
        propose_pos = vis_pack["y_hat_"][vis_pack["proposal_score_best_mode"].item(), ...]
        propose_pos_last_p = propose_pos[-1, :2].norm().item()
        sorter = np.argsort([GT_last_p, pi_last_p, propose_pos_last_p])
        zorders = [0, 0, 0, 4]
        for index, sorter_ in enumerate(sorter):
            zorders[sorter_] = [6,5,4][index]


        for i in range(6):
            if i == vis_pack["pi_best_mode"].item():
                alpha = 1
                tmp_index = 1
            elif i == vis_pack['proposal_score_best_mode'].item():
                alpha = 1
                tmp_index = 2
            else:
                alpha = 0
                tmp_index = 0
            prediction = vis_pack["y_hat_"][i, ...]
            pos_xs = prediction[:, 0].detach().numpy()
            pos_ys = prediction[:, 1].detach().numpy()
            # if vis_pack["intraj_count"][i] > 8:
            #     filtered_by_roadblock.plot(pos_xs, pos_ys, linewidth=10, c=multimodal_traj_color[i], zorder=3, label=str(round(vis_pack["pi_"][i].item(),4)))  
            #     filtered_by_roadblock.scatter(pos_xs, pos_ys, s = 80, c=multimodal_traj_color[i], edgecolor='black', zorder=3)
            pi_i = round(vis_pack["pi_"][i].item(),4)
            score_i = round(vis_pack["scores"][i].item(),4)
            collision = round(vis_pack["combi_test"]['[0]'][0][i], 2)
            dr_direction = round(vis_pack["combi_test"]['[1]'][0][i], 2)
            dr_area = round(vis_pack["combi_test"]['[2]'][0][i], 2)
            progress = min(round(vis_pack["combi_test"]['[3]'][i],2)*12/5, 1.0)
            ttc = min(round(vis_pack["combi_test"]['[4]'][i],2)*12/5, 1.0)
            comfortable = min(round(vis_pack["combi_test"]['[5]'][i],2)*12/2, 1.0)
            if selected_mode_ == i:
                tmp_c = 'orange'
            else:
                tmp_c = "skyblue"
            plot_fig.plot(pos_xs, pos_ys, linewidth=10, c=tmp_c, zorder=zorders[tmp_index], label=f"{pi_i:.2f}/{score_i:.2f}[{collision:.2f}, {dr_direction:.2f}, {dr_area:.2f}, {progress:.2f}, {ttc:.2f}, {comfortable:.2f}]", alpha=alpha)   
            plot_fig.scatter(pos_xs, pos_ys, s = 80, c=tmp_c, edgecolor='black', zorder=zorders[tmp_index], alpha=alpha)
        
        plot_fig.legend()
        #Nonfiltered_by_roadblock.legend()
        
        agent_pos = x[av_index, valid_fut_t]
        pos_xs = agent_pos[:, 0].numpy()
        pos_ys = agent_pos[:, 1].numpy()
        pos_yaws = agent_pos[:, 2].numpy()
        plot_fig.plot(pos_xs, pos_ys, linewidth=10, c='red', zorder=zorders[0])   
        plot_fig.scatter(pos_xs, pos_ys, s = 80, c="red", edgecolor='black', alpha=1.0, zorder=zorders[0])
        # Nonfiltered_by_roadblock.plot(pos_xs, pos_ys, linewidth=10, c='red', zorder=3)   
        # Nonfiltered_by_roadblock.scatter(pos_xs, pos_ys, s = 80, c="red", edgecolor='black', alpha=1.0, zorder=3)
        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        plt.clf()
        plt.close("all")

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

        # radius 키워주기
        planner_map_radius = 120
        # # Update/Create drivable area polygon map
        # self._drivable_area_map = get_drivable_area_map(
        #     self._map_api, ego_state, self._map_radius
        # )
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, planner_map_radius
        )
        

        # Create centerline
        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))
        centerline_lane_objects = self._route_plan
        
        pdm_feature, vis_pack = create_hivt_feature(
            self._model, current_input, self._centerline, centerline_lane_objects, current_lane, None, self._model.device, scenario=scenario, map_api=self._map_api, initialization = self.initialization
        )
        temporal_feature = TemporalData(**pdm_feature)
        # torch.save(temporal_feature, cache_path)
        self._model.to("cuda")
        predictions = self._model.forward({"pdm_features": temporal_feature})
        # predictions.to("cpu")
        self._model.to("cpu")
        predictions["trajectory"] = predictions["trajectory"].to("cpu")
        predictions["pi"] = predictions["pi"].to("cpu")
        predictions["occupancy_map"] = predictions["occupancy_map"].to("cpu")
        # # 시각화
        # if self._iteration==0:
        #     print()
        #     img_path = f"/home/workspace/pictures/{scenario.token}_{self._iteration}.png"
        #     self._vis(temporal_feature, vis_pack, predictions, img_path)
        # else:
        #     raise ValueError
        ## 환경 OCC 설정
        occupancy_range = 100
        occupancy_resolution = 0.5
        occupancy_size = 200
        
        # 멀티모달 실험
        pi = predictions['pi']
        pi = pi[temporal_feature['av_index'], :].float()
        softmax = nn.Softmax(dim=0)
        pi = softmax(pi) # 
        
        pi_sort_index = torch.argsort(pi) # [0]이면 제일 안좋은거 [-1]이면 제일 좋은거
        y_hats = predictions["trajectory"][pi_sort_index,temporal_feature['av_index'], ...] # (6, 66, 16, 3)
        pi = pi[pi_sort_index]
        
        # # 그냥 처음부터 짜는 게 편할 거 같음
        # # Lane 맵 구성
        # input: HiVTFeature = temporal_feature
        # input = input.to("cpu")
        # occupancy_map_env = torch.zeros(1, 1, occupancy_size+1, occupancy_size+1, dtype=torch.float32)
        # drop_edge_av = DistanceDropEdge(occupancy_range/2)
        # input['edge_attr'] = input['positions'][input['edge_index'][0], 10, :2] - input['positions'][input['edge_index'][1], 10, :2]
        # edge_index, _ = drop_edge_av(input['edge_index'], input['edge_attr'])
        # lane_edge_index, lane_attr_origin = drop_edge_av(input['lane_actor_index'], input['lane_actor_vectors'])


        # lane_mask = (lane_edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
        # lane_mask_index = torch.where(lane_mask ==True)[0] #torch.Size([1419])
        # lanes_indx = lane_edge_index[1][lane_mask_index]
        # lane_attr = lane_attr_origin[lane_mask_index] + input.positions[lanes_indx, 10, :2]

        # # batch_idx_for_lane = lanes_indx.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
        # # batch_idx_for_lane = torch.where((batch_idx_for_lane >= 0), 1, 0).sum(dim=-1)
        # batch_idx_for_lane = torch.zeros(lane_attr.shape[0], dtype=torch.int64) #torch.where((batch_idx_for_lane > 0), 1, 0).sum(dim=-1)
        
        # occ_mask_lane = (abs(lane_attr[:, 0]) < 50) * (abs(lane_attr[:, 1]) < 50)
        # lane = torch.where(((occ_mask_lane).unsqueeze(-1)),
        #                         lane_attr,
        #                         50*torch.ones(lane_attr.shape[0], 2).to(lane_attr.device))
        # occupancy_x_lane, occupancy_y_lane = \
        #             (lane / occupancy_resolution).type(torch.int)[:, 0], \
        #                 (lane / occupancy_resolution).type(torch.int)[:, 1]
        # occupancy_y_lane[~(occ_mask_lane)] = \
        #     -1 * occupancy_y_lane[~(occ_mask_lane)]
        # occupancy_map_env[batch_idx_for_lane, \
        #     torch.arange(1).repeat(batch_idx_for_lane.shape[0]), \
        #         (occupancy_range - occupancy_y_lane).type(torch.long), \
        #             (occupancy_x_lane + occupancy_range).type(torch.long)] = 1

        # AV Trajectory 사용
        pred_map = torch.zeros(6, 16, occupancy_size+1, occupancy_size+1, dtype=torch.float32)
        box_pred_map = torch.zeros(6, 16, occupancy_size+1, occupancy_size+1, dtype=torch.float32)
        # print()
        pred_x_av, pred_y_av = (y_hats/ occupancy_resolution).type(torch.int)[..., 0], (y_hats / occupancy_resolution).type(torch.int)[..., 1]
        # 일단 확실하게 만들어 놓고 나중에 없애기
        for i in range(6):
            for j in range(16):
                tmp_y = (occupancy_range - pred_y_av[i,j])
                tmp_x = (occupancy_range + pred_x_av[i, j])
                if tmp_y >= 0 and tmp_y<=occupancy_range*2 and tmp_x>=0 and tmp_x<=occupancy_range*2:
                    pred_map[i, j, tmp_y, tmp_x] = 1
                    min_x = max(occupancy_range + pred_x_av[i, j]-1, 0)
                    max_x = min(occupancy_range + pred_x_av[i, j]+1, occupancy_range*2)
                    min_y = max(occupancy_range - pred_y_av[i, j]-1, 0)
                    max_y = min(occupancy_range - pred_y_av[i, j]+1, occupancy_range*2)
                    box_pred_map[i, j, min_y:max_y, min_x:max_x] = 1
        box_pred_map = box_pred_map[..., :occupancy_size, :occupancy_size]
        pred_map = pred_map[..., :occupancy_size, :occupancy_size]
        
        # box_pred_map 시각화
        # for i in range(6):
        #     for j in range(16):
        #         map_ = np.array(box_pred_map[i,j])*255 + 50
        #         from PIL import Image
        #         im = Image.fromarray(map_)
        #         im = im.convert('RGB')
        #         dir = f"/home/workspace/pictures_ver_occ/{i}modal/"
        #         os.makedirs(dir, exist_ok=True)
        #         im.save(dir+f"{i}modal_{j}step.png")
        
        # 주변 agent 예측
        sig = nn.Sigmoid()
        other_occ = predictions["occupancy_map"].repeat(6, 1, 1, 1)
        other_occ_ = sig(other_occ)
        other_occ_[other_occ_>=0.5] = 1
        other_occ_[other_occ_<0.5] = 0
        # 주변 agent를 박스로 예측
        # 일단 확실하게 만들어 놓고 나중에 없애기
        # mask__ = torch.where(other_occ_ == 1)
        # box_other_occ_ = torch.zeros_like(other_occ_)
        # for modal_, time_step, pred_y, pred_x in zip(mask__[0], mask__[1], mask__[2], mask__[3]):
        #     min_x = max(pred_x-1, 0)
        #     max_x = min(pred_x+1, occupancy_range*2)
        #     min_y = max(pred_y-1, 0)
        #     max_y = min(pred_y+1, occupancy_range*2)
        #     box_other_occ_[modal_, time_step, min_y:max_y, min_x:max_x] = 1
        # # box_other_map 시각화
        # for i in range(6):
        #     for j in range(16):
        #         map_ = np.array(box_pred_map[i,j])*255 + 100
        #         map_ = map_ * (np.array(box_other_occ_[i,j])==0)*1
        #         from PIL import Image
        #         im = Image.fromarray(map_)
        #         im = im.convert('RGB')
        #         dir = f"/home/workspace/pictures_ver_occ/{scenario.token}_{self._iteration}/{i}modal/"
        #         os.makedirs(dir, exist_ok=True)
        #         im.save(dir+f"{i}modal_{j}_step.png")
        collision = other_occ_*pred_map 
        collision_ = collision.sum(axis=1)
        collision__ = collision_.sum(axis=1)
        collision___ = collision__.sum(axis=1)
        col_score = torch.zeros_like(collision___)
        col_score[collision___ == 0] = 1
        col_score = np.array(col_score)
        col_score = np.expand_dims(col_score, axis=0)
        
        # collision = box_other_occ_*box_pred_map
        # collision_ = collision.sum(axis=1)
        # collision__ = collision_.sum(axis=1)
        # collision___ = collision__.sum(axis=1)
        # box_col_score = torch.zeros_like(collision___)
        # box_col_score[collision___ == 0] = 1
        # box_col_score = np.array(box_col_score)
        # box_col_score = np.expand_dims(box_col_score, axis=0)
        
        
        # 아 쉽지 않겠는데
        # current(10) 정보를 이용하여 predictions를 global coords로 번환하기
        # temporal_feature["bos_mask"][:,0]
        # temporal_feature['obs_ids_for_occupancy']
        # temporal_feature['global_vehicle_for_occupancy'][:, 10, :]
        # predictions["trajectory"][pi_sort_index, ...][-1, ...] ~ (66, 16, 2)
        tmp_velocity = temporal_feature["velocity_for_occupancy"][:, 10, :] # ~ (66, 2)
        tmp_prediction = np.zeros([len(temporal_feature['obs_ids_for_occupancy']), 17, 3])
        # df_prediction[:, 0, :] = temporal_feature['global_vehicle_for_occupancy'][:, 10, :]
        global_state = temporal_feature['global_vehicle_for_occupancy'][:, 10, :]
        tmp_predictions = predictions["trajectory"][pi_sort_index, ...][-1, ...]
        valid_mask = temporal_feature["bos_mask"][:,0]
        observe_id_list = temporal_feature['obs_ids_for_occupancy']
        
        # 0513 수정, 주변 agent도 거의 멈춰있도록 예측되었다면 GC-PGP 사용
        for index_ in range(len(temporal_feature['obs_ids_for_occupancy'])):
            if temporal_feature["bos_mask"][index_, 0] == False:
                continue
            single_traj = tmp_predictions[index_, ...].detach().cpu()
            move_distance = torch.sum(torch.norm(single_traj[1:, :2]-single_traj[:-1, :2],dim=1))
            if move_distance < 3:
                current_ego_vel = np.array([torch.norm(tmp_velocity[index_]).item()])
                single_traj = waypoints_to_trajectory(single_traj[:, :2].unsqueeze(0), current_ego_vel).detach()[0] # (16, 3)
            
            # current_ego_vel = np.array([temporal_feature["velocity_for_occupancy"][index_, 10, :].norm().item()])
            # traj_w_heading = waypoints_to_trajectory(single_traj, current_ego_vel).detach().numpy()[0] # (16, 3)
            # gx, gy, gheading = global_prediction[index_, 0, 0], global_prediction[index_, 0, 1], global_prediction[index_, 0, 2]
            # traj_w_heading_p = traj_w_heading[:, :2]
            # rot_mat = np.array([[np.cos(gheading), -np.sin(gheading)], [np.sin(gheading), np.cos(gheading)]])
            # traj_w_heading_g_p = np.matmul(traj_w_heading_p, rot_mat.T) + np.array([[gx, gy]])
            # traj_w_heading_g_h = traj_w_heading[:, 2] + gheading
            tmp_prediction[index_, 1:, :] = single_traj.numpy()
        # observation map 만들기
        ego_state, observation = current_input.history.current_state
        self._observation.update_predict(ego_state,
                                        observation,
                                        current_input.traffic_light_data,
                                        self._route_lane_dict,
                                        tmp_prediction,
                                        valid_mask,
                                        observe_id_list,
                                        global_state)
        # occ_maps = self._observation._occupancy_maps
        state_array = np.zeros((6, 17, 11))
        state_array[:, 0] = ego_state_to_state_array(ego_state)
        for modal in range(6):
            traj_w_heading = y_hats[modal, ...].detach().cpu()
            # 0513 개선 실험 1. 정지에 가까울 경우 GC-PGP Method 사용
            move_distance = torch.sum(torch.norm(traj_w_heading[1:, :2]-traj_w_heading[:-1, :2],dim=1))
            if move_distance < 3:
                current_ego_vel = np.array([current_input.history.current_state[0].dynamic_car_state.speed])
                traj_w_heading = waypoints_to_trajectory(traj_w_heading[:, :2].unsqueeze(0), current_ego_vel).detach()[0] # (16, 3)
            traj_w_heading = traj_w_heading.numpy()
            # current_ego_vel = np.array([current_input.history.current_state[0].dynamic_car_state.speed])
            # traj_w_heading = waypoints_to_trajectory(single_traj, current_ego_vel).detach().numpy()[0] # (16, 3)
            # 글로벌 좌표로 변환
            gx, gy, gheading = state_array[modal, 0, 0], state_array[modal, 0, 1], state_array[modal, 0, 2]
            traj_w_heading_p = traj_w_heading[:, :2]
            rot_mat = np.array([[np.cos(gheading), -np.sin(gheading)],
                                [np.sin(gheading), np.cos(gheading)]])
            # 경로 회전 후, 글로벌 좌표로 이동시킴
            traj_w_heading_g_p = np.matmul(traj_w_heading_p, rot_mat.T) + np.array([[gx, gy]])
            # heading은 더해줌
            traj_w_heading_g_h = traj_w_heading[:, 2] + gheading
            state_array[modal, 1:, :2] = traj_w_heading_g_p
            state_array[modal, 1:, 2] =  traj_w_heading_g_h
            # velocity 구함
            vel = state_array[modal, 1:, 0:2] - state_array[modal, :-1, 0:2]
            vel = vel[1:] * 2
            state_array[modal, 1:16, 3:5] = vel
            # Acc 구함
            acc = state_array[modal, 1:, 3:5] - state_array[modal, :-1, 3:5]
            acc = acc[1:] * 2
            state_array[modal, 1:16, 5:7] = acc
            # Steering 관련해서는 구하는 방법도 모르겠고, metric에서 사용되는 부분을 못 찾음
            # 각속도
            ang_vel = state_array[modal, 1:, 2] - state_array[modal, :-1, 2]
            ang_vel = ang_vel[1:] * 2
            state_array[modal, 1:16, 9] = ang_vel
            # 각 가속도
            ang_acc = state_array[modal, 1:, 9] - state_array[modal, :-1, 9]
            ang_acc = ang_acc[1:] * 2
            state_array[modal, 1:16, 10] = ang_acc
            
        # GT Global AV Traj
        iteration = current_input.iteration.index
        gGT = [ego_state for ego_state in scenario.get_ego_future_trajectory(iteration=iteration, time_horizon=8.0, num_samples=16)]    
        
        # 4초 정도만 쓰는 게 좋을 거 같음(8초는 GT조차 부딪히는 경우도 발생함)
        # 해석이 좀 괴상함 CV에서는 5 Hz, 8초 경로 쓰는 거 맞다
        # IDM에서는 10Hz로 4s 동안 시뮬레이션을 진행한다. 코드가 좀 이상한데
        # 

        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        # import cv2
        # for modal in range(4,5):
        #     for occ_index, occ_map in enumerate(occ_maps):
        #         # Step1. plt 초기화
        #         orig_x = ego_state.center.x
        #         orig_y = ego_state.center.y
        #         plt.clf()
        #         plt.figure(figsize=(17,17), frameon=False)
        #         #GT
        #         if occ_index == 0:
        #             ego_poly = ego_state.car_footprint.geometry
        #             x__, y__ = ego_poly.exterior.xy
        #             plt.fill(x__,y__, color='blue')
        #             theta = ego_state.center.heading
        #             x__, y__ = ego_state.center.x, ego_state.center.y
        #             plt.arrow(x__, y__, 3*np.cos(theta), 3*np.sin(theta),  head_width=1, head_length=1.4, fc='g', ec='black', label='Heading')
        #         else:
        #             ego_poly = gGT[occ_index-1].car_footprint.geometry
        #             x__, y__ = ego_poly.exterior.xy
        #             plt.fill(x__,y__, color='blue')
        #             theta = gGT[occ_index-1].center.heading
        #             x__, y__ = gGT[occ_index-1].center.x, gGT[occ_index-1].center.y
        #             plt.arrow(x__, y__, 3*np.cos(theta), 3*np.sin(theta),  head_width=1, head_length=1.4, fc='g', ec='black', label='Heading')
        #         plt.xlim(orig_x-70, orig_x+70)
        #         plt.ylim(orig_y-70, orig_y+70)
        #         plt.axis('off')
        #         # 주변 객체
        #         for poly in occ_map._geometries:
        #             x__, y__ = poly.exterior.xy
        #             plt.fill(x__,y__, color='black')
        #         # pred    
        #         x__, y__ = state_array[modal, occ_index, :2]
        #         theta = state_array[modal, occ_index, 2]
        #         plt.scatter(x__,y__, color='red', s=50)
        #         plt.arrow(x__, y__, 3*np.cos(theta), 3*np.sin(theta),  head_width=1, head_length=1.4, fc='g', ec='black', label='Heading')
        #         img_path = f"/home/workspace/pictures/tmp/Modal{modal}_occ_{occ_index}.png"
        #         plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
        
        # state_array = state_array[:, :9, :]
        
        # 5. Score proposals
        # modal별로 스코어는 낼 수 있게 되긴 했는데 이걸 어떻게 표현해야 할지는 좀 고민해보겠음
        proposal_scores, combi_test = self._scorer.score_proposals(
            state_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )
        # print(proposal_scores)
        
        # 실험 두 개 돌려보겠음
        # 1. CLS 만으로 선택
        # 2. CLS + pi로 선택
        
        # if sum(proposal_scores) != 0:
        #     new_proposal_scores = softmax(torch.tensor(proposal_scores))
        #     cls_best_mode = torch.argmax(new_proposal_scores)
        # else:
        #     cls_best_mode = 5
        # best_mode = 5
        proposal_scores_best_mode = torch.argmax(softmax(torch.tensor(proposal_scores)) *100000 + pi)
        # proposal_scores_best_mode = torch.argmax(softmax(torch.tensor(proposal_scores)) *100000)
        pi_best_mode = torch.argmax(pi)

        GT = temporal_feature['y'][temporal_feature['av_index']][-1, :2].unsqueeze(0)
        # error_mat_pros = (y_hats[proposal_scores_best_mode, -1, :2].unsqueeze(0) - GT).norm()
        # error_mat_pi = (y_hats[pi_best_mode, -1, :2].unsqueeze(0) - GT).norm()
        # best_mode = pi_best_mode if error_mat_pros > error_mat_pi else proposal_scores_best_mode
        
        # 조건에따라 best mode를 취사 선택
        # pi가 정렬되어 있음
        # 이 조건이 너무 빡빡하거나 혹은 일부 충돌 조건만 사용해야 할수도? 어쨌든 분석
        # 0513 개선 실험 3. pi_best_mode랑 proposal_scores_best_mode가 다를 경우
    
        # if pi_best_mode != proposal_scores_best_mode:
        #     if proposal_scores[pi_best_mode] != 0:
        #         selected_best_mode = pi_best_mode
        #     else:
        #         # 0513 개선 실험 2. confidence_score 조건 완화
        #         # if pi[proposal_scores_best_mode] <= 0.1:
        #         if pi[proposal_scores_best_mode] <= pi[pi_best_mode]*0.2:
        #             selected_best_mode = pi_best_mode
        #         else:
        #             tmp_score = torch.tensor(proposal_scores) * pi
        #             selected_best_mode = torch.argmax(tmp_score)
        # else:
        #     selected_best_mode = pi_best_mode
        
        # collision_scores = combi_test['[0]']
        collision_scores = col_score
        dr_score = combi_test['[2]']
        new_score = torch.tensor(collision_scores)*pi*torch.tensor(dr_score)
        if torch.sum(new_score) != 0:
            best_mode = torch.argmax(new_score)
        else:
            best_mode = torch.argmax(pi)
        # print()
            
        
        # best_mode = torch.argmax(new_proposal_scores)
        # pi_best_mode = torch.argmax(pi)
        # y_hat_Fs = y_hats[:, -1, :2].unsqueeze(0).float()
        # GT_F = temporal_feature['y'][temporal_feature['av_index'], -1, :2].unsqueeze(0).float()
        # GT_best_mode = torch.argmin(torch.cdist(y_hat_Fs, GT_F))
        y_hat = y_hats[best_mode, ...].unsqueeze(0)
        
        # current_ego_vel = np.array([current_input.history.current_state[0].dynamic_car_state.speed])
        # traj_w_heading = waypoints_to_trajectory(y_hat, current_ego_vel).detach().numpy()[0]
        # 시각화
        
        # vis_pack["GT"] = GT
        # vis_pack["pi_"] = pi
        # vis_pack["y_hat_"] = y_hats
        # vis_pack["pi_best_mode"] = pi_best_mode
        # vis_pack["proposal_score_best_mode"] = selected_best_mode
        # vis_pack["pis"] = pi
        # vis_pack["scores"] = proposal_scores
        # vis_pack["combi_test"] = combi_test
        # if pi_best_mode.item() != selected_best_mode.item():
        #     os.makedirs(f"/home/workspace/pictures/idm_score_VS_multimodal_score_ver6/{scenario.scenario_type}_{scenario.token}/", exist_ok=True)
        #     img_path = f"/home/workspace/pictures/idm_score_VS_multimodal_score_ver6/{scenario.scenario_type}_{scenario.token}/{self._iteration}.png"
        #     self._vis(temporal_feature, vis_pack, y_hats, img_path, selected_best_mode)
        
        
        current_ego_vel = np.array([current_input.history.current_state[0].dynamic_car_state.speed])
        traj_w_heading = waypoints_to_trajectory(y_hat[..., :2], current_ego_vel).detach().numpy()[0]
        traj_w_heading[...,-1] = y_hat[...,-1].detach().numpy()
        traj_w_heading = traj_w_heading.reshape(16, 3)
            
        trajectory = InterpolatedTrajectory(
            transform_predictions_to_states(
                traj_w_heading,
                current_input.history.ego_states,
                self._model.trajectory_sampling.time_horizon,
                self._model.trajectory_sampling.step_time,
            )
        )
        
        # Score와는 별개로 조합 실험
        # 조합을 찾아보겠음
        # 1. 각 모달의 마지막 point와 GT의 final point간 거리 측정 
        # 2. 역수를 취한 후, softmax로 normalize, 이러면 거리가 가까울수록 값이 큼
        # 3. final score = (confidence score + softmax(CLS score))/2를 구함 이때 CLS score에서 다양한 조합을 시험하기
        # 4, final score * normalized FDE를 CLS score 조합별로 계산하고, 긍정적인 영향을 미치는 조합을 찾아보기
        # import pickle
        # GT = temporal_feature['y'][temporal_feature['av_index']][-1]
        # error_mat = (y_hats[:,-1] - GT).norm(dim=1)
        # normalized_GT_score = softmax(1/error_mat)
        # result = {}
        # for key_, val_ in zip(combi_test.keys(), combi_test.values()):
        #     tmp_result = softmax(torch.tensor(val_).squeeze(0))/2
        #     tmp_result_ = tmp_result * normalized_GT_score
        #     tmp_result_ = sum(tmp_result_)
        #     result[key_] = tmp_result_.item()
        # with open(f'/home/workspace/trash_bin/test2/{scenario.token}_{self._iteration}.pkl', 'wb') as f:
	    #     pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
     
     
        self._iteration += 1
        return trajectory, 0
