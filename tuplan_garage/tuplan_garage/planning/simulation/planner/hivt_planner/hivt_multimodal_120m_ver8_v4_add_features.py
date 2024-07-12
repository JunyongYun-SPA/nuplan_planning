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

# BH ì¶”ê°€ import
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

        self._device = "cpu"
        model.device = "cpu"
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
        
        # ì²«ë²ˆì§¸ í™”ë©´ êµ¬ì„±
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
                    t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + plot_fig.transData
                    rect1.set_transform(t)
                    plot_fig.add_patch(rect1)

                    # Nonfiltered_by_roadblock ê°ì²´ì— ëŒ€í•´ íŒ¨ì¹˜ ì¶”ê°€
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

        # radius í‚¤ì›Œì£¼ê¸°
        planner_map_radius = 120
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
        predictions = self._model.forward({"pdm_features": temporal_feature})
        
        # ë©€í‹°ëª¨ë‹¬ ì‹¤í—˜
        pi = predictions['pi']
        pi = pi[temporal_feature['av_index'], :].float()
        softmax = nn.Softmax(dim=0)
        pi = softmax(pi) # 
        
        pi_sort_index = torch.argsort(pi) # [0]ì´ë©´ ì œì¼ ì•ˆì¢‹ì€ê±° [-1]ì´ë©´ ì œì¼ ì¢‹ì€ê±°
        y_hats = predictions["trajectory"][pi_sort_index,temporal_feature['av_index'], ...] # (6, 66, 16, 3)
        pi = pi[pi_sort_index]
        
        tmp_velocity = temporal_feature["velocity_for_occupancy"][:, 10, :] # ~ (66, 2)
        tmp_prediction = np.zeros([len(temporal_feature['obs_ids_for_occupancy']), 17, 3])
        # df_prediction[:, 0, :] = temporal_feature['global_vehicle_for_occupancy'][:, 10, :]
        global_state = temporal_feature['global_vehicle_for_occupancy'][:, 10, :]
        tmp_predictions = predictions["trajectory"][pi_sort_index, ...][-1, ...]
        valid_mask = temporal_feature["bos_mask"][:,0]
        observe_id_list = temporal_feature['obs_ids_for_occupancy']
        
        # 0513 ìˆ˜ì •, ì£¼ë³€ agentë„ ê±°ì˜ ë©ˆì¶°ìžˆë„ë¡ ì˜ˆì¸¡ë˜ì—ˆë‹¤ë©´ GC-PGP ì‚¬ìš©
        for index_ in range(len(temporal_feature['obs_ids_for_occupancy'])):
            if temporal_feature["bos_mask"][index_, 0] == False:
                continue
            single_traj = tmp_predictions[index_, ...].detach().cpu()
            move_distance = torch.sum(torch.norm(single_traj[1:, :2]-single_traj[:-1, :2],dim=1))
            if move_distance < 3:
                current_ego_vel = np.array([torch.norm(tmp_velocity[index_]).item()])
                single_traj = waypoints_to_trajectory(single_traj[:, :2].unsqueeze(0), current_ego_vel).detach()[0] # (16, 3)

            tmp_prediction[index_, 1:, :] = single_traj.numpy()
        # observation map ë§Œë“¤ê¸°
        ego_state, observation = current_input.history.current_state
        self._observation.update(ego_state,
                                observation,
                                current_input.traffic_light_data,
                                self._route_lane_dict)
        # occ_maps = self._observation._occupancy_maps
        state_array = np.zeros((6, 17, 11))
        state_array[:, 0] = ego_state_to_state_array(ego_state)
        for modal in range(6):
            traj_w_heading = y_hats[modal, ...].detach().cpu()
            # 0513 ê°œì„  ì‹¤í—˜ 1. ì •ì§€ì— ê°€ê¹Œìš¸ ê²½ìš° GC-PGP Method ì‚¬ìš©
            move_distance = torch.sum(torch.norm(traj_w_heading[1:, :2]-traj_w_heading[:-1, :2],dim=1))
            if move_distance < 3:
                current_ego_vel = np.array([current_input.history.current_state[0].dynamic_car_state.speed])
                traj_w_heading = waypoints_to_trajectory(traj_w_heading[:, :2].unsqueeze(0), current_ego_vel).detach()[0] # (16, 3)
            traj_w_heading = traj_w_heading.numpy()
            # current_ego_vel = np.array([current_input.history.current_state[0].dynamic_car_state.speed])
            # traj_w_heading = waypoints_to_trajectory(single_traj, current_ego_vel).detach().numpy()[0] # (16, 3)
            # ê¸€ë¡œë²Œ ì¢Œí‘œë¡œ ë³€í™˜
            gx, gy, gheading = state_array[modal, 0, 0], state_array[modal, 0, 1], state_array[modal, 0, 2]
            traj_w_heading_p = traj_w_heading[:, :2]
            rot_mat = np.array([[np.cos(gheading), -np.sin(gheading)],
                                [np.sin(gheading), np.cos(gheading)]])
            # ê²½ë¡œ íšŒì „ í›„, ê¸€ë¡œë²Œ ì¢Œí‘œë¡œ ì´ë™ì‹œí‚´
            traj_w_heading_g_p = np.matmul(traj_w_heading_p, rot_mat.T) + np.array([[gx, gy]])
            # headingì€ ë”í•´ì¤Œ
            traj_w_heading_g_h = traj_w_heading[:, 2] + gheading
            state_array[modal, 1:, :2] = traj_w_heading_g_p
            state_array[modal, 1:, 2] =  traj_w_heading_g_h
            # velocity êµ¬í•¨
            vel = state_array[modal, 1:, 0:2] - state_array[modal, :-1, 0:2]
            vel = vel[1:] * 2
            state_array[modal, 1:16, 3:5] = vel
            # Acc êµ¬í•¨
            acc = state_array[modal, 1:, 3:5] - state_array[modal, :-1, 3:5]
            acc = acc[1:] * 2
            state_array[modal, 1:16, 5:7] = acc
            # Steering ê´€ë ¨í•´ì„œëŠ” êµ¬í•˜ëŠ” ë°©ë²•ë„ ëª¨ë¥´ê² ê³ , metricì—ì„œ ì‚¬ìš©ë˜ëŠ” ë¶€ë¶„ì„ ëª» ì°¾ìŒ
            # ê°ì†ë„
            ang_vel = state_array[modal, 1:, 2] - state_array[modal, :-1, 2]
            ang_vel = ang_vel[1:] * 2
            state_array[modal, 1:16, 9] = ang_vel
            # ê° ê°€ì†ë„
            ang_acc = state_array[modal, 1:, 9] - state_array[modal, :-1, 9]
            ang_acc = ang_acc[1:] * 2
            state_array[modal, 1:16, 10] = ang_acc
            
        # GT Global AV Traj
        iteration = current_input.iteration.index
        future_gGT = np.array([[ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading] for ego_state in scenario.get_ego_future_trajectory(iteration=iteration, time_horizon=8.0, num_samples=16)])
        cur_gGT = scenario.get_ego_state_at_iteration(iteration=iteration)
        cur_gGT = np.array([cur_gGT.rear_axle.x, cur_gGT.rear_axle.y, cur_gGT.rear_axle.heading])
        
        # 5. Score proposals
        # modalë³„ë¡œ ìŠ¤ì½”ì–´ëŠ” ë‚¼ ìˆ˜ ìžˆê²Œ ë˜ê¸´ í–ˆëŠ”ë° ì´ê±¸ ì–´ë–»ê²Œ í‘œí˜„í•´ì•¼ í• ì§€ëŠ” ì¢€ ê³ ë¯¼í•´ë³´ê² ìŒ
        state_array = state_array[:, :9, :]#state 4ì´ˆë¡œ ìžë¥´ê¸°
        proposal_scores, combi_test = self._scorer.score_proposals(
            state_array,
            ego_state,
            self._observation,
            self._centerline,
            self._route_lane_dict,
            self._drivable_area_map,
            self._map_api,
        )

        proposal_scores_best_mode = torch.argmax(softmax(torch.tensor(proposal_scores)) *100000 + pi)

        GT = temporal_feature['y'][temporal_feature['av_index']][-1, :2].unsqueeze(0)
        
        best_mode = proposal_scores_best_mode
        y_hat = y_hats[best_mode, ...].unsqueeze(0)
        
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
        GT_trajectory = temporal_feature['y'][temporal_feature['av_index']][:, :3].detach().cpu().numpy()
        GT_trajectory = InterpolatedTrajectory(
            transform_predictions_to_states(
                GT_trajectory,
                current_input.history.ego_states,
                self._model.trajectory_sampling.time_horizon,
                self._model.trajectory_sampling.step_time,
            )
        )
        # ìºì‹œ íŒŒì¼ ê¸°ë¡ìš©
        cache = {}
        cache["GT"] = temporal_feature['y'][temporal_feature['av_index']][:, :2]
        cache["vis_pack"] = vis_pack
        cache["av_index"] = temporal_feature["av_index"]
        cache["future_gGT"] = future_gGT
        cache["cur_gGT"] = cur_gGT
        cache["GT_trajectory"] = GT_trajectory
        import pickle
        import os
        with open(f"/home/workspace/nuplanv2_0528/GT_traj/{scenario.token}_{self._iteration}.pickle","wb") as fw:
            pickle.dump(cache, fw)
     
        self._iteration += 1
        return trajectory, 0