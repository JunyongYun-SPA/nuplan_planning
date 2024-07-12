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
from tuplan_garage.planning.simulation.planner.hivt_planner.utils.bh_hivt_120m_goal_feature_utils_ver10 import (
    create_hivt_feature,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

# BH 추가 import
from tuplan_garage.planning.training.preprocessing.features.hivt_feature import (
    HiVTFeature, TemporalData #JY
)
import numpy as np
import pickle

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


class HivtUnimodal(AbstractPDMPlanner):
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
        super(HivtUnimodal, self).__init__(map_radius)

        # self._device = "cpu"
        self._device = "cuda"
        model.device = "cuda"
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

    def _vis(self, temporal_feature, vis_pack, predictions, img_path, all_predict):
        ### visualization
        # visualization
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        cur_t = 11
        hist_t = list(range(0, 11))
        fut_t = list(range(11, 27))
        fig = plt.figure(figsize=(20,20))
        
        # ì²«ë²ˆì§¸ í™”ë©´ êµ¬ì„±
        plot_fig = fig.add_subplot(1, 1, 1)
        # plot_fig2 = fig.add_subplot(1, 2, 2)
        draw_range = 90 #torch.max(torch.max(vis_pack["y_hat_"]), torch.max(vis_pack["GT"])) + 20
        plot_fig.axis(xmin=-draw_range, xmax=draw_range)
        plot_fig.axis(ymin=-draw_range, ymax=draw_range)
        # plot_fig2.axis(xmin=-draw_range, xmax=draw_range)
        # plot_fig2.axis(ymin=-draw_range, ymax=draw_range)
        padding_mask = vis_pack["padding_mask"]
        x = vis_pack["x"]
        av_index = temporal_feature["av_index"]
        lane_positions = vis_pack["lane_positions"]
        agent_w = 5
        agent_h = 2
        
        lane_x = lane_positions[:, 0]
        lane_y = lane_positions[:, 1]
        plot_fig.scatter(lane_x, lane_y, s = 25, c="black", alpha=0.2, zorder=1)
        # plot_fig2.scatter(lane_x, lane_y, s = 25, c="black", alpha=0.2, zorder=1)
        for local_route_ in vis_pack['local_roadblock']:
            x_, y_ = local_route_.exterior.xy
            plot_fig.fill(x_, y_, color='lightgreen', alpha=0.30)
            # plot_fig2.fill(x_, y_, color='lightgreen', alpha=0.30)
        
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
                                edgecolor='red', facecolor='red', alpha = 1/(pos_xs.shape[0]-index_), zorder=10)
                        rect2 = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                                edgecolor='red', facecolor='red', alpha = 1/(pos_xs.shape[0]-index_), zorder=10)
                    else:
                        rect1 = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                                edgecolor='blue', facecolor='blue', alpha = 1/(pos_xs.shape[0]-index_))
                        rect2 = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
                                edgecolor='blue', facecolor='blue', alpha = 1/(pos_xs.shape[0]-index_))
                    # ax = filtered_by_roadblock.gca()
                    # t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + ax.transData
                    # rect.set_transform(t)
                    # ax.add_patch(rect)
                    # ax = plot_fig.gca()
                    # t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + ax.transData
                    # rect.set_transform(t)
                    # ax.add_patch(rect)
                    # filtered_by_roadblock ê°ì²´ì— ëŒ€í•´ íŒ¨ì¹˜ ì¶”ê°€
                    t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + plot_fig.transData
                    # t2 = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + plot_fig2.transData
                    rect1.set_transform(t)
                    # rect2.set_transform(t2)
                    plot_fig.add_patch(rect1)
                    # plot_fig2.add_patch(rect2)

                    # Nonfiltered_by_roadblock ê°ì²´ì— ëŒ€í•´ íŒ¨ì¹˜ ì¶”ê°€
                    # t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + Nonfiltered_by_roadblock.transData
                    # rect2.set_transform(t)
                    # Nonfiltered_by_roadblock.add_patch(rect2)
                    
                
        valid_fut_t = [t_ for t_ in fut_t if padding_mask[av_index, t_] == False]

        # multimodal_traj_color = ['royalblue', 'skyblue','purple',"teal", 'brown', 'orange']
        
        
        GT_pos = x[av_index, valid_fut_t]
        GT_last_p = GT_pos[-1, :2].norm().item()
        pi_pos = vis_pack["y_hat_"][vis_pack["pi_best_mode"].item(), ...]
        pi_last_p = pi_pos[-1, :2].norm().item()
        plot_fig.legend()
        # plot_fig2.legend()
        #Nonfiltered_by_roadblock.legend()
        
        # # agent_pos = x[av_index, valid_fut_t]
        # agent_pos = all_predict[5, av_index, ...]
        # agent_pos = agent_pos.detach().cpu()
        # pos_xs = agent_pos[:, 0].numpy()
        # pos_ys = agent_pos[:, 1].numpy()
        # pos_yaws = agent_pos[:, 2].numpy()
        # plot_fig.plot(pos_xs, pos_ys, linewidth=3, c='red', zorder=2)   
        # plot_fig.scatter(pos_xs, pos_ys, s = 80, c="red", edgecolor='black', alpha=0.0, zorder=2)
        
        for index in range(x.shape[0]):
            if padding_mask[index, 11] == True:
                continue
            if x[index, 10, 3] < 0.2:
                continue
            
            # mode selection
            predict_final_point = all_predict[:,index, ~padding_mask[index, 11:], :2][:, -1, :]
            y = temporal_feature["y"]
            #GT_final_point = x[index, ~padding_mask[index, :], :2][-1, :]
            GT_final_point = y[index, ~padding_mask[index, 11:], :2][-1, :]
            mode__ = torch.argmin(torch.norm((predict_final_point - GT_final_point), dim=1))
            for cur_mode in range(6):
                agent_pos = all_predict[cur_mode, index, ...]
                # agent_pos_ref = all_predict_ref[cur_mode, index, ...]
                cur_agent_loc = x[index, 10, :3]
                rot_mat = torch.tensor([[torch.cos(-cur_agent_loc[2]), -torch.sin(-cur_agent_loc[2])],
                                        [torch.sin(-cur_agent_loc[2]), torch.cos(-cur_agent_loc[2])]])
                agent_pos = (agent_pos[:, :2].type(torch.float32)@rot_mat + cur_agent_loc[:2])#
                agent_pos = agent_pos.detach().cpu()
                # agent_pos_ref = (agent_pos_ref[:, :2].type(torch.float32)@rot_mat + cur_agent_loc[:2])#
                # agent_pos_ref = agent_pos_ref.detach().cpu()
                pos_xs = agent_pos[~padding_mask[index, 11:], 0].numpy()
                pos_ys = agent_pos[~padding_mask[index, 11:], 1].numpy()

                pos_xs = np.concatenate((np.array([x[index,10, 0]]), pos_xs))
                pos_ys = np.concatenate((np.array([x[index,10, 1]]), pos_ys))
                
                # pos_xs_ref = agent_pos_ref[~padding_mask[index, 11:], 0].numpy()
                # pos_ys_ref = agent_pos_ref[~padding_mask[index, 11:], 1].numpy()

                # pos_xs_ref = np.concatenate((np.array([x[index,10, 0]]), pos_xs_ref))
                # pos_ys_ref = np.concatenate((np.array([x[index,10, 1]]), pos_ys_ref))
                
                # if pos_xs.shape[0]>8:
                #     pos_xs = pos_xs[:8]
                #     pos_ys = pos_ys[:8]
                if cur_mode == mode__:
                    alpha_=1.0
                else:
                    alpha_=0.2
                if index != av_index:
                    if cur_mode == mode__:
                        # pos_yaws = agent_pos[:, 2].numpy()
                        plot_fig.plot(pos_xs, pos_ys, linewidth=3, c='blue', zorder=2)   
                        plot_fig.scatter(pos_xs, pos_ys, s = 80, c="blue", edgecolor='black', alpha=0.0, zorder=2)
                        
                        # plot_fig2.plot(pos_xs, pos_ys, linewidth=3, c='blue', zorder=2)   
                        # plot_fig2.scatter(pos_xs, pos_ys, s = 80, c="blue", edgecolor='black', alpha=0.0, zorder=2)
                else:
                    if cur_mode != mode__:
                        # pos_yaws = agent_pos[:, 2].numpy()
                        plot_fig.plot(pos_xs, pos_ys, linewidth=3, c='red', zorder=4, alpha=alpha_)   
                        # plot_fig.scatter(pos_xs, pos_ys, s = 80, c="red", edgecolor='black', alpha=0.0, zorder=2)
                        # plot_fig2.plot(pos_xs_ref, pos_ys_ref, linewidth=3, c='red', zorder=4, alpha=alpha_)   
                        # plot_fig2.scatter(pos_xs_ref, pos_ys_ref, s = 80, c="red", edgecolor='black', alpha=0.0, zorder=2)
                    if cur_mode == mode__:
                        plot_fig.plot(pos_xs, pos_ys, linewidth=3, c='red', zorder=6, alpha=alpha_)   
                        # plot_fig.scatter(pos_xs, pos_ys, s = 80, c="red", edgecolor='black', alpha=0.0, zorder=2)
                        # plot_fig2.plot(pos_xs_ref, pos_ys_ref, linewidth=3, c='red', zorder=6, alpha=alpha_)   
                        # plot_fig2.scatter(pos_xs_ref, pos_ys_ref, s = 80, c="red", edgecolor='black', alpha=0.0, zorder=2)
                        GT_traj = y[av_index, :, :2]
                        GT_traj = np.concatenate((np.array([[0, 0]]), GT_traj))
                        plot_fig.plot(GT_traj[:, 0], GT_traj[:, 1], linewidth=4, c='green', zorder=5, alpha=alpha_)
                        # plot_fig2.plot(GT_traj[:, 0], GT_traj[:, 1], linewidth=4, c='green', zorder=5, alpha=alpha_)  
                    
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

        # Update/Create drivable area polygon map
        self._drivable_area_map = get_drivable_area_map(
            self._map_api, ego_state, self._map_radius
        )

        # Create centerline
        current_lane = self._get_starting_lane(ego_state)
        self._centerline = PDMPath(self._get_discrete_centerline(current_lane))
        centerline_lane_objects = self._route_plan
        
        # pdm_feature, vis_pack = create_hivt_feature(
        #     self._model, current_input, self._centerline, None, self._model.device, scenario=scenario, map_api=self._map_api, initialization = self.initialization
        # )
        pdm_feature, vis_pack = create_hivt_feature(
            self._model, current_input, self._centerline, centerline_lane_objects, current_lane, None, self._model.device, scenario=scenario, map_api=self._map_api, initialization = self.initialization
        )
        temporal_feature = TemporalData(**pdm_feature)
        # torch.save(temporal_feature, cache_path)
        self._model.device = 'cpu'
        predictions = self._model.forward({"pdm_features": temporal_feature})
        # predictions["trajectory"] = predictions["trajectory"].to("cpu")
        # predictions["pi"] = predictions["pi"].to("cpu")
        
        pi = predictions['pi']
        pi = pi[temporal_feature['av_index'], :].float()
        softmax = nn.Softmax(dim=0)
        pi = softmax(pi) # 
        
        pi_sort_index = torch.argsort(pi) # [0]ì´ë©´ ì œì¼ ì•ˆì¢‹ì€ê±° [-1]ì´ë©´ ì œì¼ ì¢‹ì€ê±°
        y_hats = predictions["trajectory"][pi_sort_index,temporal_feature['av_index'], ...] # (6, 66, 16, 3)
        pi = pi[pi_sort_index]

        vis_pack["origin_pi"] = predictions['pi']
        # vis_pack["origin_pi_ref"] =  predictions['pi_ref']
        vis_pack["origin_traj"] = predictions["trajectory"]
        # vis_pack["origin_traj_ref"] = predictions["trajectory_ref"]
        vis_pack["av_index"] = temporal_feature['av_index']
        vis_pack["pi_"] = pi
        vis_pack["y_hat_"] = y_hats
        vis_pack["pi_best_mode"] = torch.argmax(pi)
        vis_pack["pis"] = pi

        import os
        os.makedirs(f"/home/workspace/tuplan_garage/tuplan_garage/planning/simulation/planner/hivt_planner/hivt_multimodal_vis/{scenario.token}/", exist_ok=True)
        img_path = f"/home/workspace/tuplan_garage/tuplan_garage/planning/simulation/planner/hivt_planner/hivt_multimodal_vis/{scenario.token}/{self._iteration}.png"
        self._vis(temporal_feature, vis_pack, y_hats, img_path, predictions["trajectory"][pi_sort_index, ...]) #, predictions["trajectory_ref"][pi_sort_index, ...])
        
        y_hat = predictions["trajectory"][:,temporal_feature['av_index'],:, :3].detach()
        
        # y_hat_xy = predictions["trajectory"][:,temporal_feature['av_index'],:, :2]
        
        # y_hat_Fs = y_hat_xy[:, -1, :]
        # GT_F = temporal_feature['y'][temporal_feature['av_index'], -1, :2].unsqueeze(0).cpu()
        # pi_best_mode = torch.argmin(torch.cdist(y_hat_Fs.to(torch.float32), GT_F.to(torch.float32)))
        
        pi = predictions['pi']
        pi = pi[temporal_feature['av_index'], :]
        pi_best_mode = torch.argmax(pi)
        
        y_hat = y_hat[pi_best_mode, ...].unsqueeze(0)

        current_ego_vel = np.array([current_input.history.current_state[0].dynamic_car_state.speed])
        traj_w_heading = waypoints_to_trajectory(y_hat[..., :2], current_ego_vel).detach().numpy()[0]
        traj_w_heading[...,-1] = y_hat[...,-1]
        traj_w_heading = traj_w_heading.reshape(16, 3)
            
        trajectory = InterpolatedTrajectory(
            transform_predictions_to_states(
                traj_w_heading,
                current_input.history.ego_states,
                self._model.trajectory_sampling.time_horizon,
                self._model.trajectory_sampling.step_time,
            )
        )
        
        # GT_trajectory = temporal_feature['y'][temporal_feature['av_index']][:, :3].detach().cpu().numpy()
        # GT_trajectory = InterpolatedTrajectory(
        #     transform_predictions_to_states(
        #         GT_trajectory,
        #         current_input.history.ego_states,
        #         self._model.trajectory_sampling.time_horizon,
        #         self._model.trajectory_sampling.step_time,
        #     )
        # )
        # # ìºì‹œ íŒŒì¼ ê¸°ë¡ìš©
        # cache = {}
        # cache["GT_trajectory"] = GT_trajectory
        # import pickle
        # import os
        # with open(f"/home/workspace/nuplanv2_0528/GT_traj/{scenario.token}_{self._iteration}.pickle","wb") as fw:
        #     pickle.dump(cache, fw)

        self._iteration += 1
        return trajectory, 0
