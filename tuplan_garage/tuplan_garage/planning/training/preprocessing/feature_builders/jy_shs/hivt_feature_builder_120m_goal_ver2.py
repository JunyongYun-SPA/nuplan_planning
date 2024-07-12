from __future__ import annotations

from typing import List, Optional, Tuple, Type, Dict

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    TimeDuration,
    TimePoint,
)
from nuplan.planning.metrics.utils.state_extractors import (
    extract_ego_acceleration,
    extract_ego_yaw_rate,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.scenario_utils import (
    sample_indices_with_time_horizon,
)
from nuplan.planning.simulation.history.simulation_history_buffer import (
    SimulationHistoryBuffer,
)
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import (
    SimulationIteration,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    build_ego_features,
)
from shapely.geometry import Point

from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import (
    PDMClosedPlanner,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    StateIndex,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from tuplan_garage.planning.training.preprocessing.features.hivt_feature import (
    HiVTFeature, TemporalData #JY
)
## 추가 라이브러리 ,BH
from nuplan.common.maps.maps_datatypes import (SemanticMapLayer,
                                                TrafficLightStatusData,
                                                TrafficLightStatusType,)
from nuplan.common.actor_state.state_representation import Point2D
import torch
from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_graph_map_feature_builder_utils import (
    calculate_lane_progress,
    convert_absolute_to_relative_array,
    points_in_polygons,
)
from itertools import product
from typing import List, Tuple, Type, cast
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from itertools import permutations
from shapely.geometry import Polygon


class HiVTFeatureBuilder(AbstractFeatureBuilder): #JY
    """Feature builder class for PDMOpen and PDMOffset."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        history_sampling: TrajectorySampling,
        planner: Optional[PDMClosedPlanner],
        centerline_samples: int = 120,
        centerline_interval: float = 1.0,
    ):
        """
        Constructor for PDMFeatureBuilder
        :param history_sampling: dataclass for storing trajectory sampling
        :param centerline_samples: number of centerline poses
        :param centerline_interval: interval of centerline poses [m]
        :param planner: PDMClosed planner for correction
        """
        assert (
            type(planner) == PDMClosedPlanner or planner is None
        ), f"PDMFeatureBuilder: Planner must be PDMClosedPlanner or None, but got {type(planner)}"
        self.LANE_LAYER = [SemanticMapLayer.LANE,
                            SemanticMapLayer.LANE_CONNECTOR,
                            SemanticMapLayer.INTERSECTION]
        self._trajectory_sampling = trajectory_sampling
        self._history_sampling = history_sampling
        self._centerline_samples = centerline_samples
        self._centerline_interval = centerline_interval

        self._planner = planner

    @classmethod
    def get_feature_type(cls):
        """Type of the built feature."""
        return Dict #JY

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "pdm_features"

    def get_features_from_scenario(self, scenario: AbstractScenario): #JY
        """Inherited, see superclass."""

        past_ego_states = [
            ego_state
            for ego_state in scenario.get_ego_past_trajectory(
                iteration=0,
                time_horizon=self._history_sampling.time_horizon,
                num_samples=self._history_sampling.num_poses,
            )
        ] + [scenario.initial_ego_state]

        current_input, initialization = self._get_planner_params_from_scenario(scenario)

        return self._compute_feature(past_ego_states, current_input, initialization, scenario)

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ): #JY
        """Inherited, see superclass."""

        history = current_input.history
        current_ego_state, _ = history.current_state
        past_ego_states = history.ego_states[:-1]

        indices = sample_indices_with_time_horizon(
            self._history_sampling.num_poses, self._history_sampling.time_horizon, history.sample_interval
        )
        past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)] + [
            current_ego_state
        ]

        return self._compute_feature(past_ego_states, current_input, initialization)

    def _get_planner_params_from_scenario(
        self, scenario: AbstractScenario
    ) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of nuPlan
        :return: tuple of planner input and initialization objects
        """

        buffer_size = int(2 / scenario.database_interval + 1)

        # Initialize Planner
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )

        history = SimulationHistoryBuffer.initialize_from_scenario(
            buffer_size=buffer_size,
            scenario=scenario,
            observation_type=DetectionsTracks,
        )

        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),
        )

        return planner_input, planner_initialization
    
    def lane_list2sampled_array(lanes, origin, rotate_mat, sampling_unit=10):
        arr_ = np.array([point.array for index, point in enumerate(lanes) if index % 5 ==0])

    def _compute_feature(
        self,
        ego_states: List[EgoState],
        current_input: PlannerInput,
        initialization: PlannerInitialization,
        scenario: AbstractScenario,
    ): #JY
        """
        Creates PDMFeature dataclass based in ego history, and planner input
        :param ego_states: list of ego states
        :param current_input: planner input of current frame
        :param initialization: planner initialization of current frame
        :return: PDMFeature dataclass
        """
        ALL_ego_states = [
            ego_state
            for ego_state in scenario.get_ego_past_trajectory(
                iteration=0,
                time_horizon=2.0,
                num_samples=10,
            )
        ] + [scenario.initial_ego_state] +[
            ego_state 
            for ego_state in scenario.get_ego_future_trajectory(
                iteration=0,
                time_horizon=8.0,
                num_samples=16,
            )
        ]
        
        ALL_observation = [
            obs
            for obs in scenario.get_past_tracked_objects(
                iteration=0,
                time_horizon=2.0,
                num_samples=10,
            )
        ] + [scenario.initial_tracked_objects] +[
            obs 
            for obs in scenario.get_future_tracked_objects(
                iteration=0,
                time_horizon=8.0,
                num_samples=16,
            )
        ]
        
        
        current_ego_state: EgoState = scenario.initial_ego_state
        current_pose: StateSE2 = current_ego_state.center
        
        actor_ids = ['ego']
        vehicle_tracks = self.filter_tracked_objects_by_type(ALL_observation, TrackedObjectType.VEHICLE)
        # vehicle_agent_feats, vehicle_agent_masks = self._get_surrounding_agents_representation(vehicle_tracks, ego_states, time_stamps)
        for s_c in vehicle_tracks:
            # if num_nodes < len(s_c.tracked_objects.tracked_objects):
            #     num_nodes = len(s_c.tracked_objects.tracked_objects)
            for agent in s_c.tracked_objects.tracked_objects:
                actor_ids.append(agent.track_token)

        actor_ids = list(set(actor_ids))
        num_nodes = len(actor_ids)
        av_index = actor_ids.index('ego')
        origin = torch.tensor(list(ego_states[-1].center.array), dtype=torch.float)
        # av_heading_vectors = []
        # for i in range(1,6):
        #     av_heading_vectors.append(torch.tensor(list(ego_states[-1-i].center.array), dtype=torch.float).unsqueeze(0))
        # av_heading_vector = torch.cat(av_heading_vectors, 0)
        # av_heading_vector = origin - av_heading_vector
        # av_heading_vector = av_heading_vector.mean(0)
        theta = torch.tensor(current_pose.heading) # global
        theta = theta.type(torch.float32)
        rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                               [torch.sin(theta), torch.cos(theta)]])
    
        # initialization
        x = torch.zeros(num_nodes, 27, 3, dtype=torch.float) # 9, 10, 16, arr[:10]-> hist, arr[10]-> current, arr[10:16] -> fut
        for index, ego_state in enumerate(ALL_ego_states):
            x[av_index, index, 0] = ego_state.center.x
            x[av_index, index, 1] = ego_state.center.y
            x[av_index, index, 2] = ego_state.center.heading
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        padding_mask = torch.ones(num_nodes, 27, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
        bos_mask = torch.zeros(num_nodes, 11, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
        # Vehicle의 경우
        for step, s_c in enumerate(vehicle_tracks):
            for agent in s_c.tracked_objects.tracked_objects:
                node_idx = actor_ids.index(agent.track_token)
                if node_idx == av_index:
                    raise ValueError
                padding_mask[node_idx, step] = False
                x[node_idx, step, :] = torch.Tensor([agent.center.x, agent.center.y, agent.center.heading])

        # AV의 경우
        padding_mask[av_index, :] = False

        for i in range(len(actor_ids)):
            if padding_mask[i, 10]:  # make no predictions for actors that are unseen at the current time step
                    padding_mask[i, 11:] = True
                
        valid_idx = np.where(padding_mask == False)
        origin_x = x.clone()
        x[valid_idx[0], valid_idx[1], :2] = torch.matmul(x[valid_idx[0], valid_idx[1], :2] - origin, rotate_mat)
        x[valid_idx[0], valid_idx[1], 2] = x[valid_idx[0], valid_idx[1], 2] - theta
        x[valid_idx[0], valid_idx[1], 2] = x[valid_idx[0], valid_idx[1], 2] % (2 * np.pi) # positive heading(0~2n)
        for index_ in range(len(valid_idx[0])):
            ad_pi =  x[valid_idx[0][index_], valid_idx[1][index_], 2].item()
            if ad_pi > np.pi:
                x[valid_idx[0][index_], valid_idx[1][index_], 2] = ad_pi - (2*np.pi)
            else:
                x[valid_idx[0][index_], valid_idx[1][index_], 2] = ad_pi

        radius = 120
        
        node_inds_ = []
        last_obs = vehicle_tracks[10]
        for agent in last_obs.tracked_objects.tracked_objects:
            node_inds_.append(agent.track_token)
        node_inds_.append('ego')
        node_inds = [actor_ids.index(node_ind) for node_ind in node_inds_]
        node_positions = origin_x[node_inds, 10, :2]
        
        ### run planner
        self._planner.initialize(initialization)
        trajectory: InterpolatedTrajectory = self._planner.compute_planner_trajectory(current_input)

        # extract planner centerline
        centerline: PDMPath = self._planner._centerline
        current_progress: float = centerline.project(Point(*current_pose.array))
        centerline_progress_values = (
            np.arange(radius, dtype=np.float64)
            * self._centerline_interval
            + current_progress
        )  # distance values to interpolate
        planner_centerline = convert_absolute_to_relative_se2_array(
            current_pose,
            centerline.interpolate(centerline_progress_values, as_array=True),
        )  # convert to relative coords
        
        current_ego_global_pose: Point2D = Point2D(current_pose.x, current_pose.y)
        LANE_obj = initialization.map_api.get_proximal_map_objects(current_ego_global_pose, radius, self.LANE_LAYER)
        LANES = LANE_obj[list(LANE_obj.keys())[0]]
        LANE_CONS = LANE_obj[list(LANE_obj.keys())[1]]
        lane_ = {"Lane":[], "Lane_polygon":[], "neightbor_Lane_with_road_block":[], "local_Lane_polygon":[],
                 "Lane_cons":[], "Lane_cons_polygon":[], "neightbor_cons_with_road_block":[], "local_cons_polygon":[],}
        lane_["Lane"] = [Lane.id for Lane in LANES]
        lane_["Lane_polygon"] = [Lane.polygon for Lane in LANES]
        for poly in lane_["Lane_polygon"]:
            x_ = list(poly.exterior.xy[0])
            y_ = list(poly.exterior.xy[1])
            arr_ = torch.tensor([x_, y_]).T
            local_arr_ = torch.matmul(arr_ - origin, rotate_mat)
            local_arr_ = local_arr_.numpy()
            polygon = Polygon(local_arr_)
            lane_["local_Lane_polygon"].append(polygon)
        lane_["neightbor_Lane_with_road_block"] = [[lane_.id for lane_ in Lane.parent.interior_edges] for Lane in LANES]
        lane_["Lane_cons"] = [Lane_con.id for Lane_con in LANE_CONS]
        lane_["Lane_cons_polygon"] = [Lane_con.polygon for Lane_con in LANE_CONS]
        lane_["neightbor_cons_with_road_block"] =  [[lane_.id for lane_ in Lane_con.parent.interior_edges] for Lane_con in LANE_CONS]
        for poly in lane_["Lane_cons_polygon"]:
            x_ = list(poly.exterior.xy[0])
            y_ = list(poly.exterior.xy[1])
            arr_ = torch.tensor([x_, y_]).T
            local_arr_ = torch.matmul(arr_ - origin, rotate_mat)
            local_arr_ = local_arr_.numpy()
            polygon = Polygon(local_arr_)
            lane_["local_cons_polygon"].append(polygon)
        
        
        end_lanes = {"Lane_ids":[], "Lane_con_ids":[], "goal_point":[]}
        for centerline_index, point in enumerate(planner_centerline[:,:2].copy()[::-1]):
            line_point = np.expand_dims(point[:2], axis=0)
            index = np.where(points_in_polygons(line_point, lane_["local_Lane_polygon"])==True)[0]
            if len(index) != 0:
                poly_index = index[0]
                end_lanes["Lane_ids"] = lane_["neightbor_Lane_with_road_block"][poly_index]
                end_lanes["goal_point"] = line_point
                break
            index = np.where(points_in_polygons(line_point, lane_["local_cons_polygon"])==True)[0]
            if len(index) != 0:
                poly_index = index[0]
                end_lanes["Lane_con_ids"] = lane_["neightbor_cons_with_road_block"][poly_index]
                end_lanes["goal_point"] = line_point
                break
            #np.array(points_in_polygons(lane_centerline, intersection_polygons))
        
        
        (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
         lane_actor_vectors, lane_positions, goals) = self.get_lane_features(current_input, initialization.map_api, node_inds, node_positions, origin, rotate_mat, radius, end_lanes, av_index=actor_ids.index('ego'))
        
        # # # visualization
        # import matplotlib.pyplot as plt
        # import matplotlib.patches as patches
        
        # cur_t = 11
        # hist_t = list(range(0, 12))
        # plt.figure(figsize=(26,26))
        # plt.xlim(-200, 200)
        # plt.ylim(-200, 200)
        
        # agent_w = 5
        # agent_h = 2
        
        
        # lane_x = lane_positions[:, 0]   
        # lane_y = lane_positions[:, 1]
        # plt.scatter(lane_x, lane_y, s = 25, c="gray", alpha=0.4)
        
        # for node_idx in range(padding_mask.shape[0]):
        #     if padding_mask[node_idx, cur_t] == True:
        #         continue
        #     else:
        #         valid_hist_t = [t_ for t_ in hist_t if padding_mask[node_idx, t_] == False]
        #         agent_pos = x[node_idx, valid_hist_t]
        #         pos_xs = agent_pos[:, 0]
        #         pos_ys = agent_pos[:, 1]
        #         pos_yaws = agent_pos[:, 2]
        #         for index_ in range(pos_xs.shape[0]):
        #             pos_x = pos_xs[index_]
        #             pos_y = pos_ys[index_]
        #             pos_yaw = pos_yaws[index_]
        #             if node_idx == av_index:
        #                 rect = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
        #                         edgecolor='red', facecolor='red', alpha = 1/(pos_xs.shape[0]-index_))
        #             else:
        #                 rect = patches.Rectangle((pos_x - agent_w / 2, pos_y - agent_h / 2), agent_w, agent_h, linewidth=1,
        #                         edgecolor='blue', facecolor='blue', alpha = 1/(pos_xs.shape[0]-index_))
        #             ax = plt.gca()
        #             t = patches.transforms.Affine2D().rotate_around(pos_x, pos_y, pos_yaw) + ax.transData
        #             rect.set_transform(t)
        #             ax.add_patch(rect)
                    
        # center_x = planner_centerline[:, 0]   
        # center_y = planner_centerline[:, 1]   
        # plt.scatter(center_x, center_y, s = 25, c="Cyan", alpha=0.8)
                        
        # goal_index = np.where(goals==1)[0]
        # lane_x = lane_positions[goal_index, 0]
        # lane_y = lane_positions[goal_index, 1]
        # plt.scatter(lane_x, lane_y, s = 200, c="green", alpha=1)
        
        
        # plt.savefig(f"/home/workspace/pictures/{scenario.token}.PNG")
        
        
        
        rotate_angles = x[:, 10, 2]

        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1: 11] = padding_mask[:, : 10] & ~padding_mask[:, 1:11]

        positions = x.clone()
        x[:, 11:, :] = torch.where((padding_mask[:, 10].unsqueeze(-1) | padding_mask[:, 11:]).unsqueeze(-1),
                                torch.zeros(num_nodes, 16, 3),
                                x[:, 11:, :] - x[:, 10, :].unsqueeze(-2))
        x[:, 1: 11, :] = torch.where((padding_mask[:, : 10] | padding_mask[:, 1: 11]).unsqueeze(-1),
                                torch.zeros(num_nodes, 10, 3),
                                x[:, 1: 11, :] - x[:, : 10, :])
        x[:, 0, :] = torch.zeros(num_nodes, 3)

        y = x[:, 11:]
        
        # extract ego vehicle history states
        ego_position = get_ego_position(ego_states)
        ego_velocity = get_ego_velocity(ego_states)
        ego_acceleration = get_ego_acceleration(ego_states)

        result = {
            'x':x, # x.shape ~ [15, 27, 3]
            'positions':positions, # positions.shape ~ [15, 27, 3]
            'edge_index':edge_index,
            'y': y, # y.shape(15, 16, 3)
            'num_nodes':num_nodes, #JY
            'padding_mask':padding_mask, #(48, 27)
            'bos_mask':bos_mask,
            'rotate_angles':rotate_angles,
            'lane_vectors': lane_vectors,  # [L, 2]
            'is_intersections': is_intersections,  # [L]
            'turn_directions': turn_directions,  # [L]
            'traffic_controls': traffic_controls,  # [L]
            'lane_actor_index': lane_actor_index,  # [2, E_{A-L}] # (2, 4765)
            'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2] # (4765, 2)
            'seq_id':scenario.token,
            'av_index':av_index,
            'agent_index':av_index,
            'city':scenario._map_name,
            'origin':origin,
            'theta':theta,
            'goal': goals
        }
        
        return result
        
    def get_lane_features(self, current_input, map_api, node_inds, node_positions, origin, rot_mat, radius, end_lanes, av_index):
        lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls, goals, lane_centerlines = [], [], [], [], [], [], []
        lane_ids = set()
        lane_con_ids = set()
        intersection_ids = set()
        for node_ind, node_position in zip(node_inds, node_positions):
            position: Point2D = Point2D(node_position[0].item(), node_position[1].item())
            if node_ind == av_index:
                LANE_obj = map_api.get_proximal_map_objects(position, radius, self.LANE_LAYER)
            else:
                LANE_obj = map_api.get_proximal_map_objects(position, 50, self.LANE_LAYER)
            LANES = LANE_obj[list(LANE_obj.keys())[0]]
            LANE_CONS = LANE_obj[list(LANE_obj.keys())[1]]
            INTERSECTIONS = LANE_obj[list(LANE_obj.keys())[2]]
            all_ids = [Lane.id for Lane in LANES]
            lane_ids.update(all_ids)
            all_ids = [Lane_con.id for Lane_con in LANE_CONS]
            lane_con_ids.update(all_ids)
            all_ids = [intersect.id for intersect in INTERSECTIONS]
            intersection_ids.update(all_ids)
        node_positions = torch.matmul(node_positions - origin, rot_mat).float()
        
        intersection_polygons = []
        traffic_lane_con_ids = [data.lane_connector_id for data in current_input.traffic_light_data]
        for intersection_id in intersection_ids:
            intersection_objs = map_api._get_intersection(intersection_id)
            intersection_polygons.append(intersection_objs.polygon)
            
        for lane_id in lane_ids:
            lane_obj = map_api._get_lane(lane_id)
            # lane_obj = map_api._get_lane_connector(lane_id)
            lane_centerline_ = lane_obj.baseline_path.discrete_path
            if len(lane_centerline_) < 10:
                continue
            target_indices = np.linspace(0, len(lane_centerline_) - 1, 10, dtype=int)
            lane_centerline_ = [lane_centerline_[index] for index in target_indices]
            lane_centerline = torch.tensor([[point.x, point.y] for point in lane_centerline_])
            is_intersection = False
            lane_centerline = torch.matmul(lane_centerline - origin, rot_mat)
            turn_direction = 0 # 직접적으로 turn direction을 알아낼 방법이 없음
            traffic_control = False
            lane_positions.append(lane_centerline[:-1])
            lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
            count = len(lane_centerline) - 1
            is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
            turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
            traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
            if lane_id in end_lanes["Lane_ids"]:
                sp_lane = False * torch.ones(count, dtype=torch.uint8)
                sp_lane[-1] = True
                goals.append(sp_lane)
            else:
                goals.append(False * torch.ones(count, dtype=torch.uint8))
            
        for lane_id in lane_con_ids:
            lane_obj = map_api._get_lane_connector(lane_id)
            # lane_obj = map_api._get_lane_connector(lane_id)
            lane_centerline_ = lane_obj.baseline_path.discrete_path
            if len(lane_centerline_) < 10:
                continue
            target_indices = np.linspace(0, len(lane_centerline_) - 1, 10, dtype=int)
            lane_centerline_ = [lane_centerline_[index] for index in target_indices]
            lane_centerline = torch.tensor([[point.x, point.y] for point in lane_centerline_])
            is_intersection = np.any(np.array(points_in_polygons(lane_centerline, intersection_polygons)))
            lane_centerline = torch.matmul(lane_centerline - origin, rot_mat)
            if lane_id in traffic_lane_con_ids:
                traffic_control = True 
            else:
                traffic_control = False 
            turn_direction = 0 # 직접적으로 turn direction을 알아낼 방법이 없음
            lane_positions.append(lane_centerline[:-1])
            lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
            count = len(lane_centerline) - 1
            is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
            turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
            traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
            if lane_id in end_lanes["Lane_con_ids"]:
                sp_lane = False * torch.ones(count, dtype=torch.uint8)
                sp_lane[-1] = True
                goals.append(sp_lane)
            else:
                goals.append(False * torch.ones(count, dtype=torch.uint8))
        lane_positions = torch.cat(lane_positions, dim=0)
        lane_vectors = torch.cat(lane_vectors, dim=0)
        is_intersections = torch.cat(is_intersections, dim=0)
        turn_directions = torch.cat(turn_directions, dim=0)
        traffic_controls = torch.cat(traffic_controls, dim=0)
        goals = torch.cat(goals,dim=0)
        
        lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
        lane_actor_vectors = lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
        mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
        lane_actor_index = lane_actor_index[:, mask]
        lane_actor_vectors = lane_actor_vectors[mask]
        
        return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors, lane_positions, goals

    def filter_tracked_objects_by_type(self, tracked_objects: List[DetectionsTracks], object_type: TrackedObjectType) -> List[DetectionsTracks]:
            return [
                DetectionsTracks(
                    TrackedObjects(p.tracked_objects.get_tracked_objects_of_type(object_type))
                )
                for p in tracked_objects
            ]
    
    
def get_ego_position(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Creates an array of relative positions (x, y, θ)
    :param ego_states: list of ego states
    :return: array of shape (num_frames, 3)
    """
    ego_poses = build_ego_features(ego_states, reverse=True)
    return ego_poses


def get_ego_velocity(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Creates an array of ego's velocities (v_x, v_y, v_θ)
    :param ego_states: list of ego states
    :return: array of shape (num_frames, 3)
    """
    v_x = np.asarray(
        [ego_state.dynamic_car_state.center_velocity_2d.x for ego_state in ego_states]
    )
    v_y = np.asarray(
        [ego_state.dynamic_car_state.center_velocity_2d.y for ego_state in ego_states]
    )
    v_yaw = extract_ego_yaw_rate(ego_states)
    return np.stack([v_x, v_y, v_yaw], axis=1)


def get_ego_acceleration(ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """
    Creates an array of ego's accelerations (a_x, a_y, a_θ)
    :param ego_states: list of ego states
    :return: array of shape (num_frames, 3)
    """
    a_x = extract_ego_acceleration(ego_states, "x")
    a_y = extract_ego_acceleration(ego_states, "y")
    a_yaw = extract_ego_yaw_rate(ego_states, deriv_order=2, poly_order=3)
    return np.stack([a_x, a_y, a_yaw], axis=1)
