from typing import List, Optional

import numpy as np
from nuplan.common.actor_state.state_representation import TimeDuration, TimePoint
from nuplan.planning.scenario_builder.scenario_utils import (
    sample_indices_with_time_horizon,
)
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from shapely.geometry import Point

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
from tuplan_garage.planning.training.preprocessing.feature_builders.pdm_feature_builder import (
    get_ego_acceleration,
    get_ego_position,
    get_ego_velocity,
)
from tuplan_garage.planning.training.preprocessing.features.pdm_feature import (
    PDMFeature,
)
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from itertools import permutations, product
import torch
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import (SemanticMapLayer,
                                                TrafficLightStatusData,
                                                TrafficLightStatusType,)
from tuplan_garage.planning.training.preprocessing.feature_builders.pgp.pgp_graph_map_feature_builder_utils import (
    calculate_lane_progress,
    convert_absolute_to_relative_array,
    points_in_polygons,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    PlannerInitialization,
    PlannerInput,
)
from nuplan.common.actor_state.state_representation import (
    StateSE2,
    TimeDuration,
    TimePoint,
)
from nuplan.planning.metrics.utils.state_extractors import approximate_derivatives

## ì¶”ê°€ ë¼ì´ë¸ŒëŸ¬ë¦¬ ,BH
from nuplan.common.maps.maps_datatypes import (SemanticMapLayer,
                                                TrafficLightStatusData,
                                                TrafficLightStatusType,)
from nuplan.common.actor_state.state_representation import Point2D
import torch
from itertools import product
from typing import List, Tuple, Type, cast
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from itertools import permutations
from nuplan.common.actor_state.ego_state import EgoState
from shapely.geometry import Polygon
    
def filter_tracked_objects_by_type(self, tracked_objects: List[DetectionsTracks], object_type: TrackedObjectType) -> List[DetectionsTracks]:
        return [
            DetectionsTracks(
                TrackedObjects(p.tracked_objects.get_tracked_objects_of_type(object_type))
            )
            for p in tracked_objects
        ]

# def get_lane_features(current_input, map_api, node_inds, node_positions, origin, rot_mat, radius, end_lanes):
#     lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls, goals, lane_centerlines = [], [], [], [], [], [], []
#     lane_ids = set()
#     lane_con_ids = set()
#     intersection_ids = set()
#     LANE_LAYER = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.INTERSECTION]
#     for node_ind, node_position in zip(node_inds, node_positions):
#         position: Point2D = Point2D(node_position[0].item(), node_position[1].item())
#         LANE_obj = map_api.get_proximal_map_objects(position, radius, LANE_LAYER)
#         LANES = LANE_obj[list(LANE_obj.keys())[0]]
#         LANE_CONS = LANE_obj[list(LANE_obj.keys())[1]]
#         INTERSECTIONS = LANE_obj[list(LANE_obj.keys())[2]]
#         all_ids = [Lane.id for Lane in LANES]
#         lane_ids.update(all_ids)
#         all_ids = [Lane_con.id for Lane_con in LANE_CONS]
#         lane_con_ids.update(all_ids)
#         all_ids = [intersect.id for intersect in INTERSECTIONS]
#         intersection_ids.update(all_ids)
#     node_positions = torch.matmul(node_positions - origin, rot_mat).float()
    
#     intersection_polygons = []
#     traffic_lane_con_ids = [data.lane_connector_id for data in current_input.traffic_light_data]
#     for intersection_id in intersection_ids:
#         intersection_objs = map_api._get_intersection(intersection_id)
#         intersection_polygons.append(intersection_objs.polygon)
        
#     for lane_id in lane_ids:
#         lane_obj = map_api._get_lane(lane_id)
#         # lane_obj = map_api._get_lane_connector(lane_id)
#         lane_centerline_ = lane_obj.baseline_path.discrete_path
#         if len(lane_centerline_) < 10:
#             continue
#         target_indices = np.linspace(0, len(lane_centerline_) - 1, 10, dtype=int)
#         lane_centerline_ = [lane_centerline_[index] for index in target_indices]
#         lane_centerline = torch.tensor([[point.x, point.y] for point in lane_centerline_])
#         is_intersection = False
#         lane_centerline = torch.matmul(lane_centerline - origin, rot_mat)
#         turn_direction = 0 # ì§ì ‘ì ìœ¼ë¡œ turn directionì„ ì•Œì•„ë‚¼ ë°©ë²•ì´ ì—†ìŒ
#         traffic_control = False
#         lane_positions.append(lane_centerline[:-1])
#         lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
#         count = len(lane_centerline) - 1
#         is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
#         turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
#         traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
#         if lane_id in end_lanes["Lane_ids"]:
#             sp_lane = False * torch.ones(count, dtype=torch.uint8)
#             sp_lane[-1] = True
#             goals.append(sp_lane)
#         else:
#             goals.append(False * torch.ones(count, dtype=torch.uint8))
        
#     for lane_id in lane_con_ids:
#         lane_obj = map_api._get_lane_connector(lane_id)
#         # lane_obj = map_api._get_lane_connector(lane_id)
#         lane_centerline_ = lane_obj.baseline_path.discrete_path
#         if len(lane_centerline_) < 10:
#             continue
#         target_indices = np.linspace(0, len(lane_centerline_) - 1, 10, dtype=int)
#         lane_centerline_ = [lane_centerline_[index] for index in target_indices]
#         lane_centerline = torch.tensor([[point.x, point.y] for point in lane_centerline_])
#         is_intersection = np.any(np.array(points_in_polygons(lane_centerline, intersection_polygons)))
#         lane_centerline = torch.matmul(lane_centerline - origin, rot_mat)
#         if lane_id in traffic_lane_con_ids:
#             traffic_control = True 
#         else:
#             traffic_control = False 
#         turn_direction = 0 # ì§ì ‘ì ìœ¼ë¡œ turn directionì„ ì•Œì•„ë‚¼ ë°©ë²•ì´ ì—†ìŒ
#         lane_positions.append(lane_centerline[:-1])
#         lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
#         count = len(lane_centerline) - 1
#         is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
#         turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
#         traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
#         if lane_id in end_lanes["Lane_con_ids"]:
#             sp_lane = False * torch.ones(count, dtype=torch.uint8)
#             sp_lane[-1] = True
#             goals.append(sp_lane)
#         else:
#             goals.append(False * torch.ones(count, dtype=torch.uint8))
#     lane_positions = torch.cat(lane_positions, dim=0)
#     lane_vectors = torch.cat(lane_vectors, dim=0)
#     is_intersections = torch.cat(is_intersections, dim=0)
#     turn_directions = torch.cat(turn_directions, dim=0)
#     traffic_controls = torch.cat(traffic_controls, dim=0)
#     goals = torch.cat(goals,dim=0)
    
#     lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
#     lane_actor_vectors = lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
#     mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
#     lane_actor_index = lane_actor_index[:, mask]
#     lane_actor_vectors = lane_actor_vectors[mask]
    
#     return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors, lane_positions, goals
def findRoute(route_list, next_roadblock, route_end_idx_list, route_candidate_list, isFindRoute):
    if next_roadblock.id in route_candidate_list:
        route_list.append(next_roadblock.id)
        if next_roadblock.id in route_end_idx_list:
            isFindRoute = True
            return isFindRoute, route_list
        for next_roadblock in next_roadblock.outgoing_edges:
            if next_roadblock.id in route_list:
                return isFindRoute, route_list
            isFindRoute, route_list2 = findRoute(route_list.copy(), next_roadblock, route_end_idx_list, route_candidate_list, isFindRoute)
            if isFindRoute:
                return isFindRoute, route_list2
        return isFindRoute, route_list
    else:
        return isFindRoute, route_list

def get_lane_features(current_input, map_api, node_inds, node_positions, origin, rot_mat, radius, end_lanes,start_lanes, centerline_lanes):
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls, goals, lane_centerlines, in_route_lanes, in_centerline_lanes  = [], [], [], [], [], [], [], [], []
    lane_ids = set()
    lane_con_ids = set()
    intersection_ids = set()
    ROADBLOCK_POLYGON = set()
    LANE_LAYER = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.INTERSECTION]
    for node_ind, node_position in zip(node_inds, node_positions):
        position: Point2D = Point2D(node_position[0].item(), node_position[1].item())
        LANE_obj = map_api.get_proximal_map_objects(position, radius, LANE_LAYER)
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
        turn_direction = 0 # ì§ì ‘ì ìœ¼ë¡œ turn directionì„ ì•Œì•„ë‚¼ ë°©ë²•ì´ ì—†ìŒ
        traffic_control = False
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
        # goal
        if lane_id in end_lanes["Lane_ids"]:
            sp_lane = False * torch.ones(count, dtype=torch.uint8)
            sp_lane[-1] = True
            goals.append(sp_lane)
        else:
            goals.append(False * torch.ones(count, dtype=torch.uint8))
        # in route_lanes
        if lane_id in end_lanes["route_lane_ids"]:
            if lane_id in start_lanes["Lane_ids"]:
                mask_start_index = torch.argmin(lane_centerline[:-1].norm(dim=1)).item()
                sp_lane = False * torch.ones(count, dtype=torch.uint8)
                sp_lane[mask_start_index:] = True
                # sp_lane[:] = True
                in_route_lanes.append(sp_lane)
            else:
                in_route_lanes.append(True * torch.ones(count, dtype=torch.uint8))
        else:
            in_route_lanes.append(False * torch.ones(count, dtype=torch.uint8))
        # centerline
        #         centerline_lanes["Lane_ids"] = list(set(centerline_lanes["Lane_ids"]))
        # centerline_lanes["Lane_con_ids"] = list(set(centerline_lanes["Lane_con_ids"]))
        if lane_id in centerline_lanes["Lane_ids"]:
            if lane_id in centerline_lanes['start_lane_id']:
                mask_start_index = torch.argmin(lane_centerline[:-1].norm(dim=1)).item()
                sp_lane = False * torch.ones(count, dtype=torch.uint8)
                sp_lane[mask_start_index:] = True
                in_centerline_lanes.append(sp_lane)
            else:
                in_centerline_lanes.append(True * torch.ones(count, dtype=torch.uint8))
        else:
            in_centerline_lanes.append(False * torch.ones(count, dtype=torch.uint8))
            
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
        turn_direction = 0 # ì§ì ‘ì ìœ¼ë¡œ turn directionì„ ì•Œì•„ë‚¼ ë°©ë²•ì´ ì—†ìŒ
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
        # goal
        if lane_id in end_lanes["Lane_con_ids"]:
            sp_lane = False * torch.ones(count, dtype=torch.uint8)
            sp_lane[-1] = True
            goals.append(sp_lane)
        else:
            goals.append(False * torch.ones(count, dtype=torch.uint8))
        # in route_lanes
        if lane_id in end_lanes["route_con_ids"]:
            in_route_lanes.append(True * torch.ones(count, dtype=torch.uint8))
        else:
            in_route_lanes.append(False * torch.ones(count, dtype=torch.uint8))
        # centerline
        if lane_id in centerline_lanes["Lane_con_ids"]:
            if lane_id in centerline_lanes['start_lane_id']:
                mask_start_index = torch.argmin(lane_centerline[:-1].norm(dim=1)).item()
                sp_lane = False * torch.ones(count, dtype=torch.uint8)
                sp_lane[mask_start_index:] = True
                in_centerline_lanes.append(sp_lane)
            else:
                in_centerline_lanes.append(True * torch.ones(count, dtype=torch.uint8))
        else:
            in_centerline_lanes.append(False * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)
    goals = torch.cat(goals,dim=0)
    in_route_lanes = torch.cat(in_route_lanes, dim=0)
    in_centerline_lanes = torch.cat(in_centerline_lanes, dim=0)
    
    
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]
    
    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors, lane_positions, goals, in_route_lanes, in_centerline_lanes

def filter_tracked_objects_by_type(tracked_objects: List[DetectionsTracks], object_type: TrackedObjectType) -> List[DetectionsTracks]:
        return [
            DetectionsTracks(
                TrackedObjects(p.tracked_objects.get_tracked_objects_of_type(object_type))
            )
            for p in tracked_objects
        ]

def create_hivt_feature(
    model: TorchModuleWrapper,
    planner_input: PlannerInput,
    centerline: PDMPath,
    centerline_lane_objects,
    current_lane, 
    closed_loop_trajectory: Optional[InterpolatedTrajectory] = None,
    device: str = "cpu",
    scenario = None,
    map_api = None,
    initialization=None,
):
    """
    Creates a PDMFeature (for PDM-Open and PDM-Offset) during simulation
    :param model: torch model (used to retrieve parameters)
    :param planner_input: nuPlan's planner input during simulation
    :param centerline: centerline path of PDM-* methods
    :param closed_loop_trajectory: trajectory of PDM-Closed (ignored if None)
    :return: PDMFeature dataclass
    """
    time_stamps = [t for t in scenario.get_past_timestamps(iteration=0,num_samples=10,time_horizon=2.0)] + [scenario.start_time]
    
    # trainingì´ëž‘ ë™ì¼í•˜ê²Œ ê°€ì ¸ê°€ëŠ” ì „ëžµ
    # ì¶”í›„ scenarioì— ëŒ€í•œ ì˜ì¡´ë„ë¥¼ ì—†ì• ì•¼ í• ê²ƒ
    iteration = planner_input.iteration.index
    # print(iteration)
    num_past_poses = model.history_sampling.num_poses
    past_time_horizon = model.history_sampling.time_horizon
    
    history_indicies = [i for i in range(len(planner_input.history)) if i%2 == 1]
    planner_input.history.ego_states[-1] = planner_input.history.current_state[0]
    ALL_ego_states = [
        ego_state
        for index, ego_state in enumerate(planner_input.history.ego_states) if index in history_indicies] +[
        ego_state 
        for ego_state in scenario.get_ego_future_trajectory(
            iteration=iteration,
            time_horizon=8.0,
            num_samples=16,
        )
    ]

    ALL_observation = [
        obs
        for index, obs in enumerate(planner_input.history.observations) if index in history_indicies] +[
        obs 
        for obs in scenario.get_future_tracked_objects(
            iteration=iteration,
            time_horizon=8.0,
            num_samples=16,
        )
    ]

    current_ego_state: EgoState = planner_input.history.ego_states[-1]
    current_pose: StateSE2 = current_ego_state.center

    actor_ids = ['ego']
    vehicle_tracks = filter_tracked_objects_by_type(ALL_observation, TrackedObjectType.VEHICLE)
    observation_tokens = []
    # vehicle_agent_feats, vehicle_agent_masks = self._get_surrounding_agents_representation(vehicle_tracks, ego_states, time_stamps)
    for s_c in vehicle_tracks:
        # if num_nodes < len(s_c.tracked_objects.tracked_objects):
        #     num_nodes = len(s_c.tracked_objects.tracked_objects)
        for agent in s_c.tracked_objects.tracked_objects:
            actor_ids.append(agent.track_token)

    actor_ids = list(set(actor_ids))
    obs_ids_for_occupancy = actor_ids.copy()
    num_nodes = len(actor_ids)
    av_index = actor_ids.index('ego')
    origin = torch.tensor(list(current_ego_state.center.array), dtype=torch.float)
    # av_heading_vectors = []
    # for i in range(1,6):
    #     av_heading_vectors.append(torch.tensor(list(ego_states[-1-i].center.array), dtype=torch.float).unsqueeze(0))
    # av_heading_vector = torch.cat(av_heading_vectors, 0)
    # av_heading_vector = origin - av_heading_vector
    # av_heading_vector = av_heading_vector.mean(0)
    theta = torch.tensor(current_pose.heading)
    theta = theta.type(torch.float32)
    rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                            [torch.sin(theta), torch.cos(theta)]])

    # initialization
    x = torch.zeros(num_nodes, 27, 6, dtype=torch.float)
    velocity_for_occupancy = torch.zeros(num_nodes, 27, 2, dtype=torch.float)
    for index, ego_state in enumerate(ALL_ego_states):
        x[av_index, index, 0] = ego_state.center.x
        x[av_index, index, 1] = ego_state.center.y
        x[av_index, index, 2] = ego_state.center.heading
        x[av_index, index, 3] = ego_state.dynamic_car_state.speed
    edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
    padding_mask = torch.ones(num_nodes, 27, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
    bos_mask = torch.zeros(num_nodes, 11, dtype=torch.bool)
    rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
    # Vehicleì˜ ê²½ìš°
    for step, s_c in enumerate(vehicle_tracks):
        for agent in s_c.tracked_objects.tracked_objects:
            node_idx = actor_ids.index(agent.track_token)
            if node_idx == av_index:
                raise ValueError
            padding_mask[node_idx, step] = False
            speed = np.linalg.norm([agent.velocity.x, agent.velocity.y], ord=2)
            # x[node_idx, step, :] = torch.Tensor([agent.center.x, agent.center.y, agent.center.heading, speed]) # BH ì§ˆë¬¸í•˜ê¸° (ì´ê±° ì—ëŸ¬ë‚˜ë˜ë°)
            x[node_idx, step, :4] = torch.Tensor([agent.center.x, agent.center.y, agent.center.heading, speed])
            velocity_for_occupancy[node_idx, step, :2] = torch.Tensor([agent._velocity.x, agent._velocity.y])
            
    global_vehicle_for_occupancy = x.clone()
    temp_x = x.clone().numpy()
    # AVì˜ ê²½ìš°
    padding_mask[av_index, :] = False

    for i in range(len(actor_ids)):
        if padding_mask[i, 10]:  # make no predictions for actors that are unseen at the current time step
                padding_mask[i, 11:] = True
    # padding_maskë¥¼ ì‚¬ìš”
    for i in range(len(actor_ids)):
        if padding_mask[i, 10] == False: 
            valid_index = [index for index,val in enumerate(padding_mask[i,:]) if val == False]
            diff_valid = np.array(valid_index[1:]) - np.array(valid_index[:-1])
            if np.where(diff_valid > 1)[0].shape[0] > 0:
                for idx in np.where(diff_valid > 1)[0]:
                    temp_x[i, idx+(diff_valid[idx]-1):idx+(diff_valid[idx]+1)] = temp_x[i, idx]
            hist_valid_index = valid_index[0]
            future_valid_index = valid_index[-1]
            if hist_valid_index != 0:
                temp_x[i,:hist_valid_index,:] = temp_x[i,hist_valid_index,:]
            elif future_valid_index != padding_mask.shape[1]-1:
                temp_x[i,future_valid_index:,:] = temp_x[i,future_valid_index,:]
        
            
    yaw_rate_horizon = approximate_derivatives(
        temp_x[:, :11, 2], np.array([stamp.time_s for stamp in time_stamps]), window_length=3
    ) # BH ì§ˆë¬¸í•˜ê¸° í†µí™”ë¡œ ë§í•˜ë˜ í•„í„°ê°€ ì´ë¶€ë¶„ì¸ì§€? ì—¬ê¸°ì— ë§ˆìŠ¤í¬ê°€ í•„ìš”í•œì§€ ì—¬ë¶€ê°€ í•„ìš”í•œê±°? ë§žëŠ”ë“¯í•¨
    acceleration_horizon = approximate_derivatives(
        temp_x[:, :11, 3], np.array([stamp.time_s for stamp in time_stamps]), window_length=3,
    )
    
    # x[:, :11, 4] = yaw_rate_horizon # BH ì§ˆë¬¸í•˜ê¸°, í•˜ë‚˜ëŠ” tensorê³  í•˜ë‚˜ëŠ” numpyë¼ì„œ ì—ëŸ¬ë‚¨
    # x[:, :11, 5] = acceleration_horizon
        
    x[:, :11, 4] = torch.tensor(yaw_rate_horizon)
    x[:, :11, 5] = torch.tensor(acceleration_horizon)

    valid_idx = np.where(padding_mask == False)
    origin_x = x.clone()
    # x[valid_idx[0], valid_idx[1], :2] = torch.matmul(x[valid_idx[0], valid_idx[1], :2] - origin, rotate_mat)
    # x[valid_idx[0], valid_idx[1], 2] = x[valid_idx[0], valid_idx[1], 2] - theta
    # x[valid_idx[0], valid_idx[1], 2] = x[valid_idx[0], valid_idx[1], 2] % (2 * np.pi) # positive heading(0~2n)
    # for index_ in range(len(valid_idx[0])):
    #     ad_pi =  x[valid_idx[0][index_], valid_idx[1][index_], 2].item()
    #     if ad_pi > np.pi:
    #         x[valid_idx[0][index_], valid_idx[1][index_], 2] = ad_pi - (2*np.pi)
    #     else:
    #         x[valid_idx[0][index_], valid_idx[1][index_], 2] = ad_pi
    num_objects, num_timestamps, num_attrs = x.shape
    center_xyz = torch.tensor([current_pose.x, current_pose.y, current_pose.heading])
    center_heading = torch.tensor(current_pose.heading).unsqueeze(0) # ì°¾ì€ë“¯
    center_xyz = center_xyz.unsqueeze(0) # 1(av), 3
    num_center_objects = center_xyz.shape[0]
    x = x.clone().view(1, num_objects, num_timestamps, num_attrs).repeat(num_center_objects, 1, 1, 1)
    x[:, :, :, 0:center_xyz.shape[1]] -= center_xyz[:, None, None, :]
    def check_numpy_to_torch(x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float(), True
        return x, False
    def rotate_points_along_z(points, angle):
        """
        Args:
            points: (B, N, 3 + C)
            angle: (B), angle along z-axis, angle increases x ==> y
        Returns:

        """
        points, is_numpy = check_numpy_to_torch(points)
        angle, _ = check_numpy_to_torch(angle)
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        if points.shape[-1] == 2:
            rot_matrix = torch.stack((
                cosa,  sina,
                -sina, cosa
            ), dim=1).view(-1, 2, 2).float()
            points_rot = torch.matmul(points, rot_matrix)
        else:
            ones = angle.new_ones(points.shape[0])
            rot_matrix = torch.stack((
                cosa,  sina, zeros,
                -sina, cosa, zeros,
                zeros, zeros, ones
            ), dim=1).view(-1, 3, 3).float()
            points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
            points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot.numpy() if is_numpy else points_rot
    x[:, :, :, 0:2] = rotate_points_along_z(points=x[:, :, :, 0:2].view(num_center_objects, -1, 2),
                                                    angle=-center_heading).view(num_center_objects, num_objects, num_timestamps, 2)
    x = x.squeeze(0)
    # center_heading -= np.pi/2
    # obj_trajs[:, :, :, 2] -= center_heading[:, None, None]

    
    # x[valid_idx[0], valid_idx[1], :2] = torch.matmul(x[valid_idx[0], valid_idx[1], :2] - origin, rotate_mat)
    # x[valid_idx[0], valid_idx[1], 2] = x[valid_idx[0], valid_idx[1], 2] - theta
    # y ì˜ˆì¸¡ì„ ìœ„í•´ ê°’ì˜ ë²”ìœ„ë¥¼ ì¡°ì ˆí•œë‹¤.
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
    av_ind = actor_ids.index('ego')
    
    node_positions = origin_x[node_inds, 10, :2]
    # extract planner centerline
    # current_lane = self._planner._get_starting_lane(ego_state)
    # centerline = PDMPath(self._planner._get_discrete_centerline(current_lane))
    # centerline = self._planner._centerline
    # centerline_lane_objects = self._planner._route_plan
    current_progress: float = centerline.project(Point(*current_pose.array))
    centerline_progress_values = (
        np.arange(radius, dtype=np.float64)
        * model.centerline_interval
        + current_progress
    )  # distance values to interpolate
    planner_centerline = convert_absolute_to_relative_se2_array(
        current_pose,
        centerline.interpolate(centerline_progress_values, as_array=True),
    )  # convert to relative coords
        
    
    LANE_LAYER = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR, SemanticMapLayer.INTERSECTION,
                  SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    current_ego_global_pose: Point2D = Point2D(current_pose.x, current_pose.y)
    LANE_obj = map_api.get_proximal_map_objects(current_ego_global_pose, radius, LANE_LAYER)
    LANES = LANE_obj[list(LANE_obj.keys())[0]]
    LANE_CONS = LANE_obj[list(LANE_obj.keys())[1]]
    ROADBLOCKS = LANE_obj[list(LANE_obj.keys())[3]]
    ROADBLOCK_CONS = LANE_obj[list(LANE_obj.keys())[4]]
    
    lane_ = {"Lane":[], "Lane_polygon":[], "neightbor_Lane_with_road_block":{}, "local_Lane_polygon":[],
                "Lane_cons":[], "Lane_cons_polygon":[], "neightbor_cons_with_road_block":{}, "local_cons_polygon":[]}
    lane_["Lane"] = [Lane.id for Lane in LANES]
    lane_["Lane_polygon"] = [Lane.polygon for Lane in LANES]
    roadblock_poly = [RB.polygon for RB in ROADBLOCKS]
    roadblock_poly = [RB.polygon for RB in ROADBLOCK_CONS] + roadblock_poly
    local_roadblock = []
    for poly in roadblock_poly:
        x_ = list(poly.exterior.xy[0])
        y_ = list(poly.exterior.xy[1])
        arr_ = torch.tensor([x_, y_]).T
        local_arr_ = torch.matmul(arr_ - origin, rotate_mat)
        local_arr_ = local_arr_.numpy()
        polygon = Polygon(local_arr_)
        local_roadblock.append(polygon)
    
    lane_["neightbor_Lane_with_road_block"] = {Lane.parent.id:[lane_.id for lane_ in Lane.parent.interior_edges] for Lane in LANES}
    for poly in lane_["Lane_polygon"]:
        x_ = list(poly.exterior.xy[0])
        y_ = list(poly.exterior.xy[1])
        arr_ = torch.tensor([x_, y_]).T
        local_arr_ = torch.matmul(arr_ - origin, rotate_mat)
        local_arr_ = local_arr_.numpy()
        polygon = Polygon(local_arr_)
        lane_["local_Lane_polygon"].append(polygon)
    lane_["Lane_cons"] = [Lane_con.id for Lane_con in LANE_CONS]
    lane_["Lane_cons_polygon"] = [Lane_con.polygon for Lane_con in LANE_CONS]
    lane_["neightbor_cons_with_road_block"] =  {Lane_con.parent.id:[lane_.id for lane_ in Lane_con.parent.interior_edges] for Lane_con in LANE_CONS}
    for poly in lane_["Lane_cons_polygon"]:
        x_ = list(poly.exterior.xy[0])
        y_ = list(poly.exterior.xy[1])
        arr_ = torch.tensor([x_, y_]).T
        local_arr_ = torch.matmul(arr_ - origin, rotate_mat)
        local_arr_ = local_arr_.numpy()
        polygon = Polygon(local_arr_)
        lane_["local_cons_polygon"].append(polygon)
    
    cliped_route_ids = []
    route_lane_ids = []
    route_con_ids = []
    find_goal = False
        
    route_candidate_list = []
    route_start_idx_list = []
    route_end_idx_list = []
    centerline_lane_ids = [lane_.id for lane_ in centerline_lane_objects]
    
    end_lanes = {"Lane_ids":[], "Lane_con_ids":[], "goal_point":[], "route_lane_ids":[], "route_con_ids":[]}
    start_lanes = {"Lane_ids":[], "Lane_con_ids":[]}
    centerline_lanes = {"start_lane_id":[],"Lane_ids":[], "Lane_con_ids":[]}
    centerline_lanes["start_lane_id"].append(current_lane.id)
    for centerline_index, point in enumerate(planner_centerline[:,:2].copy()[::-1]):
        line_point = np.expand_dims(point[:2], axis=0)
        indexs = np.where(points_in_polygons(line_point, lane_["local_Lane_polygon"])==True)[0]
        if len(indexs) != 0:
            for index in indexs:
                route_candidate_list.append(LANES[index].parent.id)
                if LANES[index].id in centerline_lane_ids:
                    centerline_lanes["Lane_ids"].append(LANES[index].id)
                if centerline_index == planner_centerline.shape[0]-1:
                    route_start_idx_list.append(LANES[index].parent)
                if centerline_index == 0:
                    route_end_idx_list.append(LANES[index].parent.id)
        indexs = np.where(points_in_polygons(line_point, lane_["local_cons_polygon"])==True)[0]
        if len(indexs) != 0:
            for index in indexs:
                route_candidate_list.append(LANE_CONS[index].parent.id)
                if LANE_CONS[index].id in centerline_lane_ids:
                    centerline_lanes["Lane_con_ids"].append(LANE_CONS[index].id)
                if centerline_index == planner_centerline.shape[0]-1:
                    route_start_idx_list.append(LANE_CONS[index].parent)
                if centerline_index == 0:
                    route_end_idx_list.append(LANE_CONS[index].parent.id)

    centerline_lanes["Lane_ids"] = list(set(centerline_lanes["Lane_ids"]))
    centerline_lanes["Lane_con_ids"] = list(set(centerline_lanes["Lane_con_ids"]))
    route_candidate_list = set(route_candidate_list)    
    route_end_idx_list = set(route_end_idx_list)   
    isFindRoute = False

    for start_roadblock in route_start_idx_list:
        route_list = [start_roadblock.id] 
        for next_roadblock in start_roadblock.outgoing_edges:
            isFindRoute, route_list2 = findRoute(route_list.copy(), next_roadblock, route_end_idx_list, route_candidate_list, isFindRoute)
            if isFindRoute:
                route_list = route_list2
                break
        if isFindRoute:
            break
        
    for centerline_index, route_id in enumerate(route_list[::-1]):
        try:
            cliped_route_ids.append(route_id)
            if find_goal == False:
                end_lanes["Lane_ids"] = lane_["neightbor_Lane_with_road_block"][route_id]
                end_lanes["goal_point"] = np.expand_dims(planner_centerline[-1, :2], axis=0)
                find_goal = True
            if centerline_index == len(route_list)-1:
                start_lanes["Lane_ids"] = lane_["neightbor_Lane_with_road_block"][route_id]
            for Lane_id in lane_["neightbor_Lane_with_road_block"][route_id]:
                route_lane_ids.append(Lane_id)
        except:
            cliped_route_ids.append(route_id)
            if find_goal == False:
                end_lanes["Lane_con_ids"] = lane_["neightbor_cons_with_road_block"][route_id]
                end_lanes["goal_point"] = np.expand_dims(planner_centerline[-1, :2], axis=0)
                find_goal = True
            if centerline_index == len(route_list)-1:
                start_lanes["Lane_con_ids"] = lane_["neightbor_cons_with_road_block"][route_id]
            for con_id in lane_["neightbor_cons_with_road_block"][route_id]:
                route_con_ids.append(con_id)
            
    cliped_route_ids = list(set(cliped_route_ids))
    route_lane_ids = list(set(route_lane_ids))
    route_con_ids = list(set(route_con_ids))
    end_lanes["route_lane_ids"] = route_lane_ids
    end_lanes["route_con_ids"] = route_con_ids
    
    #np.array(points_in_polygons(lane_centerline, intersection_polygons))
    
    local_route_polygon = []
    for id_ in initialization.route_roadblock_ids:
        try:
            poly = map_api._get_roadblock_connector(id_).polygon
        except Exception as e:
            poly = map_api._get_roadblock(id_).polygon
        x_ = list(poly.exterior.xy[0])
        y_ = list(poly.exterior.xy[1])
        arr_ = torch.tensor([x_, y_]).T
        local_arr_ = torch.matmul(arr_ - origin, rotate_mat)
        local_arr_ = local_arr_.numpy()
        polygon = Polygon(local_arr_)
        local_route_polygon.append(polygon)
    
    
    (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
        lane_actor_vectors, lane_positions, goals, in_route_lanes,in_centerline_lanes) = get_lane_features(planner_input, initialization.map_api, node_inds, node_positions, origin, rotate_mat, radius, end_lanes, start_lanes, centerline_lanes)

    # # visualization
    # import matplotlib.pyplot as plt
    # import matplotlib.patches as patches
    
    # cur_t = 11
    # hist_t = list(range(0, 12))
    # plt.figure(figsize=(26,26))
    # plt.xlim(-50, 50)
    # plt.ylim(-50, 50)
    
    # agent_w = 5
    # agent_h = 2
    
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
            
    # lane_x = lane_positions[:, 0]
    # lane_y = lane_positions[:, 1]
    # plt.scatter(lane_x, lane_y, s = 25, c="gray", alpha=0.7)
    # plt.savefig(f"/home/workspace/pictures/{scenario.token}.PNG")
    vis_pack = {}
    vis_pack["padding_mask"] = padding_mask.clone()
    vis_pack["x"] = x.clone()
    vis_pack["lane_positions"] = lane_positions.clone()
    vis_pack["centerline"] = planner_centerline[:,:2].copy()
    vis_pack["local_route"] = local_route_polygon
    vis_pack['local_roadblock'] = local_roadblock
    lane_positions = lane_positions.clone()
    
    
    rotate_angles = x[:, 10, 2].clone()
    rotate_angle_test = x[:, 10, 2]

    bos_mask[:, 0] = ~padding_mask[:, 0]
    bos_mask[:, 1: 11] = padding_mask[:, : 10] & ~padding_mask[:, 1:11]

    positions = x.clone()
    x[:, 11:, :3] = torch.where((padding_mask[:, 10].unsqueeze(-1) | padding_mask[:, 11:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 16, 3),
                            x[:, 11:, :3] - x[:, 10, :3].unsqueeze(-2))
    x[:, 1: 11, :3] = torch.where((padding_mask[:, : 10] | padding_mask[:, 1: 11]).unsqueeze(-1),
                            torch.zeros(num_nodes, 10, 3),
                            x[:, 1: 11, :3] - x[:, : 10, :3])
    x[:, 0, :3] = torch.zeros(num_nodes, 3)
    
    x[..., 2] = torch.tensor(np.arctan2(torch.sin(x[..., 2]).detach().numpy(), torch.cos(x[..., 2]).detach().numpy()))

    y = x[:, 11:].clone()

    # Others ì¶”ì •
    # origin = ego_states[-1]
    # theta = ego_states[-1].rear_axle.heading
    # rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
    #                     [np.sin(theta), np.cos(theta)]])
    # # ego_loc = np.array([origin.rear_axle.x, origin.rear_axle.y])
    # node_positions = [ego_states[-4], ego_states[-8]]
    
    # initialization.map_api
    # node_positions -> ì£¼ë³€ agentì˜ ìœ„ì¹˜ë¥¼ global_coordinate í˜•ì‹ìœ¼ë¡œ ìž…ë ¥, ì´ë•Œ ê°ì²´ ì •ë³´ëŠ” ìœ ì§€ë¨
    # origin -> egoì˜ ìœ„ì¹˜ë¥¼ global_coordinate í˜•ì‹ìœ¼ë¡œ ìž…ë ¥, ì´ë•Œ ê°ì²´ ì •ë³´ëŠ” ìœ ì§€ë¨
    # rot_mat -> global_coordinate ìƒì—ì„œ AVì°¨ëŸ‰ì˜ headingì„ í†µí•´ ë§Œë“¤ì–´ì§„ rotation matric
    # radius -> ego ë° agent ì°¨ëŸ‰ ê°ê°ì— ëŒ€í•´ì„œ ê°€ì ¸ì˜¬ laneì˜ ë²”ìœ„
    # ego_states[-1] -> 
    
    # extract planner centerline
    # current_progress: float = centerline.project(
    #     Point(*current_ego_state.rear_axle.array)
    # )
    # centerline_progress_values = (
    #     np.arange(model.centerline_samples, dtype=np.float64)
    #     * model.centerline_interval
    #     + current_progress
    # )  # distance values to interpolate
    # planner_centerline = convert_absolute_to_relative_se2_array(
    #     current_ego_state.rear_axle,
    #     centerline.interpolate(centerline_progress_values, as_array=True),
    # )  # convert to relative coords

    if closed_loop_trajectory is not None:
        current_time: TimePoint = current_ego_state.time_point
        future_step_time: TimeDuration = TimeDuration.from_s(
            model.trajectory_sampling.step_time
        )
        future_time_points: List[TimePoint] = [
            current_time + future_step_time * (i + 1)
            for i in range(model.trajectory_sampling.num_poses)
        ]
        trajectory_ego_states = closed_loop_trajectory.get_state_at_times(
            future_time_points
        )  # sample to model trajectory

        planner_trajectory = ego_states_to_state_array(
            trajectory_ego_states
        )  # convert to array
        planner_trajectory = planner_trajectory[
            ..., StateIndex.STATE_SE2
        ]  # drop values
        planner_trajectory = convert_absolute_to_relative_se2_array(
            current_ego_state.rear_axle, planner_trajectory
        )  # convert to relative coords

    else:
        # use centerline as dummy value
        planner_trajectory = planner_centerline
    
    result = {
        'x':x, # x.shape ~ [15, 27, 3]
        'positions':positions, # positions.shape ~ [15, 27, 3]
        'edge_index':edge_index,
        'y': y, # y.shape(15, 16, 3)
        'num_nodes':num_nodes, #JY
        'padding_mask':padding_mask, #(48, 27)
        'bos_mask':bos_mask,
        'rotate_angles':rotate_angles,
        'rotate_angle_test': rotate_angle_test, 
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
        'goal': goals,
        'in_route_lanes':in_route_lanes,
        'centerline': planner_centerline,
        'centerline_mask': in_centerline_lanes,
        'local_route':local_route_polygon,
        'local_roadblock':local_roadblock,
        'lane_positions':lane_positions, 
        'obs_ids_for_occupancy': obs_ids_for_occupancy, 
        'global_vehicle_for_occupancy': global_vehicle_for_occupancy,
        'velocity_for_occupancy':velocity_for_occupancy,
        'initialization_for_occupancy': initialization}
        
    return result, vis_pack
    