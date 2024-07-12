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

## 추가 라이브러리 ,BH
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
#         turn_direction = 0 # 직접적으로 turn direction을 알아낼 방법이 없음
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
#         turn_direction = 0 # 직접적으로 turn direction을 알아낼 방법이 없음
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

def get_lane_features(current_input, map_api, node_inds, node_positions, origin, rot_mat, radius, end_lanes, planner_centerline):
    lane_positions, lane_vectors, is_intersections, turn_directions, traffic_controls, goals, lane_centerlines, in_route_lanes = [], [], [], [], [], [], [], []
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
        turn_direction = 0 # 직접적으로 turn direction을 알아낼 방법이 없음
        traffic_control = False
        lane_positions.append(lane_centerline[:-1])
        lane_vectors.append(lane_centerline[1:] - lane_centerline[:-1])
        count = len(lane_centerline) - 1
        is_intersections.append(is_intersection * torch.ones(count, dtype=torch.uint8))
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
        # # goal
        # if lane_id in end_lanes["Lane_ids"]:
        #     sp_lane = False * torch.ones(count, dtype=torch.uint8)
        #     sp_lane[-1] = True
        #     goals.append(sp_lane)
        # else:
        #     goals.append(False * torch.ones(count, dtype=torch.uint8))
        # # in route_lanes
        if lane_id in end_lanes["route_lane_ids"]:
            in_route_lanes.append(True * torch.ones(count, dtype=torch.uint8))
        else:
            in_route_lanes.append(False * torch.ones(count, dtype=torch.uint8))
        
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
        # # goal
        # if lane_id in end_lanes["Lane_con_ids"]:
        #     sp_lane = False * torch.ones(count, dtype=torch.uint8)
        #     sp_lane[-1] = True
        #     goals.append(sp_lane)
        # else:
        #     goals.append(False * torch.ones(count, dtype=torch.uint8))
        # # in route_lanes
        if lane_id in end_lanes["route_con_ids"]:
            in_route_lanes.append(True * torch.ones(count, dtype=torch.uint8))
        else:
            in_route_lanes.append(False * torch.ones(count, dtype=torch.uint8))
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = torch.cat(lane_vectors, dim=0)
    is_intersections = torch.cat(is_intersections, dim=0)
    turn_directions = torch.cat(turn_directions, dim=0)
    traffic_controls = torch.cat(traffic_controls, dim=0)
    # goals = torch.cat(goals,dim=0)
    in_route_lanes = torch.cat(in_route_lanes, dim=0)
    goal_index = torch.argmin(torch.cdist(torch.tensor(planner_centerline[-1][:2]).unsqueeze(0), lane_positions.type(torch.double)))
    goals = torch.zeros_like(in_route_lanes) > 1
    goals[goal_index] = True
    
    
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]
    
    return lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index, lane_actor_vectors, lane_positions, goals, in_route_lanes

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
    
    # training이랑 동일하게 가져가는 전략
    # 추후 scenario에 대한 의존도를 없애야 할것
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
    # vehicle_agent_feats, vehicle_agent_masks = self._get_surrounding_agents_representation(vehicle_tracks, ego_states, time_stamps)
    for s_c in vehicle_tracks:
        # if num_nodes < len(s_c.tracked_objects.tracked_objects):
        #     num_nodes = len(s_c.tracked_objects.tracked_objects)
        for agent in s_c.tracked_objects.tracked_objects:
            actor_ids.append(agent.track_token)

    actor_ids = list(set(actor_ids))
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
    x = torch.zeros(num_nodes, 27, 3, dtype=torch.float)
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
    center_heading = torch.tensor(current_pose.heading).unsqueeze(0) # 찾은듯
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
    # y 예측을 위해 값의 범위를 조절한다.
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
    centerline: PDMPath = centerline
    current_progress: float = centerline.project(Point(*current_pose.array))
    centerline_progress_values = (
        np.arange(radius, dtype=np.float64) # Test centerline
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
    end_lanes = {"Lane_ids":[], "Lane_con_ids":[], "goal_point":[], "route_lane_ids":[], "route_con_ids":[]}
    for centerline_index, point in enumerate(planner_centerline[:,:2].copy()[::-1]):
        line_point = np.expand_dims(point[:2], axis=0)
        indexs = np.where(points_in_polygons(line_point, lane_["local_Lane_polygon"])==True)[0]
        if len(indexs) != 0:
            for index in indexs:
                road_block_id_from_Lane = LANES[index].parent.id
                if road_block_id_from_Lane in initialization.route_roadblock_ids:
                    cliped_route_ids.append(road_block_id_from_Lane)
                    if find_goal == False:
                        end_lanes["Lane_ids"] = lane_["neightbor_Lane_with_road_block"][road_block_id_from_Lane]
                        end_lanes["goal_point"] = line_point
                        find_goal = True
                    for Lane_id in lane_["neightbor_Lane_with_road_block"][road_block_id_from_Lane]:
                        route_lane_ids.append(Lane_id)
        indexs = np.where(points_in_polygons(line_point, lane_["local_cons_polygon"])==True)[0]
        if len(indexs) != 0:
            for index in indexs:
                road_block_id_from_con = LANE_CONS[index].parent.id
                if road_block_id_from_con in initialization.route_roadblock_ids:
                    cliped_route_ids.append(road_block_id_from_con)
                    if find_goal == False:
                        end_lanes["Lane_con_ids"] = lane_["neightbor_cons_with_road_block"][road_block_id_from_con]
                        end_lanes["goal_point"] = line_point
                        find_goal = True
                    for con_id in lane_["neightbor_cons_with_road_block"][road_block_id_from_con]:
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
        lane_actor_vectors, lane_positions, goals, in_route_lanes) = get_lane_features(planner_input, initialization.map_api, node_inds, node_positions, origin, rotate_mat, radius, end_lanes, planner_centerline)

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
    x[:, 11:, :] = torch.where((padding_mask[:, 10].unsqueeze(-1) | padding_mask[:, 11:]).unsqueeze(-1),
                            torch.zeros(num_nodes, 16, 3),
                            x[:, 11:, :] - x[:, 10, :].unsqueeze(-2))
    x[:, 1: 11, :] = torch.where((padding_mask[:, : 10] | padding_mask[:, 1: 11]).unsqueeze(-1),
                            torch.zeros(num_nodes, 10, 3),
                            x[:, 1: 11, :] - x[:, : 10, :])
    x[:, 0, :] = torch.zeros(num_nodes, 3)

    y = x[:, 11:]

    # Others 추정
    # origin = ego_states[-1]
    # theta = ego_states[-1].rear_axle.heading
    # rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
    #                     [np.sin(theta), np.cos(theta)]])
    # # ego_loc = np.array([origin.rear_axle.x, origin.rear_axle.y])
    # node_positions = [ego_states[-4], ego_states[-8]]
    
    # initialization.map_api
    # node_positions -> 주변 agent의 위치를 global_coordinate 형식으로 입력, 이때 객체 정보는 유지됨
    # origin -> ego의 위치를 global_coordinate 형식으로 입력, 이때 객체 정보는 유지됨
    # rot_mat -> global_coordinate 상에서 AV차량의 heading을 통해 만들어진 rotation matric
    # radius -> ego 및 agent 차량 각각에 대해서 가져올 lane의 범위
    # ego_states[-1] -> 
    
    # extract planner centerline
    current_progress: float = centerline.project(
        Point(*current_ego_state.rear_axle.array)
    )
    centerline_progress_values = (
        np.arange(model.centerline_samples, dtype=np.float64)
        * model.centerline_interval
        + current_progress
    )  # distance values to interpolate
    planner_centerline = convert_absolute_to_relative_se2_array(
        current_ego_state.rear_axle,
        centerline.interpolate(centerline_progress_values, as_array=True),
    )  # convert to relative coords

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
        'local_route':local_route_polygon,
        'local_roadblock':local_roadblock,
        'lane_positions':lane_positions}
        
    return result, vis_pack
    