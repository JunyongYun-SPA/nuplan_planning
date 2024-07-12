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
import matplotlib.pyplot as plt

def create_pdm_feature(
    model: TorchModuleWrapper,
    planner_input: PlannerInput,
    centerline: PDMPath,
    closed_loop_trajectory: Optional[InterpolatedTrajectory],
    device: str) -> PDMFeature:
    """
    Creates a PDMFeature (for PDM-Open and PDM-Offset) during simulation
    :param model: torch model (used to retrieve parameters)
    :param planner_input: nuPlan's planner input during simulation
    :param centerline: centerline path of PDM-* methods
    :param closed_loop_trajectory: trajectory of PDM-Closed (ignored if None)
    :return: PDMFeature dataclass
    """
    # feature building
    num_past_poses = model.history_sampling.num_poses
    past_time_horizon = model.history_sampling.time_horizon

    history = planner_input.history
    current_ego_state, _ = history.current_state
    past_ego_states = history.ego_states[:-1]
    
    indices = sample_indices_with_time_horizon(
        num_past_poses, past_time_horizon, history.sample_interval
    )
    sampled_past_ego_states = [past_ego_states[-idx] for idx in reversed(indices)]
    sampled_past_ego_states = sampled_past_ego_states + [current_ego_state]

    ego_position = get_ego_position(sampled_past_ego_states)
    ego_velocity = get_ego_velocity(sampled_past_ego_states)
    ego_acceleration = get_ego_acceleration(sampled_past_ego_states)

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

    # # BH 확인용 visualization
    # if BH_pack['current_iteration'] == 0:
    #     plt.figure(figsize=(36,36))
    #     out_boundary = 20
    #     #step 3-1. plt 사이즈 고정(이쁘게 Visualization)
    #     figure_size_x = [BH_pack['goal'][0]-out_boundary, BH_pack['goal'][0]+out_boundary]
    #     figure_size_y = [BH_pack['goal'][1]-out_boundary, BH_pack['goal'][1]+out_boundary]
    #     plt.xlim(figure_size_x[0], figure_size_x[1])
    #     plt.ylim(figure_size_y[0], figure_size_y[1])
    #     for con_polygon, con_id in zip(BH_pack['con_polygon'], BH_pack['con_id']):
    #         x, y = con_polygon.exterior.xy
    #         min_x, max_x, min_y, max_y = np.min(x)-out_boundary, np.max(x)+out_boundary, np.min(y)-out_boundary, np.max(y)+out_boundary
    #         figure_size_x[0] = min_x if figure_size_x[0] > min_x else figure_size_x[0]
    #         figure_size_x[1] = max_x if figure_size_x[1] < max_x else figure_size_x[1]
    #         figure_size_y[0] = min_y if figure_size_y[0] > min_y else figure_size_y[0]
    #         figure_size_y[1] = max_y if figure_size_y[1] < max_y else figure_size_y[1]
    #     x_diff = figure_size_x[1]-figure_size_x[0]
    #     y_diff = figure_size_y[1]-figure_size_y[0]
    #     if x_diff > y_diff:
    #         plt.xlim(figure_size_x[0], figure_size_x[1])
    #         mean_y = np.mean(figure_size_y)
    #         plt.ylim(mean_y - x_diff/2, mean_y + x_diff/2)
    #     else:
    #         plt.ylim(figure_size_y[0], figure_size_y[1])
    #         mean_x = np.mean(figure_size_x)
    #         plt.xlim(mean_x - y_diff/2, mean_x + y_diff/2)
    #     ego_current_loc = [current_ego_state.rear_axle.x, current_ego_state.rear_axle.y]
    #     centerline_ = {'x':[], 'y':[]}
    #     # rot_mat = np.array([[np.cos(current_ego_state.rear_axle.heading), -np.sin(current_ego_state.rear_axle.heading)],
    #     #                     [np.sin(current_ego_state.rear_axle.heading), np.cos(current_ego_state.rear_axle.heading)]])
    #     # for index in range(len(planner_centerline)):
    #     #     point = np.array([planner_centerline[index][0], planner_centerline[index][1]])
    #     #     point = np.dot(rot_mat, point)
    #     #     point = point + np.array(ego_current_loc)
    #     #     centerline_['x'].append(point[0])
    #     #     centerline_['y'].append(point[1])
    #     for point in centerline._discrete_path:
    #         centerline_['x'].append(point.x)
    #         centerline_['y'].append(point.y)
    #     for polygon, id  in zip(BH_pack['polygon'], BH_pack['id']):
    #         x, y = polygon.exterior.xy
    #         if id in BH_pack['route_id']:
    #             plt.fill(x, y, color='yellow', edgecolor='red', alpha = 0.3)
    #         else:
    #             plt.fill(x, y, color='gray', edgecolor='black', alpha = 0.3)
    #     for polygon, id in zip(BH_pack['con_polygon'],BH_pack['con_id']):
    #         x, y = polygon.exterior.xy
    #         if id in BH_pack['route_id']:
    #             plt.fill(x, y, color='yellow', edgecolor='red', alpha = 0.3)
    #         else:
    #             plt.fill(x, y, color='gray', edgecolor='black', alpha = 0.3)
    #     plt.scatter(ego_current_loc[0], ego_current_loc[1], s = 36, color='red')
    #     plt.scatter(centerline_['x'], centerline_['y'], s = 3, color='blue')
    #     plt.plot(BH_pack['goal'][0], BH_pack['goal'][1], marker='*', markersize=22, color='green')
    #     plt.savefig(f"/home/workspace/tmp_save_fig/{BH_pack['token']}.png")
    
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

    pdm_feature = PDMFeature(
        ego_position=ego_position,
        ego_velocity=ego_velocity,
        ego_acceleration=ego_acceleration,
        planner_centerline=planner_centerline,
        planner_trajectory=planner_trajectory,
    )

    pdm_feature = pdm_feature.to_feature_tensor()
    pdm_feature = pdm_feature.to_device(device)
    pdm_feature = pdm_feature.collate([pdm_feature])

    return pdm_feature
