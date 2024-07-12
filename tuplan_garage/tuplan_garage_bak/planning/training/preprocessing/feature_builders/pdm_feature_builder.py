from __future__ import annotations

from typing import List, Optional, Tuple, Type

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
from tuplan_garage.planning.training.preprocessing.features.pdm_feature import (
    PDMFeature,
)

# BH가 추가한 라이브러리
from nuplan.database.nuplan_db.nuplan_scenario_queries import get_scenarios_from_db
from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_lidarpc_tokens_with_scenario_tag_from_db,
    get_traffic_light_status_for_lidarpc_token_from_db,
)
from typing import Any, List, Generator, Optional, Tuple, Type, Union, cast
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
from nuplan.database.nuplan_db.query_session import execute_many, execute_one
from nuplan.database.nuplan_db.nuplan_db_utils import (SensorDataSource,
                                                        get_camera_channel_sensor_data,
                                                        get_lidarpc_sensor_data)
import matplotlib.pyplot as plt
from nuplan.common.maps.maps_datatypes import (SemanticMapLayer,
                                                TrafficLightStatusData,
                                                TrafficLightStatusType) # BH가 추가함
from tqdm import tqdm

class PDMFeatureBuilder(AbstractFeatureBuilder):
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

        self._trajectory_sampling = trajectory_sampling
        self._history_sampling = history_sampling
        self._centerline_samples = centerline_samples
        self._centerline_interval = centerline_interval

        self._planner = planner

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Type of the built feature."""
        return PDMFeature

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Unique string identifier of the built feature."""
        return "pdm_features"

    def get_features_from_scenario(self, scenario: AbstractScenario) -> PDMFeature:
        """Inherited, see superclass."""

        past_ego_states = [
            ego_state
            for ego_state in scenario.get_ego_past_trajectory(
                iteration=0,
                time_horizon=self._history_sampling.time_horizon,
                num_samples=self._history_sampling.num_poses,
            )
        ] + [scenario.initial_ego_state]
        
        ### Test
        # my_log = scenario.log_name
        # results = self.get_BH_minmax_sensor_time_from_db(log_file = my_log, sensor_source=self.get_lidarpc_sensor_data()) # 시나리오의 기원이 되는 DB에 직접 접근하여 해당 DB에 기록된 시간의 범위를 가져옴
        # cur_iteration_time_point = scenario.get_time_point(iteration=0) # 현재 시나리오의 iteration=0일 때의 timestamp를 가져옴
        # max_hist_time_horizon = (cur_iteration_time_point.time_us - results[0])*1e-6//1.0
        # max_future_time_horizon = (results[1] - cur_iteration_time_point.time_us)*1e-6//1.0
        # past_traffic = [status for status in scenario.get_past_traffic_light_status_history(iteration=0, time_horizon=max_hist_time_horizon, num_samples=int(max_hist_time_horizon*2))]
        # current_traffic = [status for status in scenario.get_traffic_light_status_at_iteration(iteration=0)]
        # future_traffic = [status for status in scenario.get_future_traffic_light_status_history(iteration=0, time_horizon=max_future_time_horizon, num_samples=int(max_future_time_horizon*2))]
        
        # past_ego_loc = [status for status in scenario.get_ego_past_trajectory(iteration=0, time_horizon=max_hist_time_horizon, num_samples=int(max_hist_time_horizon*2))] #
        # current_ego_loc = [scenario.get_ego_state_at_iteration(iteration = 0)]
        # future_ego_loc = [status for status in scenario.get_ego_future_trajectory(iteration=0, time_horizon=max_future_time_horizon, num_samples=int(max_future_time_horizon*2))]
        
        # past[0]에는 가장 과거가, past[-1]에는 과거 중 가장 최신인 것이 저장됨
        # future[0]에는 가장 현재와 가까운 것이 future[-1]에는 현재와 가장 먼 미래가 저장됨
        
        # total_len_of_iteration = scenario.get_number_of_iterations() # 작동 안됨 # 그냥 1만 리턴됨
        # past_traffic = [status for status in scenario.get_past_traffic_light_status_history(iteration=0, time_horizon=self._history_sampling.time_horizon, num_samples=self._history_sampling.num_poses)]
        
        # future_traffic = [status for status in scenario.get_future_traffic_light_status_history(iteration=0, time_horizon=self._future_sampling.time_horizon, num_samples=self._history_sampling.num_poses)]
        # [status for status in scenario.get_past_traffic_light_status_history(iteration=0, time_horizon=3)]
        current_input, initialization = self._get_planner_params_from_scenario(scenario)
        # BH visualization
        # print()
        # Map_Layers = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
        # out_boundary = 150
        # total_traffic = past_traffic + current_traffic + future_traffic
        # total_ego_loc = past_ego_loc + current_ego_loc + future_ego_loc
        # bh_map_api = initialization.map_api
        # for index, ego_state in tqdm(enumerate(total_ego_loc)):
        #     ego_loc = ego_state.rear_axle.point
        #     da = bh_map_api.get_proximal_map_objects(ego_loc, out_boundary, Map_Layers)
        #     lanes = list(da.values())[0]
        #     lane_cons = list(da.values())[1]
        #     plt.clf()
        #     plt.figure(figsize=(23,23))
        #     fig_size_x = [ego_loc.x - out_boundary, ego_loc.x + out_boundary]
        #     fig_size_y = [ego_loc.y - out_boundary, ego_loc.y + out_boundary]
        #     plt.plot(ego_loc.x, ego_loc.y, marker='s', markersize=16, color='Blue')
        #     for lane in lanes:
        #         x, y = lane.polygon.exterior.xy
        #         plt.fill(x, y, color='gray', edgecolor='black', alpha=0.4)
            
        #     BH_lane_con_pack = {'red_id':[], 'green_id':[]}
        #     traffic_states = total_traffic[index]
        #     try:
        #         for traffic_state in traffic_states.traffic_lights:
        #             if traffic_state.status == TrafficLightStatusType.GREEN:
        #                 BH_lane_con_pack['green_id'].append(traffic_state.lane_connector_id)
        #             elif traffic_state.status == TrafficLightStatusType.RED:
        #                 BH_lane_con_pack['red_id'].append(traffic_state.lane_connector_id)
        #     except Exception as e:
        #         pass
        #     try:
        #         for traffic_state in traffic_states:
        #             if traffic_state.status == TrafficLightStatusType.GREEN:
        #                 BH_lane_con_pack['green_id'].append(traffic_state.lane_connector_id)
        #             elif traffic_state.status == TrafficLightStatusType.RED:
        #                 BH_lane_con_pack['red_id'].append(traffic_state.lane_connector_id)
        #     except Exception as e:
        #         pass
        #     for lane_con in lane_cons:
        #         x, y = lane_con.polygon.exterior.xy
        #         lane_con_id = int(lane_con.id)
        #         if lane_con_id in BH_lane_con_pack['green_id']:
        #             plt.fill(x, y, color='green', edgecolor='black', alpha=0.4)
        #         elif lane_con_id in BH_lane_con_pack['red_id']:
        #             plt.fill(x, y, color='red', edgecolor='black', alpha=0.4)
        #         else:
        #             plt.fill(x, y, color='gray', edgecolor='black', alpha=0.4)
        #     plt.xlim = (fig_size_x[0], fig_size_x[1])
        #     plt.ylim = (fig_size_y[0], fig_size_y[1])
        #     plt.savefig(f"/home/workspace/tmp_save_fig/flow_traffic/{str(index).zfill(5)}.png")
            
        return self._compute_feature(past_ego_states, current_input, initialization)
    
    def get_lidarpc_sensor_data(self) -> SensorDataSource:
        """
        Builds the SensorDataSource for a lidar_pc.
        :return: The query parameters for lidar_pc.
        """
        return SensorDataSource('lidar_pc', 'lidar', 'lidar_token', 'MergedPointCloud')
      
    def get_BH_minmax_sensor_time_from_db(self, log_file: str, sensor_source: SensorDataSource) -> int:
        """
        Get the timestamp of the last sensor data recorded in the log file.
        :param log_file: The db file to query.
        :param sensor_source: Parameters for querying the correct table.
        :return: The timestamp of the last sensor data.
        """
        import os
        corrected_log_file= f"/home/workspace/dataset/nuplan/nuplan-v1.1/splits/mini/{log_file}.db"
        if os.path.exists(corrected_log_file) == False:
            raise ValueError
            
        
        query = f"""
        SELECT MAX(timestamp) AS max_time
        FROM {sensor_source.table};
        """
        result = execute_one(query, [], corrected_log_file)
        max_time = result["max_time"]
        
        query = f"""
        SELECT MIN(timestamp) AS min_time
        FROM {sensor_source.table};
        """
        result = execute_one(query, [], corrected_log_file)
        min_time = result["min_time"]
        return min_time, max_time

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> PDMFeature:
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

    def _compute_feature(
        self,
        ego_states: List[EgoState],
        current_input: PlannerInput,
        initialization: PlannerInitialization,
    ) -> PDMFeature:
        """
        Creates PDMFeature dataclass based in ego history, and planner input
        :param ego_states: list of ego states
        :param current_input: planner input of current frame
        :param initialization: planner initialization of current frame
        :return: PDMFeature dataclass
        """

        current_ego_state: EgoState = ego_states[-1]
        current_pose: StateSE2 = current_ego_state.rear_axle

        # extract ego vehicle history states
        ego_position = get_ego_position(ego_states)
        ego_velocity = get_ego_velocity(ego_states)
        ego_acceleration = get_ego_acceleration(ego_states)

        # run planner
        self._planner.initialize(initialization)
        trajectory: InterpolatedTrajectory = self._planner.compute_planner_trajectory(
            current_input
        )

        # extract planner trajectory
        future_step_time: TimeDuration = TimeDuration.from_s(
            self._trajectory_sampling.step_time
        )
        future_time_points: List[TimePoint] = [
            trajectory.start_time + future_step_time * (i + 1)
            for i in range(self._trajectory_sampling.num_poses)
        ]
        trajectory_ego_states = trajectory.get_state_at_times(
            future_time_points
        )  # sample to model trajectory

        planner_trajectory = ego_states_to_state_array(
            trajectory_ego_states
        )  # convert to array
        planner_trajectory = planner_trajectory[
            ..., StateIndex.STATE_SE2
        ]  # drop values
        planner_trajectory = convert_absolute_to_relative_se2_array(
            current_pose, planner_trajectory
        )  # convert to relative coords

        # extract planner centerline
        centerline: PDMPath = self._planner._centerline
        current_progress: float = centerline.project(Point(*current_pose.array))
        centerline_progress_values = (
            np.arange(self._centerline_samples, dtype=np.float64)
            * self._centerline_interval
            + current_progress
        )  # distance values to interpolate
        planner_centerline = convert_absolute_to_relative_se2_array(
            current_pose,
            centerline.interpolate(centerline_progress_values, as_array=True),
        )  # convert to relative coords

        return PDMFeature(
            ego_position=ego_position,
            ego_velocity=ego_velocity,
            ego_acceleration=ego_acceleration,
            planner_centerline=planner_centerline,
            planner_trajectory=planner_trajectory,
        )


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
