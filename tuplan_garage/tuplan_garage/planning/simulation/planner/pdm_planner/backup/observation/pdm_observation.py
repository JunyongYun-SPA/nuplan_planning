from typing import Dict, List, Optional, Tuple

import numpy as np
import shapely.creation
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import (
    TrafficLightStatusData,
    TrafficLightStatusType,
)
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import Polygon

from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_object_manager import (
    PDMObjectManager,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import (
    PDMOccupancyMap,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    BBCoordsIndex,
)


class PDMObservation:
    """PDM's observation class for forecasted occupancy maps."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        proposal_sampling: TrajectorySampling,
        map_radius: float,
        observation_sample_res: int = 2,
    ):
        """
        Constructor of PDMObservation
        :param trajectory_sampling: Sampling parameters for final trajectory
        :param proposal_sampling: Sampling parameters for proposals
        :param map_radius: radius around ego to consider, defaults to 50
        :param observation_sample_res: sample resolution of forecast, defaults to 2
        """
        assert (
            trajectory_sampling.interval_length == proposal_sampling.interval_length
        ), "PDMObservation: Proposals and Trajectory must have equal interval length!"

        # observation needs length of trajectory horizon or proposal horizon +1s (for TTC metric)
        self._sample_interval: float = trajectory_sampling.interval_length  # [s]

        # self._observation_samples: int = (
        #     proposal_sampling.num_poses + int(1 / self._sample_interval)
        #     if proposal_sampling.num_poses + int(1 / self._sample_interval)
        #     > trajectory_sampling.num_poses
        #     else trajectory_sampling.num_poses
        # )

        # BH 샘플 수 수정 # +2는 TTC 때문에 추가
        self._observation_samples: int = (proposal_sampling.num_poses + 2) 
        
        self._map_radius: float = map_radius
        self._observation_sample_res: int = observation_sample_res

        # useful things
        self._global_to_local_idcs = [
            idx // observation_sample_res
            for idx in range(self._observation_samples + observation_sample_res)
        ]
        self._collided_track_ids: List[str] = []
        self._red_light_token = "red_light"

        # lazy loaded (during update)
        self._occupancy_maps: Optional[List[PDMOccupancyMap]] = None
        self._object_manager: Optional[PDMObjectManager] = None

        self._initialized: bool = False

    def __getitem__(self, time_idx) -> PDMOccupancyMap:
        """
        Retrieves occupancy map for time_idx and adapt temporal resolution.
        :param time_idx: index for future simulation iterations [10Hz]
        :return: occupancy map
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        assert (
            0 <= time_idx < len(self._global_to_local_idcs)
        ), f"PDMObservation: index {time_idx} out of range!"

        local_idx = self._global_to_local_idcs[time_idx]
        return self._occupancy_maps[local_idx]

    @property
    def collided_track_ids(self) -> List[str]:
        """
        Getter for past collided track tokens.
        :return: list of tokens
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._collided_track_ids

    @property
    def red_light_token(self) -> str:
        """
        Getter for red light token indicator
        :return: string
        """
        return self._red_light_token

    @property
    def unique_objects(self) -> Dict[str, TrackedObject]:
        """
        Getter for unique tracked objects
        :return: dictionary of tokens, tracked objects
        """
        assert self._initialized, "PDMObservation: Has not been updated yet!"
        return self._object_manager.unique_objects

    def update(
        self,
        ego_state: EgoState,
        observation: Observation,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject]) -> None:
        """
        Update & lazy loads information  of PDMObservation.
        :param ego_state: state of ego vehicle
        :param observation: input observation of nuPlan
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :param map_api: map object of nuPlan
        """

        self._occupancy_maps: List[PDMOccupancyMap] = []
        self._object_manager = self._get_object_manager(ego_state, observation)

        (
            traffic_light_tokens,
            traffic_light_polygons,
        ) = self._get_traffic_light_geometries(traffic_light_data, route_lane_dict)

        (
            static_object_tokens,
            static_object_coords,
            dynamic_object_tokens,
            dynamic_object_coords,
            dynamic_object_dxy,
        ) = self._object_manager.get_nearest_objects(ego_state.center.point)

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(dynamic_object_tokens) > 0,
        )

        if has_static_object and static_object_coords.ndim == 1:
            static_object_coords = static_object_coords[None, ...]

        if has_dynamic_object and dynamic_object_coords.ndim == 1:
            dynamic_object_coords = dynamic_object_coords[None, ...]
            dynamic_object_dxy = dynamic_object_dxy[None, ...]

        if has_static_object:
            static_object_coords[..., BBCoordsIndex.CENTER, :] = static_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
            static_object_polygons = shapely.creation.polygons(static_object_coords)

        else:
            static_object_polygons = np.array([], dtype=np.object_)
                
        if has_dynamic_object:
            dynamic_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res,
            self._observation_sample_res,
        ):
            if has_dynamic_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_object_coords_t = (
                    dynamic_object_coords + delta_t * dynamic_object_dxy[:, None]
                )
                dynamic_object_polygons = shapely.creation.polygons(
                    dynamic_object_coords_t
                )

            all_polygons = np.concatenate(
                [
                    static_object_polygons,
                    dynamic_object_polygons,
                    traffic_light_polygons,
                ],
                axis=0,
            )

            occupancy_map = PDMOccupancyMap(
                static_object_tokens + dynamic_object_tokens + traffic_light_tokens,
                all_polygons,
            )
            self._occupancy_maps.append(occupancy_map)

        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state.car_footprint.geometry
        intersecting_obstacles = self._occupancy_maps[0].intersects(ego_polygon)
        new_collided_track_ids = []

        for intersecting_obstacle in intersecting_obstacles:
            if self._red_light_token in intersecting_obstacle:
                within = ego_polygon.within(
                    self._occupancy_maps[0][intersecting_obstacle]
                )
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        self._collided_track_ids = self._collided_track_ids + new_collided_track_ids
        self._initialized = True
        
        
    def update_predict(
        self,
        ego_state: EgoState,
        observation: Observation,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
        predictions,
        valid_mask,
        observe_id_list,
        global_state) -> None:
        """
        Update & lazy loads information  of PDMObservation.
        :param ego_state: state of ego vehicle
        :param observation: input observation of nuPlan
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :param map_api: map object of nuPlan
        """

        self._occupancy_maps: List[PDMOccupancyMap] = []
        self._object_manager = self._get_object_manager(ego_state, observation)

        (
            traffic_light_tokens,
            traffic_light_polygons,
        ) = self._get_traffic_light_geometries(traffic_light_data, route_lane_dict)

        (
            static_object_tokens,
            static_object_coords,
            dynamic_object_tokens,
            dynamic_object_coords,
            dynamic_object_dxy,
        ) = self._object_manager.get_nearest_objects(ego_state.center.point)

        has_static_object, has_dynamic_object = (
            len(static_object_tokens) > 0,
            len(dynamic_object_tokens) > 0,
        )

        if has_static_object and static_object_coords.ndim == 1:
            static_object_coords = static_object_coords[None, ...]

        if has_dynamic_object and dynamic_object_coords.ndim == 1:
            dynamic_object_coords = dynamic_object_coords[None, ...]
            dynamic_object_dxy = dynamic_object_dxy[None, ...]

        if has_static_object:
            static_object_coords[..., BBCoordsIndex.CENTER, :] = static_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
            static_object_polygons = shapely.creation.polygons(static_object_coords)

        else:
            static_object_polygons = np.array([], dtype=np.object_)
        
        
        # occupancy map에서 구한 샘플 토큰에 따라 전처리 및 모델 출력값 재배치
        moved_by_model_index = []
        try:
            predicted_df = np.zeros((len(dynamic_object_tokens), predictions.shape[1], 5, 2))
            predicted_df[:, 0, :, :] = dynamic_object_coords
            for index_in_dynamic_tokens, token_ in enumerate(dynamic_object_tokens):
                if token_ in observe_id_list:
                    index__ = np.where(np.array(observe_id_list) == token_)[0][0]
                    if valid_mask[index__] == False:
                        continue
                    # polygon을 polygon의 중심 좌표로 표현 후 회전, 이후 회전 이후 - 회전 이전을 통해 회전으로 인해 얼만큼의 보정값이 발생했는지 알아내기
                    df__ = predictions[index__]
                    local_dynamic_object_coords = dynamic_object_coords[index_in_dynamic_tokens, :, :] - dynamic_object_coords[index_in_dynamic_tokens, 4, :]
                    rots = df__[:, -1]
                    rot_mats = np.array([[np.cos(rots), -np.sin(rots)], [np.sin(rots), np.cos(rots)]])
                    rotation_revise_ = np.transpose(local_dynamic_object_coords@rot_mats, (2, 1, 0)) - np.expand_dims(local_dynamic_object_coords, axis=0)
                    gh = -global_state[index__, -1].item()
                    xy_move = df__[:, :2]@np.array([[np.cos(gh), -np.sin(gh)], [np.sin(gh), np.cos(gh)]])
                    move_fit = rotation_revise_ + np.expand_dims(xy_move, axis=1)
                    move_fit[:, BBCoordsIndex.CENTER, :] = move_fit[:, BBCoordsIndex.FRONT_LEFT, :]
                    predicted_df[index_in_dynamic_tokens, :move_fit.shape[0], ...] = move_fit
                    moved_by_model_index.append(index_in_dynamic_tokens)
        except Exception as e:
            print(f"tuplan_garage/tuplan_garage/planning/simulation/planner/pdm_planner/observation/pdm_observation.py::{e}")
            
                
        if has_dynamic_object:
            dynamic_object_coords[..., BBCoordsIndex.CENTER, :] = dynamic_object_coords[
                ..., BBCoordsIndex.FRONT_LEFT, :
            ]
        else:
            dynamic_object_polygons = np.array([], dtype=np.object_)
            dynamic_object_tokens = []

        traffic_light_polygons = np.array(traffic_light_polygons, dtype=np.object_)
        

                
        

        for sample in np.arange(
            0,
            self._observation_samples + self._observation_sample_res,
            self._observation_sample_res,
        ):
            if has_dynamic_object:
                delta_t = float(sample) * self._sample_interval
                dynamic_object_coords_t = (
                    dynamic_object_coords + delta_t * dynamic_object_dxy[:, None]
                )
                # moved_model로 덮어씌우기
                # 요구 샘플수가 prediction 결과보다 많을 때는 last_predict-바로 이전 예측을 통해 cv 확장
                if sample < predicted_df.shape[1]:
                    for moved_by_model_ in moved_by_model_index:
                        dynamic_object_coords_t[moved_by_model_] = predicted_df[moved_by_model_, sample, ...] + dynamic_object_coords[moved_by_model_]
                else:
                    for moved_by_model_ in moved_by_model_index:
                        tmp_df = predicted_df[moved_by_model_, -1, ...] - predicted_df[moved_by_model_, -2, ...]
                        tmp_delta_t = float(sample - predicted_df.shape[1]+1) * self._sample_interval
                        dynamic_object_coords_t[moved_by_model_] = predicted_df[moved_by_model_, -1, ...] + tmp_delta_t * tmp_df + dynamic_object_coords[moved_by_model_]
                
                dynamic_object_polygons = shapely.creation.polygons(dynamic_object_coords_t)

            all_polygons = np.concatenate(
                [
                    static_object_polygons,
                    dynamic_object_polygons,
                    traffic_light_polygons,
                ],
                axis=0,
            )

            occupancy_map = PDMOccupancyMap(
                static_object_tokens + dynamic_object_tokens + traffic_light_tokens,
                all_polygons,
            )
            self._occupancy_maps.append(occupancy_map)

        # save collided objects to ignore in the future
        ego_polygon: Polygon = ego_state.car_footprint.geometry
        intersecting_obstacles = self._occupancy_maps[0].intersects(ego_polygon)
        new_collided_track_ids = []

        for intersecting_obstacle in intersecting_obstacles:
            if self._red_light_token in intersecting_obstacle:
                within = ego_polygon.within(
                    self._occupancy_maps[0][intersecting_obstacle]
                )
                if not within:
                    continue
            new_collided_track_ids.append(intersecting_obstacle)

        self._collided_track_ids = self._collided_track_ids + new_collided_track_ids
        self._initialized = True

    def _get_object_manager(
        self, ego_state: EgoState, observation: Observation
    ) -> PDMObjectManager:
        """
        Creates object manager class, but adding valid tracked objects.
        :param ego_state: state of ego-vehicle
        :param observation: input observation of nuPlan
        :return: PDMObjectManager class
        """
        object_manager = PDMObjectManager()

        for object in observation.tracked_objects:
            if (
                (object.tracked_object_type == TrackedObjectType.EGO)
                or (
                    self._map_radius
                    and ego_state.center.distance_to(object.center) > self._map_radius
                )
                or (object.track_token in self._collided_track_ids)
            ):
                continue

            object_manager.add_object(object)

        return object_manager

    def _get_traffic_light_geometries(
        self,
        traffic_light_data: List[TrafficLightStatusData],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ) -> Tuple[List[str], List[Polygon]]:
        """
        Collects red traffic lights along ego's route.
        :param traffic_light_data: list of traffic light states
        :param route_lane_dict: dictionary of on-route lanes
        :return: tuple of tokens and polygons of red traffic lights
        """
        traffic_light_tokens, traffic_light_polygons = [], []

        for data in traffic_light_data:
            lane_connector_id = str(data.lane_connector_id)

            if (data.status == TrafficLightStatusType.RED) and (
                lane_connector_id in route_lane_dict.keys()
            ):
                lane_connector = route_lane_dict[lane_connector_id]
                traffic_light_tokens.append(
                    f"{self._red_light_token}_{lane_connector_id}"
                )
                traffic_light_polygons.append(lane_connector.polygon)

        return traffic_light_tokens, traffic_light_polygons
