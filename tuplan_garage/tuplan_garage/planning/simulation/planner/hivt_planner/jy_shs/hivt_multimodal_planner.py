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
from tuplan_garage.planning.simulation.planner.hivt_planner.utils.hivt_feature_utils import (
    create_hivt_feature,
)
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

# BH 추가 import
from tuplan_garage.planning.training.preprocessing.features.hivt_feature import (
    HiVTFeature, TemporalData #JY
)
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
        
        pi = predictions['pi']
        pi = pi[pdm_feature['av_index'], :]
        pi_best_mode = torch.argmax(pi)
        predictions['trajectory'] = predictions["trajectory"][pi_best_mode, ...]

        # convert to absolute
        y_hat = predictions["trajectory"][temporal_feature['av_index'],:, :2].unsqueeze(0)
        y_vec = y_hat[0, 0, :] - torch.tensor([0,0])
        # y_vec = torch.cat((y_vec, y_vec[-1, :].unsqueeze(0)))
        # ego_states = [ego_state.center.heading for ego_state in scenario.get_ego_future_trajectory(
        #     iteration=self._iteration,
        #     time_horizon=8.0,
        #     num_samples=16)]
        # headings = torch.tensor(ego_states)
        # headings = headings - temporal_feature['theta']
        # y_heading = torch.atan2(y_vec[:, 1], y_vec[:, 0])
        # y_hat = torch.cat((y_hat,y_heading.reshape(1,-1,1)), dim=-1)
        '''
        def waypoints_to_trajectory(
            waypoints: torch.Tensor, current_velocity: torch.Tensor, supersampling_ratio=100
        ) -> torch.Tensor:
            """
            Generates a yaw angle from a series of waypoints. Therefore the trajectory is interpolated with a spline,
                wich is then supersampled to calculate the yaw angle. Kinematic feasibility is not guaranteed.
                Waypoints are expected to be in local coordinates, with ego being located at (0,0) with heading 0.
                The x-axis is assumed to be the cars longitudinal axis, and the y-axis is assumed to point to the left in direction of travel
            Note:
                Gradients are preserved for waypoints. Heading is calculated without gradient information.
            :waypoints: [batch_size, num_poses, 2]
            :current_velocity: [batch_size]
            :supersampling_ratio: number of intermediate waypoints per original waypoint
            :returns: [batch_size, num_poses, 3]
            """
        '''
        current_velocity = torch.tensor([y_vec[0]])
        traj_w_heading = waypoints_to_trajectory(y_hat, current_velocity).detach().numpy()[0]
        
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
