import itertools
from typing import List, Optional

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.metrics.evaluation_metrics.base.metric_base import MetricBase
from nuplan.planning.metrics.metric_result import MetricStatistics, Statistic, TimeSeries
from nuplan.planning.metrics.utils.expert_comparisons import compute_traj_errors, compute_traj_heading_errors, compute_traj_errors_v2
from nuplan.planning.metrics.utils.state_extractors import extract_ego_center_with_heading, extract_ego_time_point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history import SimulationHistory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory


class PlannerExpertAverageL2ErrorStatistics(MetricBase):
    """Average displacement error metric between the planned ego pose and expert."""

    def __init__(
        self,
        name: str,
        category: str,
        comparison_horizon: List[int],
        comparison_frequency: int,
        max_average_l2_error_threshold: float,
        metric_score_unit: Optional[str] = None,
    ) -> None:
        """
        Initialize the PlannerExpertL2ErrorStatistics class.
        :param name: Metric name.
        :param category: Metric category.
        :param comparison_horizon: List of horizon times in future (s) to find displacement errors.
        :param comparison_frequency: Frequency to sample expert and planner trajectory.
        :param max_average_l2_error_threshold: Maximum acceptable error threshold.
        :param metric_score_unit: Metric final score unit.
        """
        super().__init__(name=name, category=category, metric_score_unit=metric_score_unit)
        self.comparison_horizon = comparison_horizon
        self._comparison_frequency = comparison_frequency
        self._max_average_l2_error_threshold = max_average_l2_error_threshold
        # Store the errors to re-use in high level metrics
        self.maximum_displacement_errors: npt.NDArray[np.float64] = np.array([0])
        self.final_displacement_errors: npt.NDArray[np.float64] = np.array([0])
        self.expert_timestamps_sampled: List[int] = []
        self.average_heading_errors: npt.NDArray[np.float64] = np.array([0])
        self.final_heading_errors: npt.NDArray[np.float64] = np.array([0])
        self.selected_frames: List[int] = [0]

    def compute_score(
        self,
        scenario: AbstractScenario,
        metric_statistics: List[Statistic],
        time_series: Optional[TimeSeries] = None,
    ) -> float:
        """Inherited, see superclass."""
        return float(max(0, 1 - metric_statistics[-1].value / self._max_average_l2_error_threshold))

    def compute(self, history: SimulationHistory, scenario: AbstractScenario) -> List[MetricStatistics]:
        """
        Return the estimated metric.
        :param history: History from a simulation engine.
        :param scenario: Scenario running this metric.
        :return the estimated metric.
        """
        self._comparison_frequency = 10
        expert_frequency = 1 / scenario.database_interval
        step_size = int(expert_frequency / self._comparison_frequency)
        sampled_indices = list(range(0, len(history.data), step_size))
        
        expert_states = list(scenario.get_expert_ego_trajectory())[0::step_size][:-1]
        expert_traj_poses = extract_ego_center_with_heading(expert_states)
        expert_timestamps_sampled = extract_ego_time_point(expert_states)
        
        planned_trajectories = history.extract_ego_state
        planned_traj_poses = extract_ego_center_with_heading(planned_trajectories)
        planned_timestamps_sampled = extract_ego_time_point(planned_trajectories)

        # Find displacement error between the proposed planner trajectory and expert driven trajectory for all sampled frames during the scenario
        average_displacement_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices))) # 3, 15
        maximum_displacement_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        final_displacement_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        average_heading_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        final_heading_errors = np.zeros((len(self.comparison_horizon), len(sampled_indices)))
        
        displacement_errors = compute_traj_errors_v2(planned_traj_poses,
                                                    expert_traj_poses,
                                                    heading_diff_weight=0) # 길이 8, 나중에 여기서 3, 5, 8만 가져감
        heading_errors = compute_traj_heading_errors(planned_traj_poses,
                                                    expert_traj_poses)
        
        average_displacement_errors = np.mean(displacement_errors)
        maximum_displacement_errors = np.max(displacement_errors)
        final_displacement_errors = displacement_errors[-1]
        average_heading_errors = np.mean(heading_errors)
        final_heading_errors = heading_errors[-1]
        
        print()

        # Save to re-use in other metrics
        self.ego_timestamps_sampled = expert_timestamps_sampled[: len(sampled_indices)]
        self.selected_frames = sampled_indices

        results: List[MetricStatistics] = self._construct_open_loop_metric_results(
            scenario,
            self.comparison_horizon,
            self._max_average_l2_error_threshold,
            metric_values=average_displacement_errors,
            name='planner_expert_ADE',
            unit='meter',
            timestamps_sampled=self.ego_timestamps_sampled,
            metric_score_unit=self.metric_score_unit,
            selected_frames=sampled_indices,
        )
        # Save to re-use in high level metrics
        self.maximum_displacement_errors = maximum_displacement_errors
        self.final_displacement_errors = final_displacement_errors
        self.average_heading_errors = average_heading_errors
        self.final_heading_errors = final_heading_errors

        return results