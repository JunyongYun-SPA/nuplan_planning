from typing import Dict, List, cast

import torch
from nuplan.planning.training.modeling.objectives.abstract_objective import (
    AbstractObjective,
)
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import (
    extract_scenario_type_weight,
)
from nuplan.planning.training.modeling.types import (
    FeaturesType,
    ScenarioListType,
    TargetsType,
)
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
import torch.nn as nn
import torch.nn.functional as F
import torchvision
        
class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
        
class L1ObjectiveOccupancy(AbstractObjective):
    """
    Objective for imitating the expert behavior via an L1-Loss function.
    """

    def __init__(
        self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0
    ):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = "l1_objective_occupancy"
        self._weight = 100.0 #weight
        self._loss_function = torch.nn.modules.loss.L1Loss(reduction="none")
        self._cls_loss_function = SoftTargetCrossEntropyLoss()
        self._occ_loss_fucntion = torch.nn.BCEWithLogitsLoss()
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(
        self,
        predictions: FeaturesType,
        targets: TargetsType,
        reg_mask,
        prefix: str
        # scenarios: ScenarioListType,
    ) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """

        occupancy_map = predictions["occupancy_map"]

        loss_occ = torchvision.ops.sigmoid_focal_loss(occupancy_map, targets['pdm_features']["occupancy_map"], reduction="mean")

        return self._weight * loss_occ # * scenario_weights[..., None])