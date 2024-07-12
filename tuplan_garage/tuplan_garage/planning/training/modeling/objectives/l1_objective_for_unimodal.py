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


# class LaplaceNLLLoss(nn.Module):

#     def __init__(self,
#                  eps: float = 1e-6,
#                  reduction: str = 'mean') -> None:
#         super(LaplaceNLLLoss, self).__init__()
#         self.eps = eps
#         self.reduction = reduction

#     def forward(self,
#                 pred: torch.Tensor,
#                 target: torch.Tensor) -> torch.Tensor:
#         loc, scale = pred.chunk(2, dim=-1)
#         scale = scale.clone()
#         with torch.no_grad():
#             scale.clamp_(min=self.eps)
#         nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
#         if self.reduction == 'mean':
#             return nll.mean()
#         elif self.reduction == 'sum':
#             return nll.sum()
#         elif self.reduction == 'none':
#             return nll
#         else:
#             raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
        
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
        
class L1Objective(AbstractObjective):
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
        self._name = "l1_objective"
        self._weight = weight
        self._loss_function = torch.nn.modules.loss.L1Loss(reduction="none")
        self._cls_loss_function = SoftTargetCrossEntropyLoss()
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
        predicted_trajectory = predictions["trajectory"][:, :, :2]
        targets_trajectory = targets['pdm_features']['y'][:, :16, :2]
        # targets_trajectory = cast(Trajectory, targets["trajectory"])

        # scenario_weights = extract_scenario_type_weight(
        #     scenarios,
        #     self._scenario_type_loss_weighting,
        #     device=predicted_trajectory.data.device,
        # )

        batch_size = predicted_trajectory.shape[0]
        # assert predicted_trajectory.data.shape == targets_trajectory.data.shape

        loss = self._loss_function(
            predicted_trajectory[reg_mask].reshape(predicted_trajectory[reg_mask].shape[0], -1),
            targets_trajectory[reg_mask].reshape(predicted_trajectory[reg_mask].shape[0], -1),
        )
        loss = torch.mean(loss)
        # if prefix == 'train':
        #     cls_loss = self._cls_loss_function(predictions['pi'], targets['soft_target'])
        #     loss = loss + cls_loss
        #loss.requires_grad = True
        # loss = self._loss_function(
        #     predicted_trajectory.reshape(batch_size, -1),
        #     targets_trajectory.reshape(batch_size, -1),
        # )

        return self._weight * loss # * scenario_weights[..., None])
        # return self._weight * loss # * scenario_weights[..., None])
