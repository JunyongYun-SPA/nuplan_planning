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
import math


class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == 'mean':
            return nll #nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))
        
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
        
class L1ObjectiveAV(AbstractObjective):
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
        self._name = "l1_objective_av"
        self._weight = weight
        self._loss_function_traj = LaplaceNLLLoss(reduction='mean')
        self._loss_function_head = torch.nn.modules.loss.L1Loss(reduction="none")
        self._cls_loss_function = SoftTargetCrossEntropyLoss(reduction='mean')
        # self._loss_function = torch.nn.modules.loss.L1Loss(reduction="none")
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
        if type(targets['pdm_features'].av_index) == type(0):
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == torch.tensor([targets['pdm_features'].av_index])).any(dim=1)
        else:
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == targets['pdm_features'].av_index.cpu()).any(dim=1)
        reg_mask = reg_mask[av_mask]
        
        predicted_trajectory = predictions["trajectory"][av_mask, :, :]
        targets_trajectory = targets['pdm_features']['y'][av_mask, :16, :3]
        # targets_trajectory = cast(Trajectory, targets["trajectory"])

        # scenario_weights = extract_scenario_type_weight(
        #     scenarios,
        #     self._scenario_type_loss_weighting,
        #     device=predicted_trajectory.data.device,
        # )

        batch_size = predicted_trajectory.shape[0]
        # assert predicted_trajectory.data.shape == targets_trajectory.data.shape
        loss_traj = self._loss_function_traj(
            torch.cat((predicted_trajectory[reg_mask][:, :2], predicted_trajectory[reg_mask][:, 3:5]), -1),
            targets_trajectory[reg_mask][:, :2].reshape(predicted_trajectory[reg_mask].shape[0], -1),
        )
        loss_head = self._loss_function_head(
            predicted_trajectory[reg_mask][:, 2].unsqueeze(-1),
            targets_trajectory[reg_mask][:, 2].unsqueeze(-1),
        )
        loss = torch.cat((loss_traj, loss_head), -1)
        
        loss = torch.mean(loss)

        if math.isnan(loss) == True:
            print('here')
        return self._weight * loss 
