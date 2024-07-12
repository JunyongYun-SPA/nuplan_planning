from typing import List

import torch

from nuplan.planning.training.modeling.metrics.abstract_training_metric import AbstractTrainingMetric
from nuplan.planning.training.modeling.types import TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
import numpy as np
# from torcheval.metrics import BinaryAUROC
from torchmetrics import AUROC
import gc
import sys

class AverageDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'avg_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType, reg_mask) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        
        if type(targets['pdm_features'].av_index) == type(0):
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == torch.tensor([targets['pdm_features'].av_index])).any(dim=1)
        else:
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == targets['pdm_features'].av_index.cpu()).any(dim=1)
        reg_mask = reg_mask[av_mask]

        predicted_trajectory: Trajectory = Trajectory(predictions["trajectory"][av_mask, :, :2]) #targets['pdm_features']['av_index']
        targets_trajectory: Trajectory = Trajectory(data=targets['pdm_features']['y'][av_mask, :16, :2]) #targets['pdm_features']['av_index']
        
        # valid_steps = reg_mask.sum(dim=-1)
        # valid_agent = valid_steps > 0
        
        # torch.norm(predicted_trajectory.xy - targets_trajectory.xy, p=2, dim=-1).sum(dim=-1)[valid_agent]/valid_steps[valid_agent]

        return (torch.norm(predicted_trajectory.xy - targets_trajectory.xy, p=2, dim=-1).sum(dim=-1)).mean()


class FinalDisplacementError(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_displacement_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType, reg_mask) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        if type(targets['pdm_features'].av_index) == type(0):
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == torch.tensor([targets['pdm_features'].av_index])).any(dim=1)
        else:
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == targets['pdm_features'].av_index.cpu()).any(dim=1)
        reg_mask = reg_mask[av_mask]
        
        predicted_trajectory: Trajectory = Trajectory(predictions["trajectory"][av_mask, :, :2])
        targets_trajectory: Trajectory = Trajectory(data=targets['pdm_features']['y'][av_mask, :16, :2])

        return (torch.norm(predicted_trajectory.terminal_position - targets_trajectory.terminal_position, dim=-1)).mean()


class AverageHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'avg_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType, reg_mask) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        if type(targets['pdm_features'].av_index) == type(0):
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == torch.tensor([targets['pdm_features'].av_index])).any(dim=1)
        else:
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == targets['pdm_features'].av_index.cpu()).any(dim=1)
        reg_mask = reg_mask[av_mask]
        
        try:
            predicted_trajectory = (predictions["trajectory"][av_mask, :, 2])
            targets_trajectory = (targets['pdm_features']['y'][av_mask, :16, 2])

            errors = torch.abs(predicted_trajectory - targets_trajectory)
            return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()
        except Exception as e:
            return 0

class FinalHeadingError(AbstractTrainingMetric):
    """
    Metric representing the heading L2 error from the final pose of a trajectory.
    """

    def __init__(self, name: str = 'final_heading_error') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType, reg_mask) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        if type(targets['pdm_features'].av_index) == type(0):
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == torch.tensor([targets['pdm_features'].av_index])).any(dim=1)
        else:
            av_mask = (torch.arange(targets['pdm_features'].num_nodes).unsqueeze(1) == targets['pdm_features'].av_index.cpu()).any(dim=1)
        reg_mask = reg_mask[av_mask]
        
        try:
            predicted_trajectory = predictions["trajectory"][av_mask, -1, 2]
            targets_trajectory = targets['pdm_features']['y'][av_mask, :16, 2][:, -1]
            errors = torch.abs(predicted_trajectory - targets_trajectory)
            return torch.atan2(torch.sin(errors), torch.cos(errors)).mean()
        except Exception as e:
            return 0

class BinaryIOU(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'binary_iou') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name
        self.smooth = 1e-6

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType, reg_mask) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        
        occupancy_map_pred = predictions["occupancy_map"].cpu().to(torch.float32)
        occupancy_map_gt = targets['pdm_features']["occupancy_map"].cpu().to(torch.int64)
        
        occupancy_map_pred = torch.where((torch.sigmoid(occupancy_map_pred) > 0.5), 1, 0) #.view(occupancy_map_gt.shape[0] * occupancy_map_gt.shape[1], -1)
        
        intersection = (occupancy_map_pred & occupancy_map_gt).float().sum((2, 3)).cpu() # (16, 16)
        union = (occupancy_map_pred | occupancy_map_gt).float().sum((2, 3)).cpu()
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        del occupancy_map_gt
        del occupancy_map_pred
        torch.cuda.empty_cache()
        
        return iou.mean()
    
class BinaryAUC(AbstractTrainingMetric):
    """
    Metric representing the displacement L2 error averaged from all poses of a trajectory.
    """

    def __init__(self, name: str = 'binary_auc') -> None:
        """
        Initializes the class.

        :param name: the name of the metric (used in logger)
        """
        self._name = name
        self.binary_auc = AUROC() #(task="binary")

    def name(self) -> str:
        """
        Name of the metric
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: TargetsType, targets: TargetsType, reg_mask) -> torch.Tensor:
        """
        Computes the metric given the ground truth targets and the model's predictions.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: metric scalar tensor
        """
        
        occupancy_map_pred = predictions["occupancy_map"].cpu().to(torch.float32)
        occupancy_map_pred_sig = torch.sigmoid(occupancy_map_pred.view(-1))
        occupancy_map_gt = targets['pdm_features']["occupancy_map"].cpu().to(torch.int64)
        # occupancy_map_pred = torch.sigmoid(predictions["occupancy_map"].to(torch.float32).view(-1))
        # occupancy_map_gt = targets['pdm_features']["occupancy_map"].to(torch.int64).view(-1)
        # auc = []
        # for i in range(batch_shape):
        #     auc_time = []
        #     for j in range(time_shape):
        #         occupancy_map_pred = predictions["occupancy_map"][i][j].cpu().to(torch.float32).view(-1)
        #         occupancy_map_gt = targets['pdm_features']["occupancy_map"][i][j].cpu().to(torch.int64).view(-1)
                
        #         auc_time.append(self.binary_auc(torch.sigmoid(occupancy_map_pred), occupancy_map_gt))
        #     auc.append(torch.tensor(auc_time).mean())
        # auc = torch.tensor(auc).mean()
        auc = self.binary_auc(occupancy_map_pred_sig, occupancy_map_gt.view(-1)).cpu()
        
        del occupancy_map_pred_sig
        del occupancy_map_pred
        del occupancy_map_gt
        torch.cuda.empty_cache()
        
        return auc