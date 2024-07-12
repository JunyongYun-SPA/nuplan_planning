import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from nuplan.planning.script.builders.lr_scheduler_builder import build_lr_scheduler
from nuplan.planning.training.modeling.metrics.planning_metrics import * #AbstractTrainingMetric
from nuplan.planning.training.modeling.objectives.abstract_objective import aggregate_objectives
from nuplan.planning.training.modeling.objectives.imitation_objective import AbstractObjective
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
import torch.nn.functional as F
import torch.nn as nn
import pickle
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import DistanceDropEdge, DistanceDropEdgeOtherAgents

logger = logging.getLogger(__name__)


class LightningModuleWrapper(pl.LightningModule):
    """
    Lightning module that wraps the training/validation/testing procedure and handles the objective/metric computation.
    """

    def __init__(
        self,
        model: TorchModuleWrapper,
        objectives: List[AbstractObjective],
        metrics: List[AbstractTrainingMetric],
        batch_size: int,
        optimizer: Optional[DictConfig] = None,
        lr_scheduler: Optional[DictConfig] = None,
        warm_up_lr_scheduler: Optional[DictConfig] = None,
        objective_aggregate_mode: str = 'mean',
    ) -> None:
        """
        Initializes the class.

        :param model: pytorch model
        :param objectives: list of learning objectives used for supervision at each step
        :param metrics: list of planning metrics computed at each step
        :param batch_size: batch_size taken from dataloader config
        :param optimizer: config for instantiating optimizer. Can be 'None' for older models.
        :param lr_scheduler: config for instantiating lr_scheduler. Can be 'None' for older models and when an lr_scheduler is not being used.
        :param warm_up_lr_scheduler: config for instantiating warm up lr scheduler. Can be 'None' for older models and when a warm up lr_scheduler is not being used.
        :param objective_aggregate_mode: how should different objectives be combined, can be 'sum', 'mean', and 'max'.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.objectives = objectives
        self.metrics = metrics
        self.metrics_valid = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.warm_up_lr_scheduler = warm_up_lr_scheduler
        self.objective_aggregate_mode = objective_aggregate_mode
        self.batch_list = []
        self.save_pkl = []
        self.epoch = 0

        # Validate metrics objectives and model
        model_targets = {builder.get_feature_unique_name() for builder in model.get_list_of_computed_target()}
        for objective in self.objectives:
            for feature in objective.get_list_of_required_target_types():
                assert feature in model_targets, f"Objective target: \"{feature}\" is not in model computed targets!"
        for metric in self.metrics:
            for feature in metric.get_list_of_required_target_types():
                assert feature in model_targets, f"Metric target: \"{feature}\" is not in model computed targets!"

    def _step(self, loss) -> torch.Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.

        This is called either during training, validation or testing stage.

        :param batch: input batch consisting of features and targets
        :param prefix: prefix prepended at each artifact's name during logging
        :return: model's scalar loss
        """
        
        return loss

    def _compute_objectives(
        self, predictions: TargetsType, targets: TargetsType, reg_mask, prefix
    ) -> Dict[str, torch.Tensor]:
        """
        Computes a set of learning objectives used for supervision given the model's predictions and targets.

        :param predictions: model's output signal
        :param targets: supervisory signal
        :return: dictionary of objective names and values
        """
        return {objective.name(): objective.compute(predictions, targets, reg_mask, prefix) for objective in self.objectives}

    def _compute_metrics(self, predictions: TargetsType, targets: TargetsType, reg_mask) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        result = {}
        for metric in self.metrics:
            # Case에 따라서 predictions 모드를 선택하여 compute 함수에 전달
            y_hat = predictions['trajectory']
            y_hat_best = y_hat[-1]
            
            # data_y = targets['pdm_features']['y']
            if metric.name() == 'avg_displacement_error':
                # l2_norm = torch.norm(y_hat[:, :, -1, : 2] - data_y[:, -1, :2], p=2, dim=-1)
                # best_mode = l2_norm.argmin(dim=0)
                # y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] =  metric.compute(min_FDE_prediction, targets, reg_mask) # 이 함수 안에서 AV만 계산하도록 되어있음, /home/workspace/nuplan/nuplan-devkit/nuplan/planning/training/modeling/metrics/planning_metrics.py
            elif metric.name() == 'avg_heading_error':
                # l2_norm = torch.norm(y_hat[:, :, -1, : 2] - data_y[:, -1, :2], p=2, dim=-1)
                # best_mode = l2_norm.argmin(dim=0)
                # y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] =  metric.compute(min_FDE_prediction, targets, reg_mask)
            elif metric.name() == 'final_displacement_error':
                # l2_norm = torch.norm(y_hat[:, :, -1, : 2] - data_y[:, -1, :2], p=2, dim=-1)
                # best_mode = l2_norm.argmin(dim=0)
                # y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] =  metric.compute(min_FDE_prediction, targets, reg_mask)
            elif metric.name() == 'final_heading_error':
                # l2_norm = torch.norm(y_hat[:, :, -1, : 2] - data_y[:, -1, :2], p=2, dim=-1)
                # best_mode = l2_norm.argmin(dim=0)
                # y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] =  metric.compute(min_FDE_prediction, targets, reg_mask)
        
        return result
    
    def _compute_metrics_valid(self, predictions: TargetsType, targets: TargetsType, pi, reg_mask) -> Dict[str, torch.Tensor]:
        """
        Computes a set of planning metrics given the model's predictions and targets.

        :param predictions: model's predictions
        :param targets: ground truth targets
        :return: dictionary of metrics names and values
        """
        result = {}
        result_for_save = {}
            
        for metric in self.metrics:
            # Case에 따라서 predictions 모드를 선택하여 compute 함수에 전달
            y_hat = predictions['trajectory']
            num_traj = pi.size()[0]
            new_pi = pi.argmax(dim=1)
            y_hat_best = y_hat[new_pi, torch.arange(num_traj), :, :]
            
            # data_y = targets['pdm_features']['y']
            if metric.name() == 'avg_displacement_error':
                # l2_norm = torch.norm(y_hat[:, :, -1, : 2] - data_y[:, -1, :2], p=2, dim=-1)
                # best_mode = l2_norm.argmin(dim=0)
                # y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] =  metric.compute(min_FDE_prediction, targets, reg_mask) # 이 함수 안에서 AV만 계산하도록 되어있음, /home/workspace/nuplan/nuplan-devkit/nuplan/planning/training/modeling/metrics/planning_metrics.py
                result_for_save[metric.name()] = result[metric.name()]
            elif metric.name() == 'avg_heading_error':
                # l2_norm = torch.norm(y_hat[:, :, -1, : 2] - data_y[:, -1, :2], p=2, dim=-1)
                # best_mode = l2_norm.argmin(dim=0)
                # y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] =  metric.compute(min_FDE_prediction, targets, reg_mask)
                result_for_save[metric.name()] = result[metric.name()]
            elif metric.name() == 'final_displacement_error':
                # l2_norm = torch.norm(y_hat[:, :, -1, : 2] - data_y[:, -1, :2], p=2, dim=-1)
                # best_mode = l2_norm.argmin(dim=0)
                # y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] =  metric.compute(min_FDE_prediction, targets, reg_mask)
                result_for_save[metric.name()] = result[metric.name()]
            elif metric.name() == 'final_heading_error':
                # l2_norm = torch.norm(y_hat[:, :, -1, : 2] - data_y[:, -1, :2], p=2, dim=-1)
                # best_mode = l2_norm.argmin(dim=0)
                # y_hat_best = y_hat[best_mode, torch.arange(y_hat.shape[1])]
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] =  metric.compute(min_FDE_prediction, targets, reg_mask)
                result_for_save[metric.name()] = result[metric.name()]
            elif metric.name() == 'binary_iou':
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] = metric.compute(min_FDE_prediction, targets, reg_mask)
                result_for_save[metric.name()] = result[metric.name()]
            elif metric.name() == 'binary_auc':
                min_FDE_prediction = predictions.copy()
                min_FDE_prediction['trajectory'] = y_hat_best
                result[metric.name()] = metric.compute(min_FDE_prediction, targets, reg_mask)
                result_for_save[metric.name()] = result[metric.name()]
        
        return result, result_for_save

    def _log_step(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        metrics: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = 'loss',
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(f'loss/{prefix}_{loss_name}', loss)

        for key, value in objectives.items():
            self.log(f'objectives/{prefix}_{key}', value)

        for key, value in metrics.items():
            self.log(f'metrics/{prefix}_{key}', value)
            
    def _log_step_train(
        self,
        loss: torch.Tensor,
        objectives: Dict[str, torch.Tensor],
        prefix: str,
        loss_name: str = 'loss',
    ) -> None:
        """
        Logs the artifacts from a training/validation/test step.

        :param loss: scalar loss value
        :type objectives: [type]
        :param metrics: dictionary of metrics names and values
        :param prefix: prefix prepended at each artifact's name
        :param loss_name: name given to the loss for logging
        """
        self.log(f'loss/{prefix}_{loss_name}', loss)

        for key, value in objectives.items():
            self.log(f'objectives/{prefix}_{key}', value)

    def training_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during training.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        features, targets  = batch
        targets['pdm_features'] = targets['pdm_features'].to('cuda')
        
        predictions = self.forward(features)
        
        y_hat = predictions['trajectory']
        reg_mask = ~features['pdm_features']['padding_mask'][:, 11:] # (batch, 16)

        metrics = self._compute_metrics(predictions, features, reg_mask) # AV만 처리하도록 되어 있음(이미)
        y_hat_best = y_hat
        predictions['trajectory'] = y_hat_best
        
        objectives = self._compute_objectives(predictions, features, reg_mask, 'train') # BH, l1과 같은 목적 함수를 의미한다. 실제로 여기서는 L1 loss를 사용한다.
        loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)
        # cls_loss 임포팅
                
        self._log_step(loss, objectives, metrics, 'train')
        
        return self._step(loss)

    def validation_step(
        self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int
    ) -> torch.Tensor:
        """
        Step called for each batch example during validation.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        features, targets  = batch
        targets['pdm_features'] = targets['pdm_features'].to('cuda')
        
        predictions = self.forward(features)
        
        y_hat = predictions['trajectory']
        reg_mask = ~features['pdm_features']['padding_mask'][:, 11:] # (batch, 16)

        metrics = self._compute_metrics(predictions, features, reg_mask) # AV만 처리하도록 되어 있음(이미)
        y_hat_best = y_hat
        predictions['trajectory'] = y_hat_best
        
        objectives = self._compute_objectives(predictions, features, reg_mask, 'validation') # BH, l1과 같은 목적 함수를 의미한다. 실제로 여기서는 L1 loss를 사용한다.
        loss = aggregate_objectives(objectives, agg_mode=self.objective_aggregate_mode)

        self._log_step(loss, objectives, metrics, 'val')
        
        return self._step(loss)    

    def test_step(self, batch: Tuple[FeaturesType, TargetsType, ScenarioListType], batch_idx: int) -> torch.Tensor:
        """
        Step called for each batch example during testing.

        :param batch: example batch
        :param batch_idx: batch's index (unused)
        :return: model's loss tensor
        """
        # return self._step(batch, 'test')
        return self.validation_step(batch, batch_idx) # BH 수정

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Propagates a batch of features through the model.

        :param features: features batch
        :return: model's predictions
        """
        return self.model(features)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Dict[str, Union[Optimizer, _LRScheduler]]]:
        """
        Configures the optimizers and learning schedules for the training.

        :return: optimizer or dictionary of optimizers and schedules
        """
        if self.optimizer is None:
            raise RuntimeError("To train, optimizer must not be None.")

        # Get optimizer
        optimizer: Optimizer = instantiate(
            config=self.optimizer,
            params=self.parameters(),
            lr=self.optimizer.lr,  # Use lr found from lr finder; otherwise use optimizer config
        )
        # decay = set()
        # no_decay = set()
        # whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM, nn.GRU)
        # blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        # for module_name, module in self.named_modules():
        #     for param_name, param in module.named_parameters():
        #         full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
        #         if 'bias' in param_name:
        #             no_decay.add(full_param_name)
        #         elif 'weight' in param_name:
        #             if isinstance(module, whitelist_weight_modules):
        #                 decay.add(full_param_name)
        #             elif isinstance(module, blacklist_weight_modules):
        #                 no_decay.add(full_param_name)
        #         elif not ('weight' in param_name or 'bias' in param_name):
        #             no_decay.add(full_param_name)
        # param_dict = {param_name: param for param_name, param in self.named_parameters()}
        # inter_params = decay & no_decay
        # union_params = decay | no_decay
        # assert len(inter_params) == 0
        # assert len(param_dict.keys() - union_params) == 0
        
        # optim_groups = [
        #     {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
        #      "weight_decay": self.optimizer.weight_decay},
        #     {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
        #      "weight_decay": 0.0},
        # ]
        
        # optimizer = torch.optim.AdamW(optim_groups, lr=self.optimizer.lr, weight_decay=self.optimizer.weight_decay)
        
        # Log the optimizer used
        logger.info(f'Using optimizer: {self.optimizer._target_}')

        # Get lr_scheduler
        lr_scheduler_params: Dict[str, Union[_LRScheduler, str, int]] = build_lr_scheduler(
            optimizer=optimizer,
            lr=self.optimizer.lr,
            warm_up_lr_scheduler_cfg=self.warm_up_lr_scheduler,
            lr_scheduler_cfg=self.lr_scheduler,
        )
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0.0)

        optimizer_dict: Dict[str, Any] = {}
        optimizer_dict['optimizer'] = optimizer
        if lr_scheduler_params:
            logger.info(f'Using lr_schedulers {lr_scheduler_params}')
            optimizer_dict['lr_scheduler'] = lr_scheduler_params

        return optimizer_dict if 'lr_scheduler' in optimizer_dict else optimizer_dict['optimizer']