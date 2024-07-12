import torch
import torch.nn as nn
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    SE2Index,
)
from tuplan_garage.planning.training.preprocessing.feature_builders.hivt_feature_builder_120m_goal_ver4 import (
    HiVTFeatureBuilder,
)
from tuplan_garage.planning.training.preprocessing.features.hivt_feature import (
    HiVTFeature, #JY
)
from tuplan_garage.planning.training.modeling.models.hivt.decoder_50m_goal_ver6 import MLPDecoder
# from tuplan_garage.planning.training.modeling.models.hivt.decoder import MLPDecoder
from tuplan_garage.planning.training.modeling.models.hivt.global_interactor import GlobalInteractor
from tuplan_garage.planning.training.modeling.models.hivt.local_encoder_120m_goal_ver8_v4_add_features import LocalEncoder
from tuplan_garage.planning.training.modeling.models.hivt.occ_generator import OccupancyGenerator, apply_gaussian_kernel, GaussianOccupancyGT, OccupancyGeneratorParallel
from tuplan_garage.planning.training.modeling.models.hivt.unet import UNet
import numpy as np
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import DistanceDropEdge, DistanceDropEdgeOtherAgents
from matplotlib import pyplot as plt


class HiVTModel(TorchModuleWrapper): #JY
    """
    Wrapper around PDM-Open MLP that consumes the ego history (position, velocity, acceleration)
    and the centerline to regresses ego's future trajectory.
    """

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        history_sampling: TrajectorySampling,
        planner: AbstractPlanner,
        centerline_samples: int = 120,
        centerline_interval: float = 1.0,
        hidden_dim: int = 512,
    ):
        """
        Constructor for PDMOpenModel
        :param trajectory_sampling: Sampling parameters of future trajectory
        :param history_sampling: Sampling parameters of past ego states
        :param planner: Planner for centerline extraction
        :param centerline_samples: Number of poses on the centerline, defaults to 120
        :param centerline_interval: Distance between centerline poses [m], defaults to 1.0
        :param hidden_dim: Size of the hidden dimensionality of the MLP, defaults to 512
        """

        feature_builders = [
            HiVTFeatureBuilder( #JY
                trajectory_sampling,
                history_sampling,
                planner,
                centerline_samples,
                centerline_interval,
            )
        ]

        target_builders = [
            EgoTrajectoryTargetBuilder(trajectory_sampling),
        ]

        self.trajectory_sampling = trajectory_sampling
        self.history_sampling = history_sampling

        self.centerline_samples = centerline_samples
        self.centerline_interval = centerline_interval

        self.hidden_dim = hidden_dim

        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=trajectory_sampling,
        )
        
        self.num_modes = 6

        self.local_encoder = LocalEncoder(historical_steps=11,
                                          node_dim=2,
                                          edge_dim=2,
                                          embed_dim=64,
                                          num_heads=8,
                                          dropout=0.1,
                                          num_temporal_layers=4,
                                          local_radius=50,
                                          parallel=False)
        self.global_interactor = GlobalInteractor(historical_steps=11,
                                                  embed_dim=64,
                                                  edge_dim=2,
                                                  num_modes=self.num_modes,
                                                  num_heads=8,
                                                  num_layers=3,
                                                  dropout=0.1,
                                                  rotate=True)
        self.decoder_av = MLPDecoder(local_channels=64,
                                  global_channels=64,
                                  future_steps=16,
                                  num_modes=self.num_modes,
                                  uncertain=False)
        self.decoder_agent = MLPDecoder(local_channels=64,
                                  global_channels=64,
                                  future_steps=16,
                                  num_modes=self.num_modes,
                                  uncertain=False)
        self.unet = UNet()

        self.rotate = True
        self.device = 'cuda'

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "pdm_features": PDFeature,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        input: HiVTFeature = features["pdm_features"] #JY
        input = input.to(self.device)
        
        if self.rotate:
            rotate_mat = torch.empty(input.num_nodes, 2, 2, device=self.device)
            sin_vals = torch.sin(input['rotate_angles'])
            cos_vals = torch.cos(input['rotate_angles'])
            rotate_mat[:, 0, 0] = cos_vals
            rotate_mat[:, 0, 1] = -sin_vals
            rotate_mat[:, 1, 0] = sin_vals
            rotate_mat[:, 1, 1] = cos_vals
            if input.y is not None:
                input.y[:, :, :2] = torch.bmm(input.y[:, :, :2], rotate_mat)
                input.y = input.y.to(torch.float16)
            input['rotate_mat'] = rotate_mat
        else:
            input['rotate_mat'] = None

        local_embed = self.local_encoder(data=input)
        global_embed = self.global_interactor(data=input, local_embed=local_embed)
        
        # av_mask = (torch.arange(data.num_nodes).unsqueeze(1) == data.av_index.cpu()).any(dim=1)
        # bh simulation fixing
        if type(input["av_index"]) == type(0):
            av_mask = (torch.arange(input["num_nodes"]).unsqueeze(1) == torch.tensor([input["av_index"]])).any(dim=1)
        else:
            av_mask = (torch.arange(input["num_nodes"]).unsqueeze(1) == input["av_index"].cpu()).any(dim=1)
        
        # av_mask = (torch.arange(data.num_nodes).unsqueeze(1) == data.av_index.cpu()).any(dim=1)
        other_agent_mask = torch.where(av_mask == False)[0]
        
        # av_mask = (torch.arange(input["num_nodes"]).unsqueeze(1) == input["av_index"].cpu()).any(dim=1)
        # other_agent_mask = torch.where(av_mask == False)[0]
        
        y_hat = input["y"].new_zeros(self.num_modes, input["y"].shape[0], input["y"].shape[1], 3) #input["y"].shape[2])
        pi = input["y"].new_zeros(input["y"].shape[0], self.num_modes)
        
        y_hat_av, pi_av = self.decoder_av(local_embed=local_embed[av_mask], global_embed=global_embed[:, av_mask])
        y_hat_agent, pi_agent = self.decoder_agent(local_embed=local_embed[other_agent_mask], global_embed=global_embed[:, other_agent_mask])

        y_hat[:, av_mask], pi[av_mask] = y_hat_av.to(torch.float16), pi_av.to(torch.float16)
        y_hat[:, other_agent_mask], pi[other_agent_mask] = y_hat_agent.to(torch.float16), pi_agent.to(torch.float16)
        
        occupancy_map, occupancy_map_gt = OccupancyGeneratorParallel(input, y_hat, pi) #occpancy map 생성 (key, value로 사용예정)
        
        generated_occupancy_map = self.unet(occupancy_map)
        
        blurred_occupancy_map_gt = GaussianOccupancyGT(occupancy_map_gt)
        input['occupancy_map'] = blurred_occupancy_map_gt
        
        fig1, axes1 = plt.subplots(1, 1, figsize=(30, 30))  # 3행 4열의 subplot 생성
        occupancy_resolution = 0.5
        occupancy_size = 200
        occupancy_range = int(occupancy_size * occupancy_resolution)
        drop_edge_av = DistanceDropEdge(occupancy_range/2)
        input['edge_attr'] = \
                input['positions'][input['edge_index'][0], 10, :2] - input['positions'][input['edge_index'][1], 10, :2]
        edge_index, edge_attr = drop_edge_av(input['edge_index'], input['edge_attr'])
        others_indx = edge_index[0][np.where(edge_index[1].cpu() == input.av_index[0].cpu())[0]]
        all_history = y_hat[:, others_indx][:, :, :, :2] + input.positions[others_indx][:, 10, :2].unsqueeze(0).unsqueeze(-2)
        padding_mask = input.padding_mask[others_indx]
        for k in range(6):
            for i, ah in enumerate(all_history[k]):
                axes1.plot(ah.cpu()[~padding_mask[i, 11:]][:, 0].detach().numpy(), ah.cpu()[~padding_mask[i, 11:]][:, 1].detach().numpy())
        plt.xlim([-50, 50])
        plt.ylim([-50, 50])
        plt.savefig(f'/home/workspace/visualization/occ/trajectory_visualization.png') 
        
        fig, axes = plt.subplots(1, 1, figsize=(30, 30))  # 3행 4열의 subplot 생성
        for i in range(32):
            for j in range(16):
                # ax = axes[i//4, i%4]  # subplot 선택
                axes.imshow(occupancy_map[:, 12:][i, j].cpu(), cmap='binary')  # 각 층의 occupancy map을 흑백 이미지로 표시
                axes.set_title(f"Layer {i+1}")  # subplot 제목 설정

                plt.tight_layout()  # subplot 간격 조정
                plt.savefig(f'/home/workspace/visualization/occ/gt_{i}_{j}.png') 
                plt.show()
        
        return {"trajectory": y_hat[:, :, :, :], "pi":pi, "occupancy_map": generated_occupancy_map} # mode, batch, 16, 3
