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
from tuplan_garage.planning.training.modeling.models.hivt.decoder import MLPDecoder
from tuplan_garage.planning.training.modeling.models.hivt.global_interactor import GlobalInteractor
from tuplan_garage.planning.training.modeling.models.hivt.local_encoder_120m_goal_ver8 import LocalEncoder


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
        
        self.num_modes = 1

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
        self.decoder = MLPDecoder(local_channels=64,
                                  global_channels=64,
                                  future_steps=16,
                                  num_modes=self.num_modes,
                                  uncertain=False)

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
                input.y = torch.bmm(input.y[:, :, :2], rotate_mat)
            input['rotate_mat'] = rotate_mat
        else:
            input['rotate_mat'] = None

        local_embed = self.local_encoder(data=input)
        global_embed = self.global_interactor(data=input, local_embed=local_embed)
        y_hat, pi = self.decoder(local_embed=local_embed, global_embed=global_embed)
        # return y_hat, pi

        occupancy_resolution = 0.5
        occupancy_size = 200
        occupancy_range = int(occupancy_size * occupancy_resolution)
        occupancy_map_env = torch.zeros(input.av_index.shape[0], 1, occupancy_size, occupancy_size, dtype=torch.float16).to(pi.device)
        occupancy_map_history = torch.zeros(input.av_index.shape[0], 11, occupancy_size, occupancy_size, dtype=torch.float16).to(pi.device)
        occupancy_map_future = torch.zeros(input.av_index.shape[0], 16, occupancy_size, occupancy_size, dtype=torch.float16).to(pi.device)
        occupancy_map_gt = torch.zeros(input.av_index.shape[0], 16, occupancy_size, occupancy_size, dtype=torch.float16).to(pi.device)
        drop_edge_av = DistanceDropEdge(occupancy_range/2)
        input['edge_attr'] = \
                input['positions'][input['edge_index'][0], 10, :2] - input['positions'][input['edge_index'][1], 10, :2]
        edge_index, _ = drop_edge_av(input['edge_index'], input['edge_attr'])
        lane_edge_index, _ = drop_edge_av(input['lane_actor_index'], input['lane_actor_vectors'])
        
        av_mask = (edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
        av_mask_index = torch.where(av_mask ==True)[0] #torch.Size([1419])
        others_indx = edge_index[0][av_mask_index]
        agent_history = input.positions[others_indx][:, :11, :2] #(456, 11, 2)
        occ_mask = (abs(agent_history[:, :, 0]) < 50) * (abs(agent_history[:, :, 1]) < 50)
        agent_history = torch.where((((~input.padding_mask[others_indx, :11]) * occ_mask).unsqueeze(-1)),
                                agent_history,
                                torch.zeros(others_indx.shape[0], 11, 2).to(agent_history.device))
        occupancy_x, occupancy_y = \
                    (agent_history / occupancy_resolution).type(torch.int)[:, :, 0], \
                        (agent_history / occupancy_resolution).type(torch.int)[:, :, 1
        
        for batch_idx, av_idx in enumerate(input.av_index):
            others_indx = edge_index[0][torch.where(edge_index[1] == av_idx)[0]]
            lane_indx = lane_edge_index[0][torch.where(lane_edge_index[1] == av_idx)[0]]
            # currne_agent_indx = others_indx.clone()[~input.padding_mask[others_indx, 10]]

            for agent_idx in others_indx.clone()[~input.padding_mask[others_indx, 10]]:
                lane_indx = lane_edge_index[0][torch.where(lane_edge_index[1] == agent_idx)[0]]
                lane = input.lane_vectors[lane_indx]
                occ_mask = (abs(lane[:, 0]) < 50) * (abs(lane[:, 1]) < 50)
                
                occupancy_x, occupancy_y = \
                    (lane[occ_mask] / occupancy_resolution).type(torch.int)[:, 0], (lane[occ_mask] / occupancy_resolution).type(torch.int)[:, 1]
                
                occupancy_map_env[batch_idx, 0, (occupancy_range - occupancy_y).type(torch.long), (occupancy_x + occupancy_range).type(torch.long)] = 1
                
            for history_idx in range(11):
                history = input.positions[others_indx][:, history_idx, :2] #padding mask 고려해야됨
                history = history[~input.padding_mask[others_indx, history_idx]]
                occ_mask = (abs(history[:, 0]) < 50) * (abs(history[:, 1]) < 50)
                
                occupancy_x, occupancy_y = \
                    (history[occ_mask] / occupancy_resolution).type(torch.int)[:, 0], (history[occ_mask] / occupancy_resolution).type(torch.int)[:, 1]
                
                occupancy_map_history[batch_idx, history_idx, (occupancy_range - occupancy_y).type(torch.long), (occupancy_x + occupancy_range).type(torch.long)] = 1
        
            for future_idx in range(16):
                future = y_hat[:, others_indx][:, :, future_idx, :2] + input.positions[others_indx][:, 10, :2].unsqueeze(0) #(6, N, 2)
                future_gt = input.positions[others_indx][:, future_idx, :2] #padding mask 고려해야됨
                future_pi = pi[others_indx]
                future_pi = future_pi[~input.padding_mask[others_indx, 11+future_idx]].permute(1, 0)
                future = future[:, ~input.padding_mask[others_indx, 11+future_idx]]
                future_gt = future_gt[~input.padding_mask[others_indx, 11+future_idx]]
                occ_mask = (abs(future[:, :, 0]) < 50) * (abs(future[:, :, 1]) < 50)
                occ_mask_gt = (abs(future_gt[:, 0]) < 50) * (abs(future_gt[:, 1]) < 50)
                
                occupancy_x, occupancy_y = \
                    (future[occ_mask] / occupancy_resolution).type(torch.int)[:, 0], (future[occ_mask] / occupancy_resolution).type(torch.int)[:, 1]
                
                occupancy_x_gt, occupancy_y_gt = \
                    (future_gt[occ_mask_gt] / occupancy_resolution).type(torch.int)[:, 0], (future_gt[occ_mask_gt] / occupancy_resolution).type(torch.int)[:, 1]
                occupancy_map_gt[batch_idx, future_idx, (occupancy_range - occupancy_y_gt).type(torch.long), (occupancy_x_gt + occupancy_range).type(torch.long)] = 1
                
                tensor_tuple = torch.stack((occupancy_x, occupancy_y)).permute(1, 0).tolist()
                tuple_data = tuple(map(tuple, tensor_tuple))
                occupancy_map_future[batch_idx, future_idx, (occupancy_range - occupancy_y).type(torch.long), (occupancy_x + occupancy_range).type(torch.long)] = future_pi[occ_mask]

        occupancy_map = torch.cat((occupancy_map_env, occupancy_map_history, occupancy_map_future), dim=1)

        #(456, 11, 2)
        #(32, 11, 200, 200)
        #(32, 11, 200, 200) --> 456, 11, 200, 200 --> for
        
        # pi --> max(300) --> pi
        
        # return {"trajectory": y_hat[0, input['av_index'], :, :]} #batch, 16, 3
        return {"trajectory": y_hat[:, :, :, :], "pi":pi} # mode, batch, 16, 3
