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
from tuplan_garage.planning.training.modeling.models.hivt.decoder_50m_goal_ver6 import MLPDecoder, MLPDecoderRef, MLPDecoderIntermediate
# from tuplan_garage.planning.training.modeling.models.hivt.decoder import MLPDecoder
from tuplan_garage.planning.training.modeling.models.hivt.global_interactor import GlobalInteractor
from tuplan_garage.planning.training.modeling.models.hivt.local_encoder_120m_goal_ver8_v4_add_features_refinement import LocalEncoder, ALEncoder
# from tuplan_garage.planning.training.modeling.models.hivt.occ_genenrator import OccGenerator
import numpy as np
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import DistanceDropEdge, DistanceDropEdgeOtherAgents
from matplotlib import pyplot as plt
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import GetInRouteLaneEdgeTotal, GetInRouteLaneEdgeRoute

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
        self.decoder_av = MLPDecoderIntermediate(local_channels=64,
                                  global_channels=64,
                                  future_steps=16,
                                  num_modes=self.num_modes,
                                  uncertain=False)
        self.decoder_agent = MLPDecoderIntermediate(local_channels=64,
                                  global_channels=64,
                                  future_steps=16,
                                  num_modes=self.num_modes,
                                  uncertain=False)
        
        self.decoder_av_ref = MLPDecoderRef(local_channels=64,
                                  global_channels=64,
                                  future_steps=16,
                                  num_modes=self.num_modes,
                                  uncertain=False)
        self.decoder_agent_ref = MLPDecoderRef(local_channels=64,
                                  global_channels=64,
                                  future_steps=16,
                                  num_modes=self.num_modes,
                                  uncertain=False)

        self.rotate = True
        self.device = 'cuda'
        
        self.drop_edge_al_route = GetInRouteLaneEdgeRoute(50)
        self.drop_edge_al_other_agent = DistanceDropEdgeOtherAgents(50)
        
        self.al_encoder_av = ALEncoder(node_dim=2,
                                    edge_dim=2,
                                    embed_dim=64,
                                    num_heads=8,
                                    dropout=0.1)
        self.al_encoder_agent = ALEncoder(node_dim=2,
                                    edge_dim=2,
                                    embed_dim=64,
                                    num_heads=8,
                                    dropout=0.1)

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
        
        y_hat = input["y"].new_zeros(self.num_modes, input["y"].shape[0], input["y"].shape[1], 3) #input["y"].shape[2])
        pi = input["y"].new_zeros(input["y"].shape[0], self.num_modes)
        
        y_hat_av, pi_av, out_av = self.decoder_av(local_embed=local_embed[av_mask], global_embed=global_embed[:, av_mask])
        y_hat_agent, pi_agent, out_agent = self.decoder_agent(local_embed=local_embed[other_agent_mask], global_embed=global_embed[:, other_agent_mask])

        y_hat[:, av_mask], pi[av_mask] = y_hat_av.to(torch.float16), pi_av.to(torch.float16)
        y_hat[:, other_agent_mask], pi[other_agent_mask] = y_hat_agent.to(torch.float16), pi_agent.to(torch.float16)
        
        out_ref = out_av.new_zeros(out_av.shape[0], input.num_nodes, out_av.shape[2])
        out_ref[:, av_mask] = out_av
        out_ref[:, other_agent_mask] = out_agent
        
        out_ref_final = out_ref.new_zeros(out_ref.shape[0], out_ref.shape[1], out_ref.shape[2])
        
        # out_ref_final_av = []
        # out_ref_final_agent = []
        for m in range(self.num_modes):
            edge_index, edge_attr = self.drop_edge_al_route(input['lane_actor_index'], input['lane_actor_vectors'], input['av_index'], input['in_route_lanes'])
            av_ref = self.al_encoder_av(x=(input['lane_vectors'], out_ref[m]), edge_index=edge_index, edge_attr=edge_attr,
                                is_intersections=input['is_intersections'], turn_directions=input['turn_directions'],
                                traffic_controls=input['traffic_controls'], rotate_mat=input['rotate_mat'])
            # edge_index, edge_attr = self.drop_edge_al_other_agent(input['lane_actor_index'], input['lane_actor_vectors'], input['av_index'])
            # agent_ref = self.al_encoder_agent(x=(input['lane_vectors'], out_ref[m]), edge_index=edge_index, edge_attr=edge_attr,
            #                     is_intersections=input['is_intersections'], turn_directions=input['turn_directions'],
            #                     traffic_controls=input['traffic_controls'], rotate_mat=input['rotate_mat'])
            
            out_ref_final[m, av_mask] = av_ref[av_mask]
            # out_ref_final[m, other_agent_mask] = agent_ref[other_agent_mask]
            # out_ref_final_av.append(av_ref)
            # out_ref_final_agent.append(agent_ref)
        
        y_hat_ref = input["y"].new_zeros(self.num_modes, input["y"].shape[0], input["y"].shape[1], 3) #input["y"].shape[2])
        pi_ref = input["y"].new_zeros(input["y"].shape[0], self.num_modes)
        
        y_hat_av_ref, pi_av_ref = self.decoder_av_ref(global_embed=out_ref_final[:, av_mask])
        # y_hat_agent_ref, pi_agent_ref = self.decoder_agent_ref(global_embed=out_ref_final[:, other_agent_mask])
        y_hat_ref[:, av_mask], pi_ref[av_mask] = y_hat_av_ref.to(torch.float16), pi_av_ref.to(torch.float16)
        
        return {"trajectory": y_hat[:, :, :, :], "pi":pi, "trajectory_ref": y_hat_ref, "pi_ref": pi_ref} # mode, batch, 16, 3