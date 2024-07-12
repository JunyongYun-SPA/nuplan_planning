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
from tuplan_garage.planning.training.modeling.models.hivt.local_encoder_120m_BEV_feature import LocalEncoder, LocalEncoderAddFeature, TemporalEncoder, TemporalEncoderCenterline, AAEncoder, AAEncoderBEV
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import GetInRouteLaneEdgeTotal, GetInRouteLaneEdgeRoute
import math
from tuplan_garage.planning.training.modeling.models.hivt.embedding  import SingleInputEmbedding
from tuplan_garage.planning.training.modeling.models.hivt.unet_parts import *
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import DistanceDropEdge, DistanceDropEdgeOtherAgents

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
        batch_size: int = 32,
        resolution: int = 100,
        map_channel_dimension: int = 4,
        iterative_centerline: bool = False,
        occupied_area: bool = False,
        pos_emb: str = 'learnable',
        agent_temporal_pos_emb: bool = False,
        additional_state: bool = False,
        multimodal: bool = False,
        cnn: str = 'default',
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
        self.batch_size = batch_size
        self.resolution = resolution
        self.map_channel_dimension = map_channel_dimension
        self.iterative_centerline = iterative_centerline
        self.occupied_area = occupied_area
        self.pos_emb = pos_emb
        self.agent_temporal_pos_emb = agent_temporal_pos_emb
        self.additional_state = additional_state
        self.multimodal = multimodal
        self.cnn = cnn

        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=trajectory_sampling,
        )
        
        self.num_modes = 16
        if additional_state:
            self.local_encoder = LocalEncoderAddFeature(historical_steps=11,
                                            node_dim=2,
                                            edge_dim=2,
                                            embed_dim=hidden_dim,
                                            num_heads=8,
                                            dropout=0.1,
                                            num_temporal_layers=4,
                                            local_radius=50,
                                            parallel=False)
        else:
            self.local_encoder = LocalEncoder(historical_steps=11,
                                            node_dim=2,
                                            edge_dim=2,
                                            embed_dim=hidden_dim,
                                            num_heads=8,
                                            dropout=0.1,
                                            num_temporal_layers=4,
                                            local_radius=50,
                                            parallel=False)
        self.global_interactor = GlobalInteractor(historical_steps=11,
                                                  embed_dim=hidden_dim,
                                                  edge_dim=2,
                                                  num_modes=self.num_modes,
                                                  num_heads=8,
                                                  num_layers=3,
                                                  dropout=0.1,
                                                  rotate=True)
        self.centerline_temporal_encoder = TemporalEncoderCenterline(historical_steps=120,
                                                embed_dim=hidden_dim,
                                                num_heads=8,
                                                dropout=0.1,
                                                num_layers=4)
        self.drop_edge_aa = DistanceDropEdge(50)
        self.aa_encoder = AAEncoderBEV(historical_steps=11,
                                    node_dim=2,
                                    edge_dim=2,
                                    embed_dim=hidden_dim,
                                    num_heads=8,
                                    dropout=0.1,
                                    parallel=False)
        self.centerline_embed = SingleInputEmbedding(in_channel=2, out_channel=hidden_dim)
        
        self.num_layers = 3
        if iterative_centerline:
            aggr_embed_list = []
            for i in range(self.num_layers):
                aggr_embed_list.append(nn.Sequential(
                    nn.Linear(hidden_dim*2, hidden_dim*2),
                    nn.LayerNorm(hidden_dim*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim*2, hidden_dim*2),
                    nn.LayerNorm(hidden_dim*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim*2, hidden_dim),
                    nn.LayerNorm(hidden_dim)))
            self.aggr_embed_centerline = nn.ModuleList([aggr_embed_list[i] for i in range(self.num_layers)])
            
        self.aggr_embed = SingleInputEmbedding(in_channel=hidden_dim*2, out_channel=hidden_dim)
        self.av_aggr_embed = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim))
        
        if cnn == 'default':
            self.bev_cnn = (DoubleConv(map_channel_dimension, hidden_dim))
        self.bev_downsample = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(hidden_dim, hidden_dim*2)
        )
        self.bev_upsample = (Up(hidden_dim*2, hidden_dim, False))
        self.bev_embed = SingleInputEmbedding(in_channel=hidden_dim, out_channel=hidden_dim)
        
        if pos_emb == 'learnable':
            self.bev_pos_embed = nn.Parameter(torch.Tensor(resolution**2, 1, hidden_dim))
            nn.init.normal_(self.bev_pos_embed, mean=0., std=.02)
        elif pos_emb == 'depthwise':
            self.bev_pos_embed = nn.Parameter(torch.Tensor(resolution**2, 1, hidden_dim))
            nn.init.normal_(self.bev_pos_embed, mean=0., std=.02)
        
        self.bev_attn_module_av = nn.ModuleList(
            [nn.TransformerDecoderLayer(hidden_dim, 8, dim_feedforward=hidden_dim*2, dropout=0.1, batch_first=False) for i in range(self.num_layers)])
        # self.bev_attn_module_agent = nn.ModuleList(
        #     [nn.TransformerDecoderLayer(hidden_dim, 8, dim_feedforward=hidden_dim*2, dropout=0.1, batch_first=False) for i in range(self.num_layers)])
        
        self.planning_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2))

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
        
        try:
            batch_size = input.av_index.shape[0]
        except:
            batch_size = 1
            
        if type(input["av_index"]) == type(0):
            av_mask = (torch.arange(input["num_nodes"]).unsqueeze(1) == torch.tensor([input["av_index"]])).any(dim=1)
        else:
            av_mask = (torch.arange(input["num_nodes"]).unsqueeze(1) == input["av_index"].cpu()).any(dim=1)
        other_agent_mask = torch.where(av_mask == False)[0]

        local_embed = self.local_encoder(data=input)
        global_embed = self.global_interactor(data=input, local_embed=local_embed)
        global_aggr_embed = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1))
        av_point_query, agent_point_query = global_aggr_embed[:, av_mask], global_aggr_embed[:, other_agent_mask]
        
        #Centerline embedding
        centerline_emb = self.centerline_embed(torch.tensor(input.centerline).to(av_point_query)[..., :2]).permute(1, 0, 2).contiguous()
        centerline_temporal_emb = self.centerline_temporal_encoder(centerline_emb)
        
        av_point_query = self.av_aggr_embed(torch.cat((av_point_query, centerline_temporal_emb.expand(16, *centerline_temporal_emb.shape).contiguous()), dim=-1))
        
        #0: Lane, 1: RoadBlock, 2: Static, 3: TrafficLight, 4: RouteBlock
        if self.map_channel_dimension == 4:
            bev_map = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, :4].contiguous()
            
            if self.occupied_area:
                bev_map[:, 3, :, :] += (~bev_map[:, 1, :, :].to(torch.bool)).to(torch.int)
                bev_map[:, 3, :, :] = (~bev_map[:, 3, :, :].to(torch.bool)).to(torch.int)
            
        elif self.map_channel_dimension == 1:
            bev_map = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, 0:1].contiguous()
        elif self.map_channel_dimension == 2:
            bev_map = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, :2].contiguous()
        elif self.map_channel_dimension == 3:
            bev_map = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, [0, 1, 3]].contiguous()
            
        bev_cnn_feature = self.bev_cnn(bev_map)
        bev_cnn_downsample_feature = self.bev_downsample(bev_cnn_feature) 
        bev_feature = self.bev_upsample(bev_cnn_downsample_feature, bev_cnn_feature) #torch.Size([32, 64, 100, 100])
        bev_embed = bev_feature.view(batch_size, bev_feature.shape[1], -1).permute(2, 0, 1).contiguous()
        bev_embed = self.bev_embed(bev_embed)
        bev_embed = bev_embed + self.bev_pos_embed
        
        # bev_embed_agent = torch.zeros(bev_embed.shape[0], agent_point_query.shape[1], bev_embed.shape[2]).to(bev_embed)
        # bev_attn_mask_agent = torch.zeros(bev_embed_agent.shape[1], bev_embed.shape[0]).to(bev_embed)
        # point_query = torch.zeros(av_point_query.shape[0], input.num_nodes, av_point_query.shape[2]).to(av_point_query)
        
        # batch_agent_idx_list = torch.cat((input.ptr[0:1], (input.ptr[1:] - input.ptr[:-1]) - 1))
        # batch_agent_idx_list = torch.cumsum((batch_agent_idx_list), dim=-1)
        # for av_idx in range(batch_size):
        #     bev_embed_agent[:, batch_agent_idx_list[av_idx]:batch_agent_idx_list[av_idx+1]] = bev_embed[:, av_idx:av_idx+1]
        #     bev_attn_mask_agent[batch_agent_idx_list[av_idx]:batch_agent_idx_list[av_idx+1]] = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[av_idx, 1].view(-1).contiguous()
                
        bev_attn_mask_av = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, 4].view(batch_size, -1).contiguous()
        bev_attn_mask_av = ~bev_attn_mask_av.type(torch.bool)
        # bev_attn_mask_agent = ~bev_attn_mask_agent.type(torch.bool)
        
        planning_trajectories = []
        out = [None] * 16
        for indx in range(self.num_layers):
            av_point_query = self.bev_attn_module_av[indx](av_point_query, bev_embed, memory_key_padding_mask=bev_attn_mask_av)
            # agent_point_query = self.bev_attn_module_agent[indx](agent_point_query, bev_embed_agent, memory_key_padding_mask=bev_attn_mask_agent)
            # point_query[:, av_mask], point_query[:, other_agent_mask] = av_point_query, agent_point_query
            
            # for t in range(16):
            #     edge_index, _ = self.drop_edge_aa(input.edge_index_10, input.edge_attr_10)
            #     out[t] = self.aa_encoder(x=point_query[t], edge_index=edge_index)
            # out = torch.stack(out)
            # av_point_query, agent_point_query = out[:, av_mask], out[:, other_agent_mask]
            planning_trajectories.append(self.planning_decoder(av_point_query).permute(1, 0, 2).contiguous())
            
            if self.iterative_centerline:
                av_point_query = self.aggr_embed_centerline[indx](torch.cat((av_point_query, centerline_temporal_emb.expand(16, *centerline_temporal_emb.shape).contiguous()), dim=-1))
        planning_trajectories = torch.stack(planning_trajectories)
            
        return {"trajectory": planning_trajectories}