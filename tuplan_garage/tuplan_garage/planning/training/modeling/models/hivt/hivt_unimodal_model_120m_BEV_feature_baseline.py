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
from tuplan_garage.planning.training.modeling.models.hivt.local_encoder_120m_BEV_feature import LocalEncoder, TemporalEncoder, TemporalEncoderCenterline
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import GetInRouteLaneEdgeTotal, GetInRouteLaneEdgeRoute
import math
from tuplan_garage.planning.training.modeling.models.hivt.embedding  import SingleInputEmbedding
from tuplan_garage.planning.training.modeling.models.hivt.unet_parts import *

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
        self.av_temporal_encoder = TemporalEncoder(historical_steps=11,
                                                embed_dim=hidden_dim,
                                                num_heads=8,
                                                dropout=0.1,
                                                num_layers=4)
        self.centerline_temporal_encoder = TemporalEncoderCenterline(historical_steps=120,
                                                embed_dim=hidden_dim,
                                                num_heads=8,
                                                dropout=0.1,
                                                num_layers=4)
        if additional_state:
            self.av_history_embed = SingleInputEmbedding(in_channel=6, out_channel=hidden_dim)
        else:
            self.av_history_embed = SingleInputEmbedding(in_channel=2, out_channel=hidden_dim)
        self.av_point_proj = nn.Linear(hidden_dim, 16 * hidden_dim)
        self.centerline_embed = SingleInputEmbedding(in_channel=2, out_channel=hidden_dim)
        self.aggr_embed = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.LayerNorm(hidden_dim*2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim))

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
            
        self.bev_attn_module = nn.ModuleList(
            [nn.TransformerDecoderLayer(hidden_dim, 8, dim_feedforward=hidden_dim*2, dropout=0.1, batch_first=False) for i in range(self.num_layers)])
        
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

        #AV history embedding
        if self.additional_state:
            av_history_emb = self.av_history_embed(input.x[input.av_index, :11]).permute(1, 0, 2).contiguous()
        else:
            av_history_emb = self.av_history_embed(input.x[input.av_index, :11, :2]).permute(1, 0, 2).contiguous()
        av_temporal_emb = self.av_temporal_encoder(av_history_emb, input.padding_mask[input.av_index, :11])
        av_point_query = self.av_point_proj(av_temporal_emb).view(-1, 16, self.hidden_dim).permute(1, 0, 2).contiguous()
        
        #Centerline embedding
        centerline_emb = self.centerline_embed(torch.tensor(input.centerline).to(av_history_emb)[..., :2]).permute(1, 0, 2).contiguous()
        centerline_temporal_emb = self.centerline_temporal_encoder(centerline_emb)
        
        av_point_query = self.aggr_embed(torch.cat((av_point_query, centerline_temporal_emb.expand(16, *centerline_temporal_emb.shape).contiguous()), dim=-1))
        
        #0: Lane, 1: RoadBlock, 2: Static, 3: TrafficLight, 4: RouteBlock
        bev_agent = input.bev_agent.view(batch_size, -1, self.resolution, self.resolution).contiguous()
        if self.map_channel_dimension == 15:
            bev_map = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, :4].contiguous()
            
            if self.occupied_area:
                bev_map[:, 3, :, :] += (~bev_map[:, 1, :, :].to(torch.bool)).to(torch.int)
                bev_map[:, 3, :, :] = (~bev_map[:, 3, :, :].to(torch.bool)).to(torch.int)
            
        elif self.map_channel_dimension == 12:
            bev_map = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, 0:1].contiguous()
        elif self.map_channel_dimension == 13:
            bev_map = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, :2].contiguous()
        elif self.map_channel_dimension == 14:
            bev_map = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, [0, 1, 3]].contiguous()
            
        bev_env = torch.cat((bev_map, bev_agent), dim=1)
        bev_cnn_feature = self.bev_cnn(bev_env)
        bev_cnn_downsample_feature = self.bev_downsample(bev_cnn_feature) 
        bev_feature = self.bev_upsample(bev_cnn_downsample_feature, bev_cnn_feature) #torch.Size([32, 64, 100, 100])
        bev_embed = bev_feature.view(batch_size, bev_feature.shape[1], -1).permute(2, 0, 1).contiguous()
        bev_embed = self.bev_embed(bev_embed)
        bev_embed = bev_embed + self.bev_pos_embed
        
        bev_attn_mask = input.bev_map.view(batch_size, -1, self.resolution, self.resolution)[:, 4].view(batch_size, -1).contiguous()
        bev_attn_mask = ~bev_attn_mask.type(torch.bool)
        
        planning_trajectories = []
        for indx in range(self.num_layers):
            av_point_query = self.bev_attn_module[indx](av_point_query, bev_embed, memory_key_padding_mask=bev_attn_mask)
            planning_trajectories.append(self.planning_decoder(av_point_query).permute(1, 0, 2).contiguous())
            
            if self.iterative_centerline:
                av_point_query = self.aggr_embed_centerline[indx](torch.cat((av_point_query, centerline_temporal_emb.expand(16, *centerline_temporal_emb.shape).contiguous()), dim=-1))
                
        planning_trajectories = torch.stack(planning_trajectories)
            
        return {"trajectory": planning_trajectories} # mode, batch, 16, 3