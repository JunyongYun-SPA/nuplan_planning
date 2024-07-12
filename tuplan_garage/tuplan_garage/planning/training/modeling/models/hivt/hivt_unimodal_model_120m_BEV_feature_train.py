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
from tuplan_garage.planning.training.modeling.models.hivt.local_encoder_120m_BEV_feature import LocalEncoder
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import GetInRouteLaneEdgeTotal, GetInRouteLaneEdgeRoute
import math
from tuplan_garage.planning.training.modeling.models.hivt.embedding  import SingleInputEmbedding

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
        
        self.num_modes = 16

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
        self.future_decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2))

        self.rotate = True
        self.device = 'cuda'
        
        self.aggr_embed = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True))
        self.centerline_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True))
        self.centerline_embed2 = nn.Sequential(
            nn.Linear(120*64, 120*64*2),
            nn.LayerNorm(120*64*2),
            nn.ReLU(inplace=True),
            nn.Linear(120*64*2, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True))
        self.query_embed2 = nn.Sequential(
            nn.Linear(64, 64*2),
            nn.LayerNorm(64*2),
            nn.ReLU(inplace=True),
            nn.Linear(64*2, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True))
        
        self.map_embed = SingleInputEmbedding(in_channel=1, out_channel=64)
        self.agent_embed = SingleInputEmbedding(in_channel=11, out_channel=64)
        
        self.in_query_fuser = SingleInputEmbedding(in_channel=64*2, out_channel=64)
        self.out_query_fuser = SingleInputEmbedding(in_channel=64*3, out_channel=64)
        
        # map_attn_module_layer = nn.TransformerDecoderLayer(64, 8, dim_feedforward=64*2, dropout=0.1, batch_first=False)
        # self.map_attn_module = nn.TransformerDecoder(map_attn_module_layer, 3)
        
        # agent_attn_module_layer = nn.TransformerDecoderLayer(64, 8, dim_feedforward=64*2, dropout=0.1, batch_first=False)
        # self.agent_attn_module = nn.TransformerDecoder(agent_attn_module_layer, 3)
        
        self.num_layers = 3
        self.map_attn_module = nn.ModuleList(
            [nn.TransformerDecoderLayer(64, 8, dim_feedforward=64*2, dropout=0.1, batch_first=False) for i in range(self.num_layers)])
        self.agent_attn_module = nn.ModuleList(
            [nn.TransformerDecoderLayer(64, 8, dim_feedforward=64*2, dropout=0.1, batch_first=False) for i in range(self.num_layers)])

        self.pos_embed = nn.Parameter(torch.Tensor(1, 120, 64))
        nn.init.normal_(self.pos_embed, mean=0., std=.02)
        
        self.map_pos_embed = nn.Parameter(torch.Tensor(10000, 1, 64))
        nn.init.normal_(self.map_pos_embed, mean=0., std=.02)
        
        self.agent_pos_embed = nn.Parameter(torch.Tensor(10000, 1, 64))
        nn.init.normal_(self.agent_pos_embed, mean=0., std=.02)
        
        self.agent_temporal_pos_embed = nn.Parameter(torch.Tensor(11, 1, 1))
        nn.init.normal_(self.agent_temporal_pos_embed, mean=0., std=.02)
        
        self.query_pos_embed = nn.Parameter(torch.Tensor(16, 1, 64))
        nn.init.normal_(self.query_pos_embed, mean=0., std=.02)

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
        
        try:
            batch_size = input.av_index.shape[0]
        except:
            batch_size = 1
        av_point_query_embed = self.aggr_embed(torch.cat((global_embed, local_embed.expand(self.num_modes, *local_embed.shape)), dim=-1)) 
        av_point_query_embed = av_point_query_embed[:, input.av_index] #.permute(1, 0, 2) # (B, N, k, D): (32, 16, 64)
        
        centerline = self.centerline_embed(torch.tensor(input.centerline).to(av_point_query_embed)[..., :2])
        centerline = self.centerline_embed2((centerline + self.pos_embed).view(batch_size, -1)).unsqueeze(0)
        
        av_point_query_embed = self.query_embed2(av_point_query_embed + self.query_pos_embed)
        av_point_query_embed = self.in_query_fuser(torch.cat((av_point_query_embed, centerline.repeat(self.num_modes, 1, 1)), dim=-1))
        
        bev_map = input.bev_map.view(batch_size, 2, 100, 100)[:, 0]
        bev_embed = bev_map.unsqueeze(-1)
        bev_embed = self.map_embed(bev_embed.view(batch_size, -1, 1)) #(32, 40000, 64)
        bev_embed = bev_embed.permute(1, 0, 2) #(40000, 32, 64)
        bev_embed = bev_embed + self.map_pos_embed
        
        bev_agent = input.bev_agent.view(batch_size, 11, 100, 100)       
        agent_embed = bev_agent.view(batch_size, 11, -1).permute(1, 0, 2) #(11, 32, 40000)
        agent_embed = self.agent_embed((agent_embed + self.agent_temporal_pos_embed).permute(1, 2, 0)).permute(1, 0, 2) #(40000, 32, 64)
        agent_embed = agent_embed + self.agent_pos_embed
        
        attn_mask = input.bev_map.view(batch_size, 2, 100, 100)[:, 1].view(batch_size, -1)
        attn_mask = ~attn_mask.type(torch.bool)
        
        for indx in range(self.num_layers):
            plan_query_map = self.map_attn_module[indx](av_point_query_embed, bev_embed, memory_key_padding_mask=attn_mask)   # [16, 2, 64]
            plan_query_agent = self.agent_attn_module[indx](av_point_query_embed, agent_embed, memory_key_padding_mask=attn_mask)   # [16, 2, 64]
            
            av_point_query_embed = self.out_query_fuser(torch.cat((plan_query_map, plan_query_agent, av_point_query_embed), dim=-1))
        
        y_hat = self.future_decoder(av_point_query_embed).permute(1, 0, 2)

        return {"trajectory": y_hat[:, :, :]} # mode, batch, 16, 3
    
    def pos2posemb2d(self, pos, num_pos_feats=120, temperature=10000):
        """
        Convert 2D position into positional embeddings.

        Args:
            pos (torch.Tensor): Input 2D position tensor.
            num_pos_feats (int, optional): Number of positional features. Default is 128.
            temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

        Returns:
            torch.Tensor: Positional embeddings tensor.
        """
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb