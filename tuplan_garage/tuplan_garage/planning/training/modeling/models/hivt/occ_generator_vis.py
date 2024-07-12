# Copyright (c) 2022, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from torch_geometric.utils import subgraph

from tuplan_garage.planning.training.modeling.models.hivt.embedding import MultipleInputEmbedding
from tuplan_garage.planning.training.modeling.models.hivt.embedding  import SingleInputEmbedding
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import DistanceDropEdge, DistanceDropEdgeOtherAgents
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import GetInRouteLaneEdgeTotal, GetInRouteLaneEdgeRoute
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import TemporalData
from tuplan_garage.planning.training.modeling.models.hivt.hivt_utils import init_weights
import numpy as np


# import copy
# from typing import Optional, List
# import math

# import torch
# import torch.nn.functional as F
# from torch import nn, Tensor
# from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

# from tuplan_garage.planning.training.modeling.models.hivt.util.misc import inverse_sigmoid
# from tuplan_garage.planning.training.modeling.models.hivt.models.ops.modules import MSDeformAttn

# class OccGenerator(nn.Module):

#     def __init__(self,
#                  historical_steps: int,
#                  node_dim: int,
#                  edge_dim: int,
#                  embed_dim: int,
#                  num_heads: int = 8,
#                  dropout: float = 0.1,
#                  num_temporal_layers: int = 4,
#                  local_radius: float = 50,
#                  parallel: bool = False) -> None:
#         super(OccGenerator, self).__init__()
#         self.historical_steps = historical_steps
#         self.parallel = parallel
        
#         self.transformer = DeformableTransformer(d_model=embed_dim, nhead=8,
#                  num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
#                  activation="relu", return_intermediate_dec=False,
#                  num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
#                  two_stage=False)
        
#         self.occ_query = nn.Parameter(torch.Tensor(16, 200, 200, 64))
#         self.center_embed = SingleInputEmbedding(in_channel=64, out_channel=embed_dim*2)
#         self.memory_embed = SingleInputEmbedding(in_channel=28, out_channel=embed_dim)
#         self.offset_decoder = nn.Linear(embed_dim, 5*2)
        
#         self.occupancy_decoer = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.LayerNorm(embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dim, 16))
        

#     def forward(self, data: TemporalData, occupancy_map) -> torch.Tensor:
        
#         occ_query_u1 = self.center_embed(self.occ_query) #torch.Size([32, 200, 200, 64])
#         bs, _, _, C = occ_query_u1.shape
#         occupancy_map = self.memory_embed(occupancy_map.permute(0, 2, 3, 1)) #torch.Size([32, 28, 200, 200])
#         occ_query_u1 = occ_query_u1.reshape(bs, -1, C)
#         occupancy_map= occupancy_map.reshape(bs, -1, C//2)
#         hs, init_reference, inter_references= self.transformer(occ_query_u1, occupancy_map)

#         outputs_classes = []
#         outputs_coords = []
#         for lvl in range(hs.shape[0]):
#             if lvl == 0:
#                 reference = init_reference
#             else:
#                 reference = inter_references[lvl - 1]
#             reference = inverse_sigmoid(reference)
#             outputs_class = self.class_embed[lvl](hs[lvl])
#             tmp = self.bbox_embed[lvl](hs[lvl])
#             if reference.shape[-1] == 4:
#                 tmp += reference
#             else:
#                 assert reference.shape[-1] == 2
#                 tmp[..., :2] += reference
#             outputs_coord = tmp.sigmoid()
#             outputs_classes.append(outputs_class)
#             outputs_coords.append(outputs_coord)
#         outputs_class = torch.stack(outputs_classes)
#         outputs_coord = torch.stack(outputs_coords)
        
#         # offset = self.offset_decoder(occ_query_u1).view(32, 200, 200, 5, 2) #torch.Size([32, 200, 200, 10])
#         # key_index = torch.tensor([[[[j, i, k] for i in range(200)] for j in range(200)] for k in range(32)]).unsqueeze(-2).repeat(1, 1, 1, 5, 1).to(offset.device)
#         # key_index_xy = (key_index[:, :, :, :, :2] + offset).type(torch.int)
#         # edge_index_key = key_index_xy[:, :, :, :, 0]*200 + key_index_xy[:, :, :, :, 1] + key_index[:, :, :, :, 2]*40000
#         # edge_index_key = edge_index_key.flatten()
#         # edge_index = torch.stack((edge_index_key, torch.arange(1280000).unsqueeze(-1).repeat(1, 5).flatten().to(edge_index_key.device)), dim=-1).permute(1, 0)
        
#         # out = self.cross_attention(x=(occupancy_map.to(occ_query_u1.device), occ_query_u1), edge_index=edge_index)
#         # out = self.occupancy_decoer(out)
#         # generated_occupancy_map = torch.softmax(out, dim=-1)
        
#         return out

# class DeformableTransformer(nn.Module):
#     def __init__(self, d_model=256, nhead=8,
#                  num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
#                  activation="relu", return_intermediate_dec=False,
#                  num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
#                  two_stage=False, two_stage_num_proposals=300):
#         super().__init__()

#         self.d_model = d_model
#         self.nhead = nhead
#         self.two_stage = two_stage
#         self.two_stage_num_proposals = two_stage_num_proposals

#         # encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
#         #                                                   dropout, activation,
#         #                                                   num_feature_levels, nhead, enc_n_points)
#         # self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

#         decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
#                                                           dropout, activation,
#                                                           num_feature_levels, nhead, dec_n_points)
#         self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

#         self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

      
#         self.reference_points = nn.Linear(d_model, 2)

#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for m in self.modules():
#             if isinstance(m, MSDeformAttn):
#                 m._reset_parameters()
#         if not self.two_stage:
#             xavier_uniform_(self.reference_points.weight.data, gain=1.0)
#             constant_(self.reference_points.bias.data, 0.)
#         normal_(self.level_embed)


#     def get_valid_ratio(self, mask):
#         _, H, W = mask.shape
#         valid_H = torch.sum(~mask[:, :, 0], 1)
#         valid_W = torch.sum(~mask[:, 0, :], 1)
#         valid_ratio_h = valid_H.float() / H
#         valid_ratio_w = valid_W.float() / W
#         valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
#         return valid_ratio

#     def forward(self, query_embed, memory):

#         bs, _, c = memory.shape
    
#         query_embed, tgt = torch.split(query_embed, c, dim=-1)
#         query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
#         tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
#         reference_points = self.reference_points(query_embed).sigmoid()
#         init_reference_out = reference_points

#         # decoder
#         hs, inter_references = self.decoder(tgt, reference_points, memory,
#                                             torch.tensor([[200, 200]], device=query_embed.device), torch.tensor([0], device=query.device), query_embed)

#         inter_references_out = inter_references
#         if self.two_stage:
#             return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
#         return hs, init_reference_out, inter_references_out, None, None


# class DeformableTransformerEncoderLayer(nn.Module):
#     def __init__(self,
#                  d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=8, n_points=4):
#         super().__init__()

#         # self attention
#         self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout2 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout3 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, src):
#         src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
#         src = src + self.dropout3(src2)
#         src = self.norm2(src)
#         return src

#     def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
#         # self attention
#         src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)

#         # ffn
#         src = self.forward_ffn(src)

#         return src


# class DeformableTransformerEncoder(nn.Module):
#     def __init__(self, encoder_layer, num_layers):
#         super().__init__()
#         self.layers = _get_clones(encoder_layer, num_layers)
#         self.num_layers = num_layers

#     @staticmethod
#     def get_reference_points(spatial_shapes, valid_ratios, device):
#         reference_points_list = []
#         for lvl, (H_, W_) in enumerate(spatial_shapes):

#             ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
#                                           torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
#             ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
#             ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
#             ref = torch.stack((ref_x, ref_y), -1)
#             reference_points_list.append(ref)
#         reference_points = torch.cat(reference_points_list, 1)
#         reference_points = reference_points[:, :, None] * valid_ratios[:, None]
#         return reference_points

#     def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
#         output = src
#         reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
#         for _, layer in enumerate(self.layers):
#             output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

#         return output


# class DeformableTransformerDecoderLayer(nn.Module):
#     def __init__(self, d_model=256, d_ffn=1024,
#                  dropout=0.1, activation="relu",
#                  n_levels=4, n_heads=8, n_points=4):
#         super().__init__()
#         # cross attention
#         self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
#         self.dropout1 = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(d_model)

#         # self attention
#         self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.norm2 = nn.LayerNorm(d_model)

#         # ffn
#         self.linear1 = nn.Linear(d_model, d_ffn)
#         self.activation = _get_activation_fn(activation)
#         self.dropout3 = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(d_ffn, d_model)
#         self.dropout4 = nn.Dropout(dropout)
#         self.norm3 = nn.LayerNorm(d_model)

#     @staticmethod
#     def with_pos_embed(tensor, pos):
#         return tensor if pos is None else tensor + pos

#     def forward_ffn(self, tgt):
#         tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
#         tgt = tgt + self.dropout4(tgt2)
#         tgt = self.norm3(tgt)
#         return tgt

#     def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
#         # self attention
#         q = k = self.with_pos_embed(tgt, query_pos)
#         tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
#         tgt = tgt + self.dropout2(tgt2)
#         tgt = self.norm2(tgt)

#         # cross attention
#         tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
#                                reference_points,
#                                src, src_spatial_shapes, level_start_index, src_padding_mask)
#         tgt = tgt + self.dropout1(tgt2)
#         tgt = self.norm1(tgt)

#         # ffn
#         tgt = self.forward_ffn(tgt)

#         return tgt


# class DeformableTransformerDecoder(nn.Module):
#     def __init__(self, decoder_layer, num_layers, return_intermediate=False):
#         super().__init__()
#         self.layers = _get_clones(decoder_layer, num_layers)
#         self.num_layers = num_layers
#         self.return_intermediate = return_intermediate
#         # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
#         reg_branch = []
#         for _ in range(2):
#             reg_branch.append(nn.Linear(64, 64))
#             reg_branch.append(nn.ReLU())
#         reg_branch.append(nn.Linear(64, 2))
#         reg_branch = nn.Sequential(*reg_branch)
#         self.bbox_embed = nn.ModuleList(
#                 [reg_branch for _ in range(num_layers)])
#         self.class_embed = None

#     def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index,
#                 query_pos=None, src_padding_mask=None):
#         output = tgt

#         intermediate = []
#         intermediate_reference_points = []
#         for lid, layer in enumerate(self.layers):
            
#             reference_points_input = reference_points[:, :, None]
#             output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

#             # hack implementation for iterative bounding box refinement
#             if self.bbox_embed is not None:
#                 tmp = self.bbox_embed[lid](output)
            
#                 new_reference_points = tmp
#                 new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
#                 new_reference_points = new_reference_points.sigmoid()
#                 reference_points = new_reference_points.detach()

#             if self.return_intermediate:
#                 intermediate.append(output)
#                 intermediate_reference_points.append(reference_points)

#         if self.return_intermediate:
#             return torch.stack(intermediate), torch.stack(intermediate_reference_points)

#         return output, reference_points


# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# def _get_activation_fn(activation):
#     """Return an activation function given a string"""
#     if activation == "relu":
#         return F.relu
#     if activation == "gelu":
#         return F.gelu
#     if activation == "glu":
#         return F.glu
#     raise RuntimeError(F"activation should be relu/gelu, not {activation}.")



# class CrossAttention(MessagePassing):

#     def __init__(self,
#                  historical_steps: int,
#                  node_dim: int,
#                  edge_dim: int,
#                  embed_dim: int,
#                  num_heads: int = 8,
#                  dropout: float = 0.1,
#                  parallel: bool = False,
#                  **kwargs) -> None:
#         super(CrossAttention, self).__init__(aggr='add', node_dim=0, **kwargs)
#         self.historical_steps = historical_steps
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.parallel = parallel

#         self.center_embed = SingleInputEmbedding(in_channel=embed_dim, out_channel=embed_dim)
#         self.nbr_embed = SingleInputEmbedding(in_channel=embed_dim, out_channel=embed_dim)
#         self.lin_q = nn.Linear(embed_dim, embed_dim)
#         self.lin_k = nn.Linear(embed_dim, embed_dim)
#         self.lin_v = nn.Linear(embed_dim, embed_dim)
#         self.lin_self = nn.Linear(embed_dim, embed_dim)
#         self.attn_drop = nn.Dropout(dropout)
#         self.lin_ih = nn.Linear(embed_dim, embed_dim)
#         self.lin_hh = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm2 = nn.LayerNorm(embed_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * 4, embed_dim),
#             nn.Dropout(dropout))
#         self.bos_token = nn.Parameter(torch.Tensor(historical_steps, embed_dim))
#         nn.init.normal_(self.bos_token, mean=0., std=.02)
#         self.apply(init_weights)

#     def forward(self,
#                 x: torch.Tensor,
#                 edge_index: Adj,
#                 size: Size = None) -> torch.Tensor:
#         x_key, x_query = x
        
#         x_key = x_key.permute(0, 2, 3, 1).reshape(-1, 30)
#         x_query = x_query.reshape(-1, 64)
        
#         # center_embed = self.center_embed(x_query)
#         x_query = x_query + self._mha_block(self.norm1(x_query), x_key, edge_index, size)
#         x_query = x_query + self._ff_block(self.norm2(x_query))
#         return x_query

#     def message(self,
#                 edge_index: Adj,
#                 center_embed_i: torch.Tensor,
#                 x_j: torch.Tensor,
#                 index: torch.Tensor,
#                 ptr: OptTensor,
#                 size_i: Optional[int]) -> torch.Tensor:

#         nbr_embed = self.nbr_embed(x_j)
        
#         query = self.lin_q(center_embed_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         key = self.lin_k(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         value = self.lin_v(nbr_embed).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         scale = (self.embed_dim // self.num_heads) ** 0.5
#         alpha = (query * key).sum(dim=-1) / scale
#         alpha = softmax(alpha, index, ptr, size_i)
#         alpha = self.attn_drop(alpha)
#         return value * alpha.unsqueeze(-1)

#     def update(self,
#                inputs: torch.Tensor,
#                center_embed: torch.Tensor) -> torch.Tensor:
#         inputs = inputs.view(-1, self.embed_dim)
#         gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(center_embed))
#         return inputs + gate * (self.lin_self(center_embed) - inputs)

#     def _mha_block(self,
#                    center_embed: torch.Tensor,
#                    x: torch.Tensor,
#                    edge_index: Adj,
#                    size: Size) -> torch.Tensor:
#         center_embed = self.out_proj(self.propagate(edge_index=edge_index, x=x, center_embed=center_embed,
#                                                     size=size))
#         return self.proj_drop(center_embed)

#     def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         return self.mlp(x)
    
def OccupancyGenerator(input, y_hat, pi):
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
    
    for batch_idx, av_idx in enumerate(input.av_index):
        others_indx = edge_index[0][torch.where(edge_index[1].cpu() == av_idx.cpu())[0]]
        lane_indx = lane_edge_index[0][torch.where(lane_edge_index[1].cpu() == av_idx.cpu())[0]]
        # currne_agent_indx = others_indx.clone()[~input.padding_mask[others_indx, 10]]
        
        for agent_idx in others_indx.clone()[~input.padding_mask[others_indx, 10]]:
            lane_indx = lane_edge_index[0][torch.where(lane_edge_index[1].cpu() == agent_idx.cpu())[0]]
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
    
    return occupancy_map, occupancy_map_gt

def OccupancyGeneratorParallelAddLaneType(input, y_hat, pi):
    modal_shape, agent_shape, future_time_shape, coord_shape = y_hat.shape
    batch_shape = input.av_index.shape[0]
    
    occupancy_resolution = 0.5
    occupancy_size = 200
    occupancy_range = int(occupancy_size * occupancy_resolution)
    occupancy_map_env = torch.zeros(batch_shape, 3, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_history = torch.zeros(batch_shape, 11, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_future = torch.zeros(batch_shape, future_time_shape, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_gt = torch.zeros(batch_shape, future_time_shape, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    drop_edge_av = DistanceDropEdge(occupancy_range/2)
    input['edge_attr'] = \
            input['positions'][input['edge_index'][0], 10, :2] - input['positions'][input['edge_index'][1], 10, :2]
    edge_index, _ = drop_edge_av(input['edge_index'], input['edge_attr'])
    lane_edge_index, lane_attr_origin = drop_edge_av(input['lane_actor_index'], input['lane_actor_vectors'])
    
    av_mask = (edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    av_mask_index = torch.where(av_mask ==True)[0] #torch.Size([1419])
    others_indx = edge_index[0][av_mask_index]
    static_indx = input.edge_index_static[0]
    others_indx_static = input.edge_index_static[1]
    
    batch_idx_for_agent = others_indx.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_agent = torch.where((batch_idx_for_agent >= 0), 1, 0).sum(dim=-1)
    
    batch_idx_for_static = others_indx_static.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_static = torch.where((batch_idx_for_static >= 0), 1, 0).sum(dim=-1)
    static_history = input.static_positions[static_indx][:, 10, :2] #(456, 11, 2)
    occ_mask_static = (abs(static_history[:, 0]) < 50) * (abs(static_history[:, 1]) < 50)
    static_history = torch.where((((~input.padding_mask_static[others_indx_static, 10]) * occ_mask_static).unsqueeze(-1)),
                            static_history,
                            50*torch.ones(others_indx_static.shape[0], 2).to(static_history.device))
    occupancy_x_static, occupancy_y_static = \
                (static_history / occupancy_resolution).type(torch.int)[:, 0], \
                    (static_history / occupancy_resolution).type(torch.int)[:, 1]
    occupancy_y_static[~((~input.padding_mask_static[others_indx_static, 10]) * occ_mask_static)] = \
        -1 * occupancy_y_static[~((~input.padding_mask_static[others_indx_static, 10]) * occ_mask_static)]
    occupancy_map_env[batch_idx_for_static, \
        torch.tensor(2).repeat(batch_idx_for_static.shape[0]), \
            (occupancy_range - occupancy_y_static).type(torch.long), \
                (occupancy_x_static + occupancy_range).type(torch.long)] = 1
    
    traffic_red_mask = torch.where(input.traffic_controls_red == 1)[0]
    traffic_red_mask_total = (lane_edge_index[1].unsqueeze(1)==input.av_index).any(dim=1)  * (lane_edge_index[0].unsqueeze(1)==traffic_red_mask).any(dim=1)
    lane_mask = (lane_edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    lane_mask_index = torch.where(lane_mask ==True)[0] #torch.Size([1419])
    lane_red_mask_index = torch.where(traffic_red_mask_total==True)[0]
    lanes_indx = lane_edge_index[1][lane_mask_index]
    lanes_red_indx = lane_edge_index[1][lane_red_mask_index]
    lane_attr = lane_attr_origin[lane_mask_index] + input.positions[lanes_indx, 10, :2]
    lane_attr_red = lane_attr_origin[lane_red_mask_index] + input.positions[lanes_red_indx, 10, :2]

    batch_idx_for_lane = lanes_indx.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_lane = torch.where((batch_idx_for_lane >= 0), 1, 0).sum(dim=-1)
    occ_mask_lane = (abs(lane_attr[:, 0]) < 50) * (abs(lane_attr[:, 1]) < 50)
    lane = torch.where(((occ_mask_lane).unsqueeze(-1)),
                            lane_attr,
                            50*torch.ones(lane_attr.shape[0], 2).to(lane_attr.device))
    occupancy_x_lane, occupancy_y_lane = \
                (lane / occupancy_resolution).type(torch.int)[:, 0], \
                    (lane / occupancy_resolution).type(torch.int)[:, 1]
    occupancy_y_lane[~(occ_mask_lane)] = \
        -1 * occupancy_y_lane[~(occ_mask_lane)]
    occupancy_map_env[batch_idx_for_lane, \
        torch.arange(1).repeat(batch_idx_for_lane.shape[0]), \
            (occupancy_range - occupancy_y_lane).type(torch.long), \
                (occupancy_x_lane + occupancy_range).type(torch.long)] = 1
    
    batch_idx_for_lane_red = lanes_red_indx.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_lane_red = torch.where((batch_idx_for_lane_red >= 0), 1, 0).sum(dim=-1)
    occ_mask_lane_red = (abs(lane_attr_red[:, 0]) < 50) * (abs(lane_attr_red[:, 1]) < 50)
    lane_red = torch.where(((occ_mask_lane_red).unsqueeze(-1)),
                            lane_attr_red,
                            50*torch.ones(lane_attr_red.shape[0], 2).to(lane_attr_red.device))
    occupancy_x_lane_red, occupancy_y_lane_red = \
                (lane_red / occupancy_resolution).type(torch.int)[:, 0], \
                    (lane_red / occupancy_resolution).type(torch.int)[:, 1]
    occupancy_y_lane_red[~(occ_mask_lane_red)] = \
        -1 * occupancy_y_lane_red[~(occ_mask_lane_red)]
    occupancy_map_env[batch_idx_for_lane_red, \
        torch.tensor(1).repeat(batch_idx_for_lane_red.shape[0]), \
            (occupancy_range - occupancy_y_lane_red).type(torch.long), \
                (occupancy_x_lane_red + occupancy_range).type(torch.long)] = 1
    
    agent_history = input.positions[others_indx][:, :11, :2] #(456, 11, 2)
    agent_future = input.positions[others_indx][:, 11:, :2] #(456, 16, 2)
    
    rotate_mat = torch.empty(input.num_nodes, 2, 2).to(y_hat.device)
    sin_vals = torch.sin(input['rotate_angles'])
    cos_vals = torch.cos(input['rotate_angles'])
    rotate_mat[:, 0, 0] = cos_vals
    rotate_mat[:, 0, 1] = sin_vals
    rotate_mat[:, 1, 0] = -sin_vals
    rotate_mat[:, 1, 1] = cos_vals
    
    y_hat_av_centric = torch.bmm(y_hat[..., :2].reshape(-1, future_time_shape, 2).to(torch.float32), rotate_mat.repeat(modal_shape, 1, 1)).reshape(modal_shape, -1, future_time_shape, 2)
            
    agent_pred = y_hat_av_centric[:, others_indx][:, :, :, :2] + input.positions[others_indx][:, 10, :2].unsqueeze(0).unsqueeze(-2) #(1, 456, 16, 2)
    agent_pred_pi = F.softmax(pi[others_indx], dim=-1).unsqueeze(-1).repeat(1, 1, 16).reshape(-1)
    
    occ_mask = (abs(agent_history[:, :, 0]) < 50) * (abs(agent_history[:, :, 1]) < 50)
    occ_mask_future = (abs(agent_future[:, :, 0]) < 50) * (abs(agent_future[:, :, 1]) < 50)
    occ_mask_pred = (abs(agent_pred[:, :, :, 0]) < 50) * (abs(agent_pred[:, :, :, 1]) < 50)
    
    agent_history = torch.where((((~input.padding_mask[others_indx, :11]) * occ_mask).unsqueeze(-1)),
                            agent_history,
                            50*torch.ones(others_indx.shape[0], 11, 2).to(agent_history.device))
    agent_future = torch.where((((~input.padding_mask[others_indx, 11:]) * occ_mask_future).unsqueeze(-1)),
                            agent_future,
                            50*torch.ones(others_indx.shape[0], 16, 2).to(agent_future.device))
    agent_pred = torch.where((((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred).unsqueeze(-1)),
                            agent_pred,
                            50*torch.ones(modal_shape, others_indx.shape[0], 16, 2).to(agent_pred.device))
    
    occupancy_x, occupancy_y = \
                (agent_history / occupancy_resolution).type(torch.int)[:, :, 0], \
                    (agent_history / occupancy_resolution).type(torch.int)[:, :, 1]
    occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)] = \
        -1 * occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)]
    occupancy_map_history[batch_idx_for_agent.unsqueeze(-1).repeat(1, 11).reshape(-1), \
        torch.arange(11).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y).type(torch.long).reshape(-1), \
                (occupancy_x + occupancy_range).type(torch.long).reshape(-1)] = 1
    
    occupancy_x_future, occupancy_y_future = \
                (agent_future / occupancy_resolution).type(torch.int)[:, :, 0], \
                    (agent_future / occupancy_resolution).type(torch.int)[:, :, 1]
    occupancy_y_future[~((~input.padding_mask[others_indx, 11:]) * occ_mask_future)] = \
        -1 * occupancy_y_future[~((~input.padding_mask[others_indx, 11:]) * occ_mask_future)]
    occupancy_map_gt[batch_idx_for_agent.unsqueeze(-1).repeat(1, 16).reshape(-1), \
        torch.arange(16).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y_future).type(torch.long).reshape(-1), \
                (occupancy_x_future + occupancy_range).type(torch.long).reshape(-1)] = 1
    
    occupancy_x_pred, occupancy_y_pred = \
                (agent_pred / occupancy_resolution).type(torch.int)[:, :, :, 0], \
                    (agent_pred / occupancy_resolution).type(torch.int)[:, :, :, 1]
    occupancy_y_pred[~((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred)] = \
        -1 * occupancy_y_pred[~((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred)]

    a = batch_idx_for_agent.unsqueeze(0).unsqueeze(-1).repeat(occupancy_x_pred.shape[0], 1, 16).permute(2, 0, 1).reshape(-1)
    b = torch.arange(16).unsqueeze(0).unsqueeze(1).repeat(occupancy_x_pred.shape[0], occupancy_x_pred.shape[1], 1).permute(2, 0, 1).reshape(-1).to(a.device)
    c = (occupancy_range - occupancy_y_pred).type(torch.long).permute(2, 0, 1).reshape(-1)
    d = (occupancy_x_pred + occupancy_range).type(torch.long).permute(2, 0, 1).reshape(-1)
    
    temp_list = torch.stack((a, b.to(a.device), c, d, agent_pred_pi), dim=-1).tolist()
    temp_list.sort(key=lambda x: (x[0], x[1], x[2], x[3], -x[4]))

    diff = torch.tensor(temp_list)[0:-1, :4] - torch.tensor(temp_list)[1:, :4]
    duplicated_idx = torch.where(torch.any(diff, dim=-1) == True)[0]
    duplicated_idx = torch.cat((duplicated_idx, (duplicated_idx[-1]+1).unsqueeze(0)))
    occ_idx, occ_value = torch.tensor(temp_list)[duplicated_idx][:, :4].to(torch.int64), torch.tensor(temp_list)[duplicated_idx][:, 4]
    o_a, o_b, o_c, o_d = occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2], occ_idx[:, 3]
    occupancy_map_future[o_a, o_b, o_c, o_d] = occ_value.to(occupancy_map_future.device)
    
    import matplotlib.pyplot as plt
    for batch in range(batch_shape):
        fig1, axes1 = plt.subplots(1, 5, figsize=(50, 10))  # 3행 4열의 subplot 생성
        occupancy_resolution = 0.5
        occupancy_size = 200
        occupancy_range = int(occupancy_size * occupancy_resolution)
        drop_edge_av = DistanceDropEdge(occupancy_range/2)
        input['edge_attr'] = \
                input['positions'][input['edge_index'][0], 10, :2] - input['positions'][input['edge_index'][1], 10, :2]
        edge_index, edge_attr = drop_edge_av(input['edge_index'], input['edge_attr'])
        others_indx = edge_index[0][np.where(edge_index[1].cpu() == input.av_index[batch].cpu())[0]]
        lanes_indx2 = np.where(lane_edge_index[1].cpu() == input.av_index[batch].cpu())[0]
        all_history = input.positions[others_indx][:, :, :2]
        padding_mask = input.padding_mask[others_indx]
        for i, ah in enumerate(all_history):
            axes1[0].plot(ah[:11].cpu()[~padding_mask[i, :11]][:, 0], ah[:11].cpu()[~padding_mask[i, :11]][:, 1], 'k-')
            axes1[0].plot(ah[11:].cpu()[~padding_mask[i, 11:]][:, 0], ah[11:].cpu()[~padding_mask[i, 11:]][:, 1], 'r')
        axes1[0].scatter(lane_attr_origin[lanes_indx2][:, 0].cpu(), lane_attr_origin[lanes_indx2][:, 1].cpu(), c="grey")
        axes1[0].set_xlim([-50, 50])
        axes1[0].set_ylim([-50, 50])

        axes1[1].imshow(occupancy_map_history[:, :, :200, :200][batch].sum(dim=0).cpu().detach().numpy(), cmap='binary')  # 각 층의 occupancy map을 흑백 이미지로 표시
        axes1[1].set_title(f"History")  # subplot 제목 설정
        
        axes1[2].imshow(occupancy_map_future[:, :, :200, :200][batch].sum(dim=0).cpu().detach().numpy(), cmap='binary')  # 각 층의 occupancy map을 흑백 이미지로 표시
        axes1[2].set_title(f"Future")  # subplot 제목 설정
        
        # axes1[3].imshow(occupancy_map_gt[:, :, :200, :200][batch].sum(dim=0).cpu().detach().numpy(), cmap='binary')  # 각 층의 occupancy map을 흑백 이미지로 표시
        # axes1[3].set_title(f"GT")  # subplot 제목 설정
        
        axes1[3].imshow(occupancy_map_env[:, :, :200, :200][batch, 2].cpu().detach().numpy(), cmap='binary')  # 각 층의 occupancy map을 흑백 이미지로 표시
        axes1[3].set_title(f"Lane")  # subplot 제목 설정
        
        axes1[4].imshow(occupancy_map_env[:, :, :200, :200][batch, 0].cpu().detach().numpy(), cmap='binary')  # 각 층의 occupancy map을 흑백 이미지로 표시
        axes1[4].set_title(f"Lane")  # subplot 제목 설정

        plt.tight_layout()  # subplot 간격 조정
        plt.savefig(f'/home/workspace/visualization/pred_{1}.png') 
    
    occupancy_map_history = occupancy_map_history[:, :, :200, :200]
    occupancy_map_future = occupancy_map_future[:, :, :200, :200]
    occupancy_map_gt = occupancy_map_gt[:, :, :200, :200]
    occupancy_map_env = occupancy_map_env[:, :, :200, :200]
    
    occupancy_map = torch.cat((occupancy_map_env, occupancy_map_history, occupancy_map_future), dim=1)
    
    return occupancy_map, occupancy_map_gt

def OccupancyGeneratorParallel(input, y_hat, pi):
    modal_shape, agent_shape, future_time_shape, coord_shape = y_hat.shape
    batch_shape = input.av_index.shape[0]
    
    occupancy_resolution = 0.5
    occupancy_size = 200
    occupancy_range = int(occupancy_size * occupancy_resolution)
    occupancy_map_env = torch.zeros(batch_shape, 1, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_history = torch.zeros(batch_shape, 11, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_future = torch.zeros(batch_shape, future_time_shape, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_gt = torch.zeros(batch_shape, future_time_shape, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    drop_edge_av = DistanceDropEdge(occupancy_range/2)
    input['edge_attr'] = \
            input['positions'][input['edge_index'][0], 10, :2] - input['positions'][input['edge_index'][1], 10, :2]
    edge_index, _ = drop_edge_av(input['edge_index'], input['edge_attr'])
    lane_edge_index, lane_attr_origin = drop_edge_av(input['lane_actor_index'], input['lane_actor_vectors'])
    
    av_mask = (edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    av_mask_index = torch.where(av_mask ==True)[0] #torch.Size([1419])
    others_indx = edge_index[0][av_mask_index]
    
    batch_idx_for_agent = others_indx.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_agent = torch.where((batch_idx_for_agent >= 0), 1, 0).sum(dim=-1)
     
    lane_mask = (lane_edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    lane_mask_index = torch.where(lane_mask ==True)[0] #torch.Size([1419])
    lanes_indx = lane_edge_index[1][lane_mask_index]
    lane_attr = lane_attr_origin[lane_mask_index] + input.positions[lanes_indx, 10, :2]

    batch_idx_for_lane = lanes_indx.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_lane = torch.where((batch_idx_for_lane >= 0), 1, 0).sum(dim=-1)
    occ_mask_lane = (abs(lane_attr[:, 0]) < 50) * (abs(lane_attr[:, 1]) < 50)
    lane = torch.where(((occ_mask_lane).unsqueeze(-1)),
                            lane_attr,
                            50*torch.ones(lane_attr.shape[0], 2).to(lane_attr.device))
    occupancy_x_lane, occupancy_y_lane = \
                (lane / occupancy_resolution).type(torch.int)[:, 0], \
                    (lane / occupancy_resolution).type(torch.int)[:, 1]
    occupancy_y_lane[~(occ_mask_lane)] = \
        -1 * occupancy_y_lane[~(occ_mask_lane)]
    occupancy_map_env[batch_idx_for_lane, \
        torch.arange(1).repeat(batch_idx_for_lane.shape[0]), \
            (occupancy_range - occupancy_y_lane).type(torch.long), \
                (occupancy_x_lane + occupancy_range).type(torch.long)] = 1
    
    agent_history = input.positions[others_indx][:, :11, :2] #(456, 11, 2)
    agent_future = input.positions[others_indx][:, 11:, :2] #(456, 16, 2)
    
    rotate_mat = torch.empty(input.num_nodes, 2, 2).to(y_hat.device)
    sin_vals = torch.sin(input['rotate_angles'])
    cos_vals = torch.cos(input['rotate_angles'])
    rotate_mat[:, 0, 0] = cos_vals
    rotate_mat[:, 0, 1] = sin_vals
    rotate_mat[:, 1, 0] = -sin_vals
    rotate_mat[:, 1, 1] = cos_vals
    
    y_hat_av_centric = torch.bmm(y_hat[..., :2].reshape(-1, future_time_shape, 2).to(torch.float32), rotate_mat.repeat(modal_shape, 1, 1)).reshape(modal_shape, -1, future_time_shape, 2)
            
    agent_pred = y_hat_av_centric[:, others_indx][:, :, :, :2] + input.positions[others_indx][:, 10, :2].unsqueeze(0).unsqueeze(-2) #(1, 456, 16, 2)
    agent_pred_pi = F.softmax(pi[others_indx], dim=-1).unsqueeze(-1).repeat(1, 1, 16).reshape(-1)
    
    occ_mask = (abs(agent_history[:, :, 0]) < 50) * (abs(agent_history[:, :, 1]) < 50)
    occ_mask_future = (abs(agent_future[:, :, 0]) < 50) * (abs(agent_future[:, :, 1]) < 50)
    occ_mask_pred = (abs(agent_pred[:, :, :, 0]) < 50) * (abs(agent_pred[:, :, :, 1]) < 50)
    
    agent_history = torch.where((((~input.padding_mask[others_indx, :11]) * occ_mask).unsqueeze(-1)),
                            agent_history,
                            50*torch.ones(others_indx.shape[0], 11, 2).to(agent_history.device))
    agent_future = torch.where((((~input.padding_mask[others_indx, 11:]) * occ_mask_future).unsqueeze(-1)),
                            agent_future,
                            50*torch.ones(others_indx.shape[0], 16, 2).to(agent_future.device))
    agent_pred = torch.where((((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred).unsqueeze(-1)),
                            agent_pred,
                            50*torch.ones(modal_shape, others_indx.shape[0], 16, 2).to(agent_pred.device))
    
    occupancy_x, occupancy_y = \
                (agent_history / occupancy_resolution).type(torch.int)[:, :, 0], \
                    (agent_history / occupancy_resolution).type(torch.int)[:, :, 1]
    occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)] = \
        -1 * occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)]
    occupancy_map_history[batch_idx_for_agent.unsqueeze(-1).repeat(1, 11).reshape(-1), \
        torch.arange(11).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y).type(torch.long).reshape(-1), \
                (occupancy_x + occupancy_range).type(torch.long).reshape(-1)] = 1
    
    occupancy_x_future, occupancy_y_future = \
                (agent_future / occupancy_resolution).type(torch.int)[:, :, 0], \
                    (agent_future / occupancy_resolution).type(torch.int)[:, :, 1]
    occupancy_y_future[~((~input.padding_mask[others_indx, 11:]) * occ_mask_future)] = \
        -1 * occupancy_y_future[~((~input.padding_mask[others_indx, 11:]) * occ_mask_future)]
    occupancy_map_gt[batch_idx_for_agent.unsqueeze(-1).repeat(1, 16).reshape(-1), \
        torch.arange(16).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y_future).type(torch.long).reshape(-1), \
                (occupancy_x_future + occupancy_range).type(torch.long).reshape(-1)] = 1
    
    occupancy_x_pred, occupancy_y_pred = \
                (agent_pred / occupancy_resolution).type(torch.int)[:, :, :, 0], \
                    (agent_pred / occupancy_resolution).type(torch.int)[:, :, :, 1]
    occupancy_y_pred[~((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred)] = \
        -1 * occupancy_y_pred[~((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred)]

    a = batch_idx_for_agent.unsqueeze(0).unsqueeze(-1).repeat(occupancy_x_pred.shape[0], 1, 16).permute(2, 0, 1).reshape(-1)
    b = torch.arange(16).unsqueeze(0).unsqueeze(1).repeat(occupancy_x_pred.shape[0], occupancy_x_pred.shape[1], 1).permute(2, 0, 1).reshape(-1).to(a.device)
    c = (occupancy_range - occupancy_y_pred).type(torch.long).permute(2, 0, 1).reshape(-1)
    d = (occupancy_x_pred + occupancy_range).type(torch.long).permute(2, 0, 1).reshape(-1)
    
    temp_list = torch.stack((a, b.to(a.device), c, d, agent_pred_pi), dim=-1).tolist()
    temp_list.sort(key=lambda x: (x[0], x[1], x[2], x[3], -x[4]))

    diff = torch.tensor(temp_list)[0:-1, :4] - torch.tensor(temp_list)[1:, :4]
    duplicated_idx = torch.where(torch.any(diff, dim=-1) == True)[0]
    duplicated_idx = torch.cat((duplicated_idx, (duplicated_idx[-1]+1).unsqueeze(0)))
    occ_idx, occ_value = torch.tensor(temp_list)[duplicated_idx][:, :4].to(torch.int64), torch.tensor(temp_list)[duplicated_idx][:, 4]
    o_a, o_b, o_c, o_d = occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2], occ_idx[:, 3]
    occupancy_map_future[o_a, o_b, o_c, o_d] = occ_value.to(occupancy_map_future.device)
    
    occupancy_map_history = occupancy_map_history[:, :, :200, :200]
    occupancy_map_future = occupancy_map_future[:, :, :200, :200]
    occupancy_map_gt = occupancy_map_gt[:, :, :200, :200]
    occupancy_map_env = occupancy_map_env[:, :, :200, :200]
    
    occupancy_map = torch.cat((occupancy_map_env, occupancy_map_history, occupancy_map_future), dim=1)
    
    return occupancy_map, occupancy_map_gt

def OccupancyGeneratorParallelforSimulation(input, y_hat, pi):
    modal_shape, future_time_shape = 6, 16
    batch_shape = 1
    
    occupancy_resolution = 0.5
    occupancy_size = 200
    occupancy_range = int(occupancy_size * occupancy_resolution)
    occupancy_map_env = torch.zeros(batch_shape, 1, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_history = torch.zeros(batch_shape, 11, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_future = torch.zeros(batch_shape, future_time_shape, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_gt = torch.zeros(batch_shape, future_time_shape, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    drop_edge_av = DistanceDropEdge(occupancy_range/2)
    input['edge_attr'] = \
            input['positions'][input['edge_index'][0], 10, :2] - input['positions'][input['edge_index'][1], 10, :2]
    edge_index, _ = drop_edge_av(input['edge_index'], input['edge_attr'])
    lane_edge_index, lane_attr_origin = drop_edge_av(input['lane_actor_index'], input['lane_actor_vectors'])
    
    av_mask = (edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    av_mask_index = torch.where(av_mask ==True)[0] #torch.Size([1419])
    others_indx = edge_index[0][av_mask_index]
    
    # batch_idx_for_agent = others_indx.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    # batch_idx_for_agent = torch.where((batch_idx_for_agent >= 0), 1, 0).sum(dim=-1)
    batch_idx_for_agent = torch.zeros(others_indx.shape[0], dtype=torch.int64) #torch.where((batch_idx_for_agent > 0), 1, 0).sum(dim=-1)
     
    lane_mask = (lane_edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    lane_mask_index = torch.where(lane_mask ==True)[0] #torch.Size([1419])
    lanes_indx = lane_edge_index[1][lane_mask_index]
    lane_attr = lane_attr_origin[lane_mask_index] + input.positions[lanes_indx, 10, :2]

    # batch_idx_for_lane = lanes_indx.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    # batch_idx_for_lane = torch.where((batch_idx_for_lane >= 0), 1, 0).sum(dim=-1)
    batch_idx_for_lane = torch.zeros(lane_attr.shape[0], dtype=torch.int64) #torch.where((batch_idx_for_lane > 0), 1, 0).sum(dim=-1)
    
    occ_mask_lane = (abs(lane_attr[:, 0]) < 50) * (abs(lane_attr[:, 1]) < 50)
    lane = torch.where(((occ_mask_lane).unsqueeze(-1)),
                            lane_attr,
                            50*torch.ones(lane_attr.shape[0], 2).to(lane_attr.device))
    occupancy_x_lane, occupancy_y_lane = \
                (lane / occupancy_resolution).type(torch.int)[:, 0], \
                    (lane / occupancy_resolution).type(torch.int)[:, 1]
    occupancy_y_lane[~(occ_mask_lane)] = \
        -1 * occupancy_y_lane[~(occ_mask_lane)]
    occupancy_map_env[batch_idx_for_lane, \
        torch.arange(1).repeat(batch_idx_for_lane.shape[0]), \
            (occupancy_range - occupancy_y_lane).type(torch.long), \
                (occupancy_x_lane + occupancy_range).type(torch.long)] = 1
    
    agent_history = input.positions[others_indx][:, :11, :2] #(456, 11, 2)
    agent_future = input.positions[others_indx][:, 11:, :2] #(456, 16, 2)
    
    rotate_mat = torch.empty(input.num_nodes, 2, 2).to(y_hat.device)
    sin_vals = torch.sin(input['rotate_angles'])
    cos_vals = torch.cos(input['rotate_angles'])
    rotate_mat[:, 0, 0] = cos_vals
    rotate_mat[:, 0, 1] = sin_vals
    rotate_mat[:, 1, 0] = -sin_vals
    rotate_mat[:, 1, 1] = cos_vals
    
    y_hat_av_centric = torch.bmm(y_hat[..., :2].reshape(-1, future_time_shape, 2).to(torch.float32), rotate_mat.repeat(modal_shape, 1, 1)).reshape(modal_shape, -1, future_time_shape, 2)
            
    agent_pred = y_hat_av_centric[:, others_indx][:, :, :, :2] + input.positions[others_indx][:, 10, :2].unsqueeze(0).unsqueeze(-2) #(1, 456, 16, 2)
    agent_pred_pi = F.softmax(pi[others_indx].to(torch.float32), dim=-1).unsqueeze(-1).repeat(1, 1, 16).reshape(-1)
    
    occ_mask = (abs(agent_history[:, :, 0]) < 50) * (abs(agent_history[:, :, 1]) < 50)
    occ_mask_future = (abs(agent_future[:, :, 0]) < 50) * (abs(agent_future[:, :, 1]) < 50)
    occ_mask_pred = (abs(agent_pred[:, :, :, 0]) < 50) * (abs(agent_pred[:, :, :, 1]) < 50)
    
    agent_history = torch.where((((~input.padding_mask[others_indx, :11]) * occ_mask).unsqueeze(-1)),
                            agent_history,
                            50*torch.ones(others_indx.shape[0], 11, 2).to(agent_history.device))
    agent_future = torch.where((((~input.padding_mask[others_indx, 11:]) * occ_mask_future).unsqueeze(-1)),
                            agent_future,
                            50*torch.ones(others_indx.shape[0], 16, 2).to(agent_future.device))
    agent_pred = torch.where((((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred).unsqueeze(-1)),
                            agent_pred,
                            50*torch.ones(modal_shape, others_indx.shape[0], 16, 2).to(agent_pred.device))
    
    occupancy_x, occupancy_y = \
                (agent_history / occupancy_resolution).type(torch.int)[:, :, 0], \
                    (agent_history / occupancy_resolution).type(torch.int)[:, :, 1]
    occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)] = \
        -1 * occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)]
    occupancy_map_history[batch_idx_for_agent.unsqueeze(-1).repeat(1, 11).reshape(-1), \
        torch.arange(11).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y).type(torch.long).reshape(-1), \
                (occupancy_x + occupancy_range).type(torch.long).reshape(-1)] = 1
    
    occupancy_x_future, occupancy_y_future = \
                (agent_future / occupancy_resolution).type(torch.int)[:, :, 0], \
                    (agent_future / occupancy_resolution).type(torch.int)[:, :, 1]
    occupancy_y_future[~((~input.padding_mask[others_indx, 11:]) * occ_mask_future)] = \
        -1 * occupancy_y_future[~((~input.padding_mask[others_indx, 11:]) * occ_mask_future)]
    occupancy_map_gt[batch_idx_for_agent.unsqueeze(-1).repeat(1, 16).reshape(-1), \
        torch.arange(16).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y_future).type(torch.long).reshape(-1), \
                (occupancy_x_future + occupancy_range).type(torch.long).reshape(-1)] = 1
    
    occupancy_x_pred, occupancy_y_pred = \
                (agent_pred / occupancy_resolution).type(torch.int)[:, :, :, 0], \
                    (agent_pred / occupancy_resolution).type(torch.int)[:, :, :, 1]
    occupancy_y_pred[~((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred)] = \
        -1 * occupancy_y_pred[~((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred)]

    a = batch_idx_for_agent.unsqueeze(0).unsqueeze(-1).repeat(occupancy_x_pred.shape[0], 1, 16).permute(2, 0, 1).reshape(-1)
    b = torch.arange(16).unsqueeze(0).unsqueeze(1).repeat(occupancy_x_pred.shape[0], occupancy_x_pred.shape[1], 1).permute(2, 0, 1).reshape(-1).to(a.device)
    c = (occupancy_range - occupancy_y_pred).type(torch.long).permute(2, 0, 1).reshape(-1)
    d = (occupancy_x_pred + occupancy_range).type(torch.long).permute(2, 0, 1).reshape(-1)
    
    temp_list = torch.stack((a, b.to(a.device), c, d, agent_pred_pi), dim=-1).tolist()
    temp_list.sort(key=lambda x: (x[0], x[1], x[2], x[3], -x[4]))

    diff = torch.tensor(temp_list)[0:-1, :4] - torch.tensor(temp_list)[1:, :4]
    duplicated_idx = torch.where(torch.any(diff, dim=-1) == True)[0]
    duplicated_idx = torch.cat((duplicated_idx, (duplicated_idx[-1]+1).unsqueeze(0)))
    occ_idx, occ_value = torch.tensor(temp_list)[duplicated_idx][:, :4].to(torch.int64), torch.tensor(temp_list)[duplicated_idx][:, 4]
    o_a, o_b, o_c, o_d = occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2], occ_idx[:, 3]
    occupancy_map_future[o_a, o_b, o_c, o_d] = occ_value.to(occupancy_map_future.device)
    
    occupancy_map_history = occupancy_map_history[:, :, :200, :200]
    occupancy_map_future = occupancy_map_future[:, :, :200, :200]
    occupancy_map_gt = occupancy_map_gt[:, :, :200, :200]
    occupancy_map_env = occupancy_map_env[:, :, :200, :200]
    
    occupancy_map = torch.cat((occupancy_map_env, occupancy_map_history, occupancy_map_future), dim=1)
    
    return occupancy_map, occupancy_map_gt

def OccupancyGeneratorParallelUnimodalforSimulation(input, y_hat, pi):
    occupancy_resolution = 0.5
    occupancy_size = 200
    occupancy_range = int(occupancy_size * occupancy_resolution)
    occupancy_map_env = torch.zeros(1, 1, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_history = torch.zeros(1, 11, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_future = torch.zeros(1, 16, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_gt = torch.zeros(1, 16, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    drop_edge_av = DistanceDropEdge(occupancy_range/2)
    input['edge_attr'] = \
            input['positions'][input['edge_index'][0], 10, :2] - input['positions'][input['edge_index'][1], 10, :2]
    edge_index, _ = drop_edge_av(input['edge_index'], input['edge_attr'])
    lane_edge_index, lane_attr_origin = drop_edge_av(input['lane_actor_index'], input['lane_actor_vectors'])
    
    av_mask = (edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    av_mask_index = torch.where(av_mask ==True)[0] #torch.Size([1419])
    others_indx = edge_index[0][av_mask_index]
    # lane_indx = lane_edge_index[0][av_mask_index]
    
    # batch_idx_for_agent = others_indx.unsqueeze(-1).repeat(1, 1) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_agent = torch.zeros(others_indx.shape[0], dtype=torch.int64) #torch.where((batch_idx_for_agent > 0), 1, 0).sum(dim=-1)
    
    # lane_mask = (lane_edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    lane_mask = (lane_edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    lane_mask_index = torch.where(lane_mask ==True)[0] #torch.Size([1419])
    lanes_indx = lane_edge_index[1][lane_mask_index]
    # lanes_indx2 = lane_edge_index[0][lane_mask_index]
    lane_attr = lane_attr_origin[lane_mask_index] + input.positions[lanes_indx, 10, :2]
    
    # batch_idx_for_lane = lane_edge_index[1].unsqueeze(-1).repeat(1, 1) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_lane = torch.zeros(lane_attr.shape[0], dtype=torch.int64) #torch.where((batch_idx_for_lane > 0), 1, 0).sum(dim=-1)
    
    occ_mask_lane = (abs(lane_attr[:, 0]) < 50) * (abs(lane_attr[:, 1]) < 50)
    lane = torch.where(((occ_mask_lane).unsqueeze(-1)),
                            lane_attr,
                            50*torch.ones(lane_attr.shape[0], 2).to(lane_attr.device))
    occupancy_x_lane, occupancy_y_lane = \
                (lane / occupancy_resolution).type(torch.int)[:, 0], \
                    (lane / occupancy_resolution).type(torch.int)[:, 1]
    occupancy_y_lane[~(occ_mask_lane)] = \
        -1 * occupancy_y_lane[~(occ_mask_lane)]
    occupancy_map_env[batch_idx_for_lane, \
        torch.arange(1).repeat(batch_idx_for_lane.shape[0]), \
            (occupancy_range - occupancy_y_lane).type(torch.long), \
                (occupancy_x_lane + occupancy_range).type(torch.long)] = 1
    
    agent_history = input.positions[others_indx][:, :11, :2] #(456, 11, 2)
    agent_future = input.positions[others_indx][:, 11:, :2] #(456, 16, 2)
    agent_pred = y_hat[:, others_indx][:, :, :, :2] + input.positions[others_indx][:, 10, :2].unsqueeze(0).unsqueeze(-2) #(1, 456, 16, 2)
    
    rotate_mat = torch.empty(input.num_nodes, 2, 2).to(y_hat.device)
    sin_vals = torch.sin(input['rotate_angles'])
    cos_vals = torch.cos(input['rotate_angles'])
    rotate_mat[:, 0, 0] = cos_vals
    rotate_mat[:, 0, 1] = sin_vals
    rotate_mat[:, 1, 0] = -sin_vals
    rotate_mat[:, 1, 1] = cos_vals
    
    y_hat_av_centric = torch.bmm(y_hat[..., :2].reshape(-1, 16, 2).to(torch.float32), rotate_mat.repeat(1, 1, 1)).reshape(1, -1, 16, 2)
            
    agent_pred = y_hat_av_centric[:, others_indx][:, :, :, :2] + input.positions[others_indx][:, 10, :2].unsqueeze(0).unsqueeze(-2) #(1, 456, 16, 2)


    occ_mask = (abs(agent_history[:, :, 0]) < 50) * (abs(agent_history[:, :, 1]) < 50)
    occ_mask_future = (abs(agent_future[:, :, 0]) < 50) * (abs(agent_future[:, :, 1]) < 50)
    occ_mask_pred = (abs(agent_pred[:, :, :, 0]) < 50) * (abs(agent_pred[:, :, :, 1]) < 50)
    
    agent_history = torch.where((((~input.padding_mask[others_indx, :11]) * occ_mask).unsqueeze(-1)),
                            agent_history,
                            50*torch.ones(others_indx.shape[0], 11, 2).to(agent_history.device))
    agent_future = torch.where((((~input.padding_mask[others_indx, 11:]) * occ_mask_future).unsqueeze(-1)),
                            agent_future,
                            50*torch.ones(others_indx.shape[0], 16, 2).to(agent_future.device))
    agent_pred = torch.where((((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred).unsqueeze(-1)),
                            agent_pred,
                            50*torch.ones(y_hat.shape[0], others_indx.shape[0], 16, 2).to(agent_pred.device))
    
    occupancy_x, occupancy_y = \
                (agent_history / occupancy_resolution).type(torch.int)[:, :, 0], \
                    (agent_history / occupancy_resolution).type(torch.int)[:, :, 1]
    occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)] = \
        -1 * occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)]
    occupancy_map_history[batch_idx_for_agent.unsqueeze(-1).repeat(1, 11).reshape(-1), \
        torch.arange(11).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y).type(torch.long).reshape(-1), \
                (occupancy_x + occupancy_range).type(torch.long).reshape(-1)] = 1
    
    occupancy_x_future, occupancy_y_future = \
                (agent_future / occupancy_resolution).type(torch.int)[:, :, 0], \
                    (agent_future / occupancy_resolution).type(torch.int)[:, :, 1]
    occupancy_y_future[~((~input.padding_mask[others_indx, 11:]) * occ_mask_future)] = \
        -1 * occupancy_y_future[~((~input.padding_mask[others_indx, 11:]) * occ_mask_future)]
    occupancy_map_gt[batch_idx_for_agent.unsqueeze(-1).repeat(1, 16).reshape(-1), \
        torch.arange(16).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y_future).type(torch.long).reshape(-1), \
                (occupancy_x_future + occupancy_range).type(torch.long).reshape(-1)] = 1
    
    occupancy_x_pred, occupancy_y_pred = \
                (agent_pred / occupancy_resolution).type(torch.int)[:, :, :, 0], \
                    (agent_pred / occupancy_resolution).type(torch.int)[:, :, :, 1]
    occupancy_y_pred[~((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred)] = \
        -1 * occupancy_y_pred[~((~input.padding_mask[others_indx, 11:]).unsqueeze(0) * occ_mask_pred)]
    occupancy_map_future[batch_idx_for_agent.unsqueeze(-1).repeat(1, 16).reshape(-1), \
        torch.arange(16).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y_pred).type(torch.long).reshape(-1), \
                (occupancy_x_pred + occupancy_range).type(torch.long).reshape(-1)] = 1
    
    occupancy_map_history = occupancy_map_history[:, :, :200, :200]
    occupancy_map_future = occupancy_map_future[:, :, :200, :200]
    occupancy_map_gt = occupancy_map_gt[:, :, :200, :200]
    occupancy_map_env = occupancy_map_env[:, :, :200, :200]
    
    occupancy_map = torch.cat((occupancy_map_env, occupancy_map_history, occupancy_map_future), dim=1)
    
    return occupancy_map, occupancy_map_gt

def apply_gaussian_kernel(occupancy_map, sigma=3):
    # 가우시안 커널 생성
    kernel_size = 5  # 커널 크기
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i, j] = np.exp(-((i - center) ** 2 + (j - center) ** 2) / (2 * sigma ** 2))
    kernel = torch.tensor(kernel, dtype=torch.float32)
    kernel = kernel.unsqueeze(0).unsqueeze(0).to(occupancy_map.device)  # 커널을 (1, 1, kernel_size, kernel_size)로 변환

    # 가우시안 필터 적용
    blurred_map = F.conv2d(occupancy_map, kernel, padding=2)

    return blurred_map

def GaussianOccupancyGT(occupancy_map_gt):
    # 가우시안 커널 적용
    # 각 채널에 가우시안 커널 적용
    blurred_maps = []
    for channel in range(occupancy_map_gt.size(1)):
        blurred_map = apply_gaussian_kernel(occupancy_map_gt[:, channel:channel+1, :, :], sigma=1)
        blurred_maps.append(blurred_map)

    blurred_occupancy_map = torch.cat(blurred_maps, dim=1)
    
    return blurred_occupancy_map