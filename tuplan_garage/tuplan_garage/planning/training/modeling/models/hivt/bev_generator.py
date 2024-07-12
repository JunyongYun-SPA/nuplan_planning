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
from itertools import permutations

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

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
    # input['edge_attr_static'] = \
    #         input['static_positions'][input['edge_index_static'][0], 10, :2] - input['positions'][input['edge_index_static'][1], 10, :2]
    edge_index, _ = drop_edge_av(input['edge_index'], input['edge_attr'])
    # edge_index_static, _ = drop_edge_av(input['edge_index_static'], input['edge_attr_static'])
    lane_edge_index, lane_attr_origin = drop_edge_av(input['lane_actor_index'], input['lane_actor_vectors'])
    
    av_mask = (edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    av_mask_index = torch.where(av_mask ==True)[0] #torch.Size([1419])
    others_indx = edge_index[0][av_mask_index]
    # static_indx = input.edge_index_static[0]
    others_indx_static = input.edge_index_static[1]
    
    batch_idx_for_agent = others_indx.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_agent = torch.where((batch_idx_for_agent >= 0), 1, 0).sum(dim=-1)
    
    batch_idx_for_static = others_indx_static.unsqueeze(-1).repeat(1, batch_shape) - input.ptr.unsqueeze(0)[:, 1:]
    batch_idx_for_static = torch.where((batch_idx_for_static >= 0), 1, 0).sum(dim=-1)
    static_history = input.static_positions[:, 10, :2] #(456, 11, 2)
    occ_mask_static = (abs(static_history[:, 0]) < 50) * (abs(static_history[:, 1]) < 50)
    static = torch.where((((~input.padding_mask_static[:, 10]) * occ_mask_static).unsqueeze(-1)),
                            static_history,
                            50*torch.ones(static_history.shape[0], 2).to(static_history.device))
    occupancy_x_static, occupancy_y_static = \
                (static / occupancy_resolution).type(torch.int)[:, 0], \
                    (static / occupancy_resolution).type(torch.int)[:, 1]
    occupancy_y_static[~((~input.padding_mask_static[:, 10]) * occ_mask_static)] = \
        -1 * occupancy_y_static[~((~input.padding_mask_static[:, 10]) * occ_mask_static)]
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

    # a = batch_idx_for_agent.unsqueeze(0).unsqueeze(-1).repeat(occupancy_x_pred.shape[0], 1, 16).permute(2, 0, 1).reshape(-1)
    # b = torch.arange(16).unsqueeze(0).unsqueeze(1).repeat(occupancy_x_pred.shape[0], occupancy_x_pred.shape[1], 1).permute(2, 0, 1).reshape(-1).to(a.device)
    # c = (occupancy_range - occupancy_y_pred).type(torch.long).permute(2, 0, 1).reshape(-1)
    # d = (occupancy_x_pred + occupancy_range).type(torch.long).permute(2, 0, 1).reshape(-1)
    
    # temp_list = torch.stack((a, b.to(a.device), c, d, agent_pred_pi), dim=-1).tolist()
    # temp_list.sort(key=lambda x: (x[0], x[1], x[2], x[3], -x[4]))

    # diff = torch.tensor(temp_list)[0:-1, :4] - torch.tensor(temp_list)[1:, :4]
    # duplicated_idx = torch.where(torch.any(diff, dim=-1) == True)[0]
    # duplicated_idx = torch.cat((duplicated_idx, (duplicated_idx[-1]+1).unsqueeze(0)))
    # occ_idx, occ_value = torch.tensor(temp_list)[duplicated_idx][:, :4].to(torch.int64), torch.tensor(temp_list)[duplicated_idx][:, 4]
    # o_a, o_b, o_c, o_d = occ_idx[:, 0], occ_idx[:, 1], occ_idx[:, 2], occ_idx[:, 3]
    # occupancy_map_future[o_a, o_b, o_c, o_d] = occ_value.to(occupancy_map_future.device)

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

def OccupancyGeneratorParallelforSimulation(input):
    batch_shape = 1
    
    occupancy_resolution = 0.5
    occupancy_size = 200
    occupancy_range = int(occupancy_size * occupancy_resolution)
    occupancy_map_env = torch.zeros(batch_shape, 2, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    occupancy_map_history = torch.zeros(batch_shape, 11, occupancy_size+1, occupancy_size+1, dtype=torch.float32).to(pi.device)
    drop_edge_av = DistanceDropEdge(occupancy_range/2)
    input['edge_attr'] = \
            input['positions'][input['edge_index'][0], 10, :2] - input['positions'][input['edge_index'][1], 10, :2]
    edge_index, _ = drop_edge_av(input['edge_index'], input['edge_attr'])
    lane_edge_index, lane_attr_origin = drop_edge_av(input['lane_actor_index'], input['lane_actor_vectors'])
    
    av_mask = (edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    av_mask_index = torch.where(av_mask ==True)[0] #torch.Size([1419])
    others_indx = edge_index[0][av_mask_index]
    
    batch_idx_for_agent = torch.zeros(others_indx.shape[0], dtype=torch.int64) #torch.where((batch_idx_for_agent > 0), 1, 0).sum(dim=-1)
     
    lane_mask = (lane_edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    lane_mask_index = torch.where(lane_mask ==True)[0] #torch.Size([1419])
    lanes_indx = lane_edge_index[1][lane_mask_index]
    lane_attr = lane_attr_origin[lane_mask_index] + input.positions[lanes_indx, 10, :2]

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
    
    occ_mask = (abs(agent_history[:, :, 0]) < 50) * (abs(agent_history[:, :, 1]) < 50)
    
    agent_history = torch.where((((~input.padding_mask[others_indx, :11]) * occ_mask).unsqueeze(-1)),
                            agent_history,
                            50*torch.ones(others_indx.shape[0], 11, 2).to(agent_history.device))
    
    occupancy_x, occupancy_y = \
                (agent_history / occupancy_resolution).type(torch.int)[:, :, 0], \
                    (agent_history / occupancy_resolution).type(torch.int)[:, :, 1]
    occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)] = \
        -1 * occupancy_y[~((~input.padding_mask[others_indx, :11]) * occ_mask)]
    occupancy_map_history[batch_idx_for_agent.unsqueeze(-1).repeat(1, 11).reshape(-1), \
        torch.arange(11).unsqueeze(0).repeat(batch_idx_for_agent.shape[0], 1).reshape(-1), \
            (occupancy_range - occupancy_y).type(torch.long).reshape(-1), \
                (occupancy_x + occupancy_range).type(torch.long).reshape(-1)] = 1

    occupancy_map_history = occupancy_map_history[:, :, :200, :200]
    occupancy_map_env = occupancy_map_env[:, :, :200, :200]
    
    return occupancy_map_env, occupancy_map_history

def OccupancyGeneratorParallelUnimodalforSimulation(input, y_hat, pi):
    occupancy_resolution = 1.0
    occupancy_size = 100
    occupancy_range = int(occupancy_size * 0.5)
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

    batch_idx_for_agent = torch.zeros(others_indx.shape[0], dtype=torch.int64) #torch.where((batch_idx_for_agent > 0), 1, 0).sum(dim=-1)

    lane_mask = (lane_edge_index[1].unsqueeze(1)==input.av_index).any(dim=1) 
    lane_mask_index = torch.where(lane_mask ==True)[0]
    lanes_indx = lane_edge_index[1][lane_mask_index]
    lane_attr = lane_attr_origin[lane_mask_index] + input.positions[lanes_indx, 10, :2]
    
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