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