from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from typing import List, Optional, Tuple
from torch_geometric.data import Data
from nuplan.planning.script.builders.utils.utils_type import validate_type
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)


@dataclass
class HiVTFeature(AbstractModelFeature): #JY
    # ego_position: FeatureDataType
    # ego_velocity: FeatureDataType
    # ego_acceleration: FeatureDataType
    # neighbor_position: FeatureDataType #JY
    # planner_centerline: FeatureDataType
    # planner_trajectory: FeatureDataType
    x: FeatureDataType
    positions: FeatureDataType
    edge_index: FeatureDataType
    num_nodes: FeatureDataType
    padding_mask: FeatureDataType
    bos_mask: FeatureDataType
    rotate_angles: FeatureDataType
    seq_id: FeatureDataType
    av_index: FeatureDataType
    agent_index: FeatureDataType
    city: FeatureDataType
    origin: FeatureDataType
    theta: FeatureDataType
    ptr: FeatureDataType

    def to_feature_tensor(self) -> HiVTFeature: #JY
        """
        :return object which will be collated into a batch
        """
        # return HiVTFeature( #JY
        #     ego_position=to_tensor(self.ego_position),
        #     ego_velocity=to_tensor(self.ego_velocity),
        #     ego_acceleration=to_tensor(self.ego_acceleration),
        #     neighbor_position=to_tensor(self.neighbor_position), #JY
        #     planner_centerline=to_tensor(self.planner_centerline),
        #     planner_trajectory=to_tensor(self.planner_trajectory),
        # )
        return HiVTFeature( #JY
            x=to_tensor(self.x),
            positions=to_tensor(self.positions),
            edge_index=to_tensor(self.edge_index),
            num_nodes=self.num_nodes, #JY
            padding_mask=to_tensor(self.padding_mask),
            bos_mask=to_tensor(self.bos_mask),
            rotate_angles=to_tensor(self.rotate_angles),
            seq_id=self.seq_id,
            av_index=self.av_index,
            agent_index=self.agent_index,
            city=self.city,
            origin=to_tensor(self.origin),
            theta=to_tensor(self.theta),
        )

    def to_device(self, device: torch.device) -> HiVTFeature: #JY
        """Implemented. See interface."""
        validate_type(self.x, torch.Tensor)
        validate_type(self.positions, torch.Tensor)
        validate_type(self.edge_index, torch.Tensor)
        
        validate_type(self.num_nodes, torch.Tensor) #JY

        validate_type(self.padding_mask, torch.Tensor)
        validate_type(self.bos_mask, torch.Tensor)
        validate_type(self.rotate_angles, torch.Tensor)
        validate_type(self.seq_id, torch.Tensor)
        validate_type(self.av_index, torch.Tensor)
        validate_type(self.agent_index, torch.Tensor)
        validate_type(self.city, torch.Tensor)
        validate_type(self.origin, torch.Tensor)
        validate_type(self.theta, torch.Tensor)
 
        return HiVTFeature( #JY
            x=self.x.to(device=device),
            positions=self.positions.to(device=device),
            edge_index=self.edge_index.to(device=device),
            num_nodes=self.num_nodes.to(device=device), #JY
            padding_mask=self.padding_mask.to(device=device),
            bos_mask=self.bos_mask.to(device=device),
            rotate_angles=self.rotate_angles.to(device=device),
            seq_id=self.seq_id.to(device=device),
            av_index=self.av_index.to(device=device),
            agent_index=self.agent_index.to(device=device),
            city=self.city.to(device=device),
            origin=self.origin.to(device=device),
            theta=self.theta.to(device=device),
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> HiVTFeature: #JY
        """
        :return: Return dictionary of data that can be serialized
        """
        return HiVTFeature( #JY
            x=data["x"],
            positions=data["positions"],
            edge_index=data["edge_index"],
            num_nodes=data["num_nodes"],
            padding_mask=data["padding_mask"],
            bos_mask=data["bos_mask"],
            rotate_angles=data["rotate_angles"],
            seq_id=data["seq_id"],
            av_index=data["av_index"],
            agent_index=data["agent_index"],
            city=data["city"],
            origin=data["origin"],
            theta=data["theta"],
        )

    def unpack(self) -> List[HiVTFeature]: #JY
        """
        :return: Unpack a batched feature to a list of features.
        """
        return [
            HiVTFeature( #JY
                x[None],
                positions[None],
                edge_index[None],
                num_nodes[None], #JY
                padding_mask[None],
                bos_mask[None],
                rotate_angles[None],
                seq_id[None],
                av_index[None],
                agent_index[None],
                city[None],
                origin[None],
                theta[None],
            )
            for x, positions, edge_index, num_nodes, padding_mask, bos_mask, rotate_angles, seq_id, av_index, agent_index, city, origin, theta in zip( #JY
                self.x,
                self.positions,
                self.edge_index,
                self.num_nodes, #JY
                self.padding_mask,
                self.bos_mask,
                self.rotate_angles,
                self.seq_id,
                self.av_index,
                self.agent_index,
                self.city,
                self.origin,
                self.theta,
            )
        ]

    @property
    def batch_size(self) -> int:
        """
        :return: number of batches
        """
        if len(self.x.shape) == 3: #2 #JY
            return self.x.shape[0]
        else:
            return None

    @classmethod
    def collate(cls, batch: List[HiVTFeature]) -> HiVTFeature: #JY
        """
        Implemented. See interface.
        Collates a list of features that each have batch size of 1.
        """
        device = batch[0].x.device

        collated_x = torch.stack(
            [item.x for item in batch], dim=0
        ).to(device)

        collated_positions = torch.stack(
            [item.positions for item in batch], dim=0
        ).to(device)

        collated_edge_index = torch.stack(
            [item.edge_index for item in batch], dim=0
        ).to(device)
        
        collated_num_nodes = torch.stack(
            [item.num_nodes for item in batch], dim=0
        ).to(device) #JY

        collated_padding_mask = torch.stack(
            [item.padding_mask for item in batch], dim=0
        ).to(device)
        collated_bos_mask = torch.stack(
            [item.bos_mask for item in batch], dim=0
        ).to(device)
        collated_rotate_angles = torch.stack(
            [item.rotate_angles for item in batch], dim=0
        ).to(device)
        collated_seq_id = torch.stack(
            [item.seq_id for item in batch], dim=0
        ).to(device)
        collated_av_index = torch.stack(
            [item.av_index for item in batch], dim=0
        ).to(device)
        collated_agent_index = torch.stack(
            [item.agent_index for item in batch], dim=0
        ).to(device)
        collated_city = torch.stack(
            [item.city for item in batch], dim=0
        ).to(device)
        collated_origin = torch.stack(
            [item.origin for item in batch], dim=0
        ).to(device)
        collated_theta = torch.stack(
            [item.theta for item in batch], dim=0
        ).to(device)

        return HiVTFeature( #JY
            x=collated_x,
            positions=collated_positions,
            edge_index=collated_edge_index,
            num_nodes=collated_num_nodes, #JY
            collated_padding_mask=collated_padding_mask,
            bos_mask=collated_bos_mask,
            rotate_angles=collated_rotate_angles,
            seq_id=collated_seq_id,
            av_index=collated_av_index,
            agent_index=collated_agent_index,
            city=collated_city,
            origin=collated_origin,
            theta=collated_theta,
        )

class TemporalData(Data):

    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 lane_vectors: Optional[torch.Tensor] = None,
                 is_intersections: Optional[torch.Tensor] = None,
                 turn_directions: Optional[torch.Tensor] = None,
                 traffic_controls: Optional[torch.Tensor] = None,
                 lane_actor_index: Optional[torch.Tensor] = None,
                 lane_actor_vectors: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 av_index = None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(x=x, positions=positions, edge_index=edge_index, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           lane_vectors=lane_vectors, is_intersections=is_intersections,
                                           turn_directions=turn_directions, traffic_controls=traffic_controls,
                                           lane_actor_index=lane_actor_index, lane_actor_vectors=lane_actor_vectors,
                                           seq_id=seq_id, av_index=av_index, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[t]

    def __inc__(self, key, value):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value)
