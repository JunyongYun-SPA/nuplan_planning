from enum import Enum
from typing import Dict, Tuple

import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncoding1D
from torch.distributions import Categorical


class TraversalReduction(Enum):
    NO_REDUCTION: Tuple[bool, bool] = False, False
    KEEP_ONLY_MOST_FREQUENT: Tuple[bool, bool] = True, True
    REPEAT_MOST_FREQUENT: Tuple[bool, bool] = True, False


class PGP(nn.Module):
    """
    Policy header + selective aggregator from "Multimodal trajectory prediction conditioned on lane graph traversals"
    1) Outputs edge probabilities corresponding to pi_route
    2) Samples pi_route to output traversed paths
    3) Selectively aggregates context along traversed paths
    """

    def __init__(
        self,
        pre_train: bool,
        node_enc_size: int,
        target_agent_enc_size: int,
        pi_h1_size: int,
        pi_h2_size: int,
        emb_size: int,
        num_heads: int,
        num_samples: int,
        horizon: int,
        use_route_mask: bool,
        hard_masking: bool,
        num_traversals: str,
        keep_only_best_traversal: bool = False,
    ):
        """
        'pre_train': bool, whether the model is being pre-trained using ground truth node sequence.
        'node_enc_size': int, size of node encoding
        'target_agent_enc_size': int, size of target agent encoding
        'pi_h1_size': int, size of first layer of policy header
        'pi_h2_size': int, size of second layer of policy header
        'emb_size': int, embedding size for attention layer for aggregating node encodings
        'num_heads: int, number of attention heads
        'num_samples': int, number of sampled traversals (and encodings) to output
        """
        super().__init__()
        self.pre_train = pre_train
        self.use_route_mask = use_route_mask
        self.hard_masking = hard_masking

        # define for interface checking
        self.agg_enc_size = emb_size + target_agent_enc_size
        self.target_agent_enc_size = target_agent_enc_size
        self.node_enc_size = node_enc_size

        # Policy header
        self.pi_h1 = nn.Linear(
            2 * node_enc_size + target_agent_enc_size + 2, pi_h1_size
        )
        self.pi_h2 = nn.Linear(pi_h1_size, pi_h2_size)
        self.pi_op = nn.Linear(pi_h2_size, 1)
        self.pi_h1_goal = nn.Linear(node_enc_size + target_agent_enc_size, pi_h1_size)
        self.pi_h2_goal = nn.Linear(pi_h1_size, pi_h2_size)
        self.pi_op_goal = nn.Linear(pi_h2_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.log_softmax = nn.LogSoftmax(dim=2)

        # For sampling policy
        self.horizon = horizon
        if not keep_only_best_traversal:
            assert (
                num_samples == num_traversals
            ), "if all traversals are used, number of sampled traversals has to match number of returned samples"
        self.num_samples = num_samples
        self.num_traversals = num_traversals
        self.keep_only_best_traversal = keep_only_best_traversal

        # Attention based aggregator
        self.pos_enc = PositionalEncoding1D(node_enc_size)
        self.query_emb = nn.Linear(target_agent_enc_size, emb_size)
        self.key_emb = nn.Linear(node_enc_size, emb_size)
        self.val_emb = nn.Linear(node_enc_size, emb_size)
        self.mha = nn.MultiheadAttention(emb_size, num_heads)

        if use_route_mask and not hard_masking:
            self.route_bonus = nn.Parameter(torch.ones(1))

    def forward(self, encodings: Dict) -> Dict:
        """
        Forward pass for PGP aggregator
        :param encodings: dictionary with encoder outputs
        :return: outputs: dictionary with
            'agg_encoding': aggregated encodings along sampled traversals
            'pi': discrete policy (probabilities over outgoing edges) for graph traversal
        """

        # Unpack encodings:
        target_agent_encoding = encodings["target_agent_encoding"] # (1, 32)
        node_encodings = encodings["context_encoding"]["combined"] # (1, 70, 32)
        node_masks = encodings["context_encoding"]["combined_masks"] # (1, 70)
        s_next = encodings["s_next"] # (1, 70, 16)
        edge_type = encodings["edge_type"] # (1, 70, 16)
        edge_on_route_mask = encodings["edge_on_route_mask"] # (1, 70, 16)
        node_on_route_mask = encodings["node_on_route_mask"] # (1, 70)

        # Compute pi (log probs)
        pi = self.compute_policy(
            target_agent_encoding,
            node_encodings,
            node_masks,
            s_next,
            edge_type,
            edge_on_route_mask,
        ) # (1, 70, 16)

        # If pretraining model, use ground truth node sequences
        if self.pre_train:
            sampled_traversals = (
                encodings["node_seq_gt"]
                .unsqueeze(1)
                .repeat(1, self.num_samples, 1)
                .long()
            )
        else:
            # Sample pi
            init_node = encodings["init_node"] # (1, 70)
            sampled_traversals = self.sample_policy(
                pi, s_next, init_node, node_on_route_mask
            )

        # Selectively aggregate context along traversed paths
        agg_enc = self.aggregate(
            sampled_traversals, node_encodings, target_agent_encoding
        ) # agg_enc.shape ~ (1, 1000, 160)

        outputs = {
            "agg_encoding": agg_enc,
            "pi": torch.log(pi + 1e-5),
            "sampled_traversals": sampled_traversals,
        }
        return outputs

    def aggregate(
        self, sampled_traversals, node_encodings, target_agent_encoding
    ) -> torch.Tensor:

        # Useful variables:
        batch_size = node_encodings.shape[0]
        max_nodes = node_encodings.shape[1]
        # num_traversals = sampled_traversals.shape[1]

        # Get unique traversals and form consolidated batch:
        counted_unique_traversals = [
            torch.unique(i, dim=0, return_counts=True) for i in sampled_traversals
        ] # sampled_traversals의 shape(1, 1000, 16) / counted_unique_traversals[0][0].shape -> [4, 16], counted_unique_traversals[0][1] -> [4, ]
        unique_traversals = [i[0] for i in counted_unique_traversals]
        traversal_counts = [i[1] for i in counted_unique_traversals]
        if self.keep_only_best_traversal: # False로 되어 있음
            # only a single traversal is used and passed to the decoder
            most_frequent_traversal_idcs = [
                torch.argmax(traversal_counts_sample)
                for traversal_counts_sample in traversal_counts
            ]
            best_traversals = [
                unique_traversals[i][most_frequent_traversal_idx].unsqueeze(0)
                for i, most_frequent_traversal_idx in enumerate(
                    most_frequent_traversal_idcs
                )
            ]
            traversal_counts = [
                torch.tensor([self.num_samples], device=sampled_traversals.device)
                for _ in traversal_counts
            ]
            unique_traversals = best_traversals

        traversals_batched = torch.cat(unique_traversals, dim=0)
        counts_batched = torch.cat(traversal_counts, dim=0)
        batch_idcs = torch.cat(
            [
                j * torch.ones(len(count)).long()
                for j, count in enumerate(traversal_counts)
            ]
        ) # (1, 4)
        batch_idcs = batch_idcs.unsqueeze(1).repeat(1, self.horizon) # 0으로 채워진 (4, 16)

        # Dummy encodings for goal nodes
        dummy_enc = torch.zeros_like(node_encodings)
        node_encodings = torch.cat((node_encodings, dummy_enc), dim=1) # (1, 140<70+70>, 32) # dim=1에서 70~140은 terminate값을 위한 것

        # Gather node encodings along traversed paths
        node_enc_selected = node_encodings[batch_idcs, traversals_batched] # travelsals_batched를 이용해 node_encoding에서 값을 가져온다. ~ (4, 16, 32)

        # Add positional encodings:
        pos_enc = self.pos_enc(torch.zeros_like(node_enc_selected)) # (4, 16, 32) positional encoding 진행 batch 단위 축으로는 같은 값이 생성됨 즉 pos_enc[0] == pos_enc[1]
        node_enc_selected += pos_enc # position 정보를 더하여 줌

        # Multi-head attention
        target_agent_enc_batched = target_agent_encoding[batch_idcs[:, 0]] # (1, 32)모양을 가지는 target_agent_encoding을 4번 복사한 것과 같은 효과 ~ (4, 32)
        query = self.query_emb(target_agent_enc_batched).unsqueeze(0) # (1, 4, 128)
        keys = self.key_emb(node_enc_selected).permute(1, 0, 2) # (4, 16, 64) -> (4, 16, 128) -> (16, 4, 128)
        vals = self.val_emb(node_enc_selected).permute(1, 0, 2) # (4, 16, 64) -> (4, 16, 128) -> (16, 4, 128)
        key_padding_mask = torch.as_tensor(traversals_batched >= max_nodes) # terminate node는 1로 아니면 0으로 만드는 마스크
        att_op, _ = self.mha(query, keys, vals, key_padding_mask) # multi head attn을 이용해 query 업데이트 ~ (1, 4, 128)

        # Repeat based on counts
        att_op = (
            att_op.squeeze(0)
            .repeat_interleave(counts_batched, dim=0)
            .view(batch_size, self.num_samples, -1)
        ) # count 정보를 이용하여 업데이트 된 query를 1000개의 개수를 가지도록 돌려놓은다. (1, 4, 128) -> (1, 1000, 128)

        # Concatenate target agent encoding
        agg_enc = torch.cat(
            (target_agent_encoding.unsqueeze(1).repeat(1, self.num_samples, 1), att_op),
            dim=-1,
        ) # concat(target_agent_encoding<1, 1000, 32>, att_op<1, 1000, 128>) ~ (1, 1000, 160)

        return agg_enc

    def sample_policy(
        self,
        pi: torch.Tensor,
        s_next: torch.Tensor,
        init_node: torch.Tensor,
        node_on_route_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample graph traversals using discrete policy.
        :param pi: tensor with probabilities corresponding to the policy
        :param s_next: look-up table for next node for a given source node and edge
        :param init_node: initial node to start the policy at
        :return:
        """
        with torch.no_grad():

            # Useful variables:
            batch_size = pi.shape[0]
            max_nodes = pi.shape[1]
            batch_idcs = (
                torch.arange(batch_size, device=pi.device)
                .unsqueeze(1)
                .repeat(1, self.num_traversals)
                .view(-1)
            )

            # Initialize output
            sampled_traversals = torch.zeros(
                batch_size, self.num_traversals, self.horizon, device=pi.device
            ).long()

            # Set up dummy self transitions for goal states:
            pi_dummy = torch.zeros_like(pi)
            pi_dummy[:, :, -1] = 1
            s_next_dummy = torch.zeros_like(s_next)
            s_next_dummy[:, :, -1] = max_nodes + torch.arange(max_nodes).unsqueeze(0).repeat(batch_size, 1) # s_next의 마지막 행 규칙을 따름
            pi = torch.cat((pi, pi_dummy), dim=1)
            s_next = torch.cat((s_next, s_next_dummy), dim=1)

            # Sample initial node:
            if self.use_route_mask and self.hard_masking:
                mask = 1.0 - ((init_node * node_on_route_mask).sum(dim=-1) > 0).float() # init_node(1, 70) * node_on_route_mask(1, 70) -> init node가 route위에 존재하는지 알아보기 위함
                node_on_route_mask = node_on_route_mask + mask.unsqueeze(-1)
                init_node = init_node * node_on_route_mask # (1, 70)
                init_node = init_node / init_node.sum(dim=-1, keepdim=True) # init node가 4개의 노드에 거쳐서 분포한다면, init_node의 값이 1/4의 값을 가지게 하기 위한 조치
            pi_s = (
                init_node.unsqueeze(1)
                .repeat(1, self.num_traversals, 1)
                .view(-1, max_nodes)
            ) # init_node를 traversal의 개수만큼 복사, (1, 70) -> (1, 1000, 70) -> (1000, 70)
            s = Categorical(pi_s).sample() # (1000, 70) -> (1000, )

            sampled_traversals[:, :, 0] = s.reshape(batch_size, self.num_traversals)

            # Sample traversed paths for a fixed horizon
            for n in range(1, self.horizon):

                # Gather policy at appropriate indices:
                pi_s = pi[batch_idcs, s]
                # pi_s[...,-1] = 0.0

                # Sample edges
                a = Categorical(pi_s).sample() # 이거 마음에 든다. 일종의 랜덤 탐색인셈이네, 여기에 몬테카를로 서치같은거 넣어줘도 좋을 듯

                # Look-up next node
                s = s_next[batch_idcs, s, a].long()

                # Add node indices to sampled traversals
                sampled_traversals[:, :, n] = s.reshape(batch_size, self.num_traversals)

        return sampled_traversals

    def compute_policy(
        self,
        target_agent_encoding: torch.Tensor,
        node_encodings: torch.Tensor,
        node_masks: torch.Tensor,
        s_next: torch.Tensor,
        edge_type: torch.Tensor,
        edge_on_route_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for policy header
        :param target_agent_encoding: tensor encoding the target agent's past motion
        :param node_encodings: tensor of node encodings provided by the encoder
        :param node_masks: masks indicating whether a node exists for a given index in the tensor
        :param s_next: look-up table for next node for a given source node and edge
        :param edge_type: look-up table with edge types
        :return pi: tensor with probabilities corresponding to the policy
        """
        device = target_agent_encoding.device

        # Useful variables:
        batch_size = node_encodings.shape[0] # 1
        max_nodes = node_encodings.shape[1] # 70
        max_nbrs = s_next.shape[2] - 1 # 15
        node_enc_size = node_encodings.shape[2] # 32
        target_agent_enc_size = target_agent_encoding.shape[1] # 32

        # Gather source node encodigns, destination node encodings, edge encodings and target agent encodings.
        src_node_enc = node_encodings.unsqueeze(2).repeat(1, 1, max_nbrs, 1) # (1, 70, 32) -> (1, 70, 15, 32)
        dst_idcs = s_next[:, :, :-1].reshape(batch_size, -1).long() # (1, 1050)
        batch_idcs = (
            torch.arange(batch_size).unsqueeze(1).repeat(1, max_nodes * max_nbrs)
        ) # (1, 1050)
        dst_node_enc = node_encodings[batch_idcs, dst_idcs].reshape(
            batch_size, max_nodes, max_nbrs, node_enc_size
        ) # node_encoding의 값을 s_next에 배치하기 위한 장치, -> (1, 70, 15, 32)
        target_agent_enc = (
            target_agent_encoding.unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, max_nodes, max_nbrs, 1)
        ) # target_agent_enc의 값을 (70, 15)개만큼 복사 (1, 70, 15, 32)
        edge_enc = torch.cat(
            (
                torch.as_tensor(edge_type[:, :, :-1] == 1, device=device)
                .unsqueeze(3)
                .float(),
                torch.as_tensor(edge_type[:, :, :-1] == 2, device=device)
                .unsqueeze(3)
                .float(),
            ),
            dim=3,
        ) # edge_enc_shape -> (1, 70, 15, 2) ~ 가장 마지막 dim을 기준으로 첫번째 차원의 (1,70, 15)는 edge가 successor edge인 경우, 2는 proximal edge인 경우를 나타낸다.
        enc = torch.cat((target_agent_enc, src_node_enc, dst_node_enc, edge_enc), dim=3)
        enc_goal = torch.cat(
            (target_agent_enc[:, :, 0, :], src_node_enc[:, :, 0, :]), dim=2
        ) # (1, 70, 64)

        # Form a single batch of encodings
        masks = torch.sum(edge_enc, dim=3, keepdim=True).bool() # (1, 70, 15, 2) -> (1, 70, 15, 1) 유효 노드를 찾는다.
        masks_goal = ~node_masks.unsqueeze(-1).bool() # (1, 70) -> (1, 70, 1)
        enc_batched = torch.masked_select(enc, masks).reshape(
            -1, target_agent_enc_size + 2 * node_enc_size + 2
        ) # (449, 98)
        enc_goal_batched = torch.masked_select(enc_goal, masks_goal).reshape(
            -1, target_agent_enc_size + node_enc_size
        ) #(70, 64) -> 70개의 노드 중 유효하지 않은 노드를 걸러내기 위함이다.

        # Compute scores for pi_route
        pi_ = self.pi_op(
            self.leaky_relu(self.pi_h2(self.leaky_relu(self.pi_h1(enc_batched))))
        ) # enc_batch에 MLP 연산을 진행하여 다음 노드(유효노드만본다)로의 score를 계산한다.
        pi = torch.zeros_like(masks, dtype=pi_.dtype)
        pi = pi.masked_scatter_(masks, pi_).squeeze(-1) # (449, 1) -> (1, 70, 15)
        pi_goal_ = self.pi_op_goal(
            self.leaky_relu(
                self.pi_h2_goal(self.leaky_relu(self.pi_h1_goal(enc_goal_batched)))
            )
        ) # (70, 64) -> (70, 1)
        pi_goal = torch.zeros_like(masks_goal, dtype=pi_goal_.dtype)
        pi_goal = pi_goal.masked_scatter_(masks_goal, pi_goal_) # (70, 1) -> (1, 70, 1)

        # In original implementation (https://github.com/nachiket92/PGP/blob/main/models/aggregators/pgp.py)
        #   op_masks = torch.log(torch.as_tensor(edge_type != 0).float())
        #   pi = self.log_softmax(pi + op_masks)
        # However, if edge_type == 0, op_masks = -inf which caused problems w/ nan gradients
        # Also, for goal-conditioning, further masks are applied.
        # Therefore, in this implementation probabilities are used instead of log-probabilities.
        # Note: before returning the probabilities (output name "pi") the probabilities are converted to log-probabilities
        #       to be able to apply the original log-likelihood loss

        pi = torch.cat((pi, pi_goal), dim=-1) # concat((1, 70, 15), (1, 70, 1)) -> (1, 70, 16)
        op_masks = torch.as_tensor(edge_type != 0).float() # (1, 70, 16) ~ terminate type이 포함되어 있으므로 edge_enc로 만들어지는 mask와는 다른 의미를 가진다.

        if self.use_route_mask:
            if self.hard_masking:
                op_masks = edge_on_route_mask * op_masks # op_masks에서 route위에 있지 않는 노드는 모두 제거하여 준다.
                pi = pi * edge_on_route_mask # pi는 다음 노드로 갈 score를 의미한다. 여기서 route위에 있지 않으면 그냥 score를 0로 만드는 것을 의미하낟.
            else:
                pi = pi + self.route_bonus * edge_on_route_mask # 학습시에는 route위에 있는 노드에 대해 보너스를 부여하고 이를 score에 더해주는 방식을 사용한다.

        return self.masked_softmax(pi, dim=2, mask=op_masks) # score를 masked softmax화해서 반환한다.

    def masked_softmax(
        self,
        input: torch.Tensor,
        dim: int,
        mask: torch.Tensor = None,
        tau: float = 1.0,
        eps: float = 1e-5,
    ):
        if mask is None:
            mask = torch.as_tensor(input != 0).float()
        input_exp = torch.exp(input / tau) * mask
        input_exp_sum = input_exp.sum(dim=-1, keepdim=True)
        input_exp[:, :, -1] += torch.as_tensor(input_exp_sum == 0).float().squeeze(-1)
        input_exp_sum = input_exp.sum(dim=dim, keepdim=True)
        return input_exp / input_exp_sum
