# -*- coding: utf-8 -*-
# @Time    : 2025
# @Author  : DamRec

"""
LinRec
################################################

Reference:
    LinRec: L2-Normalized Linear Attention for Sequential Recommendation (SIGIR 2023)
    https://arxiv.org/abs/2305.03942

Architecture: Same as SASRec but replaces standard attention with LinRec linear attention.
Supports streaming with O(1) per-token state update.
"""

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import LinRecEncoder
from recbole.model.loss import BPRLoss


class LinRec(SequentialRecommender):
    r"""
    LinRec: L2-Normalized Linear Attention for Sequential Recommendation.

    Replaces standard self-attention in SASRec with linear-complexity LinRec attention:
    A'(Q,K,V) = ρ1(elu(Q)) @ (ρ2(elu(K))^T @ V)

    Streaming: maintains _streaming_state (KV per layer per user) for O(1) inference.
    Position: uses modulo for positions >= max_seq_length to avoid IndexError on long sequences.
    """

    def __init__(self, config, dataset):
        super(LinRec, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.streaming_mode = config.get("streaming_mode", False)

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.linrec_encoder = LinRecEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)
        self._streaming_state = {}

        if self.streaming_mode:
            self.logger.info("[LinRec] Streaming ON: O(1) per-token KV state")
        else:
            self.logger.info("[LinRec] Streaming OFF: batch forward")

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def reset_streaming_state(self):
        self._streaming_state.clear()

    def _get_position_ids(self, item_seq):
        """Position ids with modulo to avoid IndexError when seq exceeds max_seq_length."""
        seq_len = item_seq.size(1)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_ids = position_ids % self.max_seq_length
        return position_ids

    def forward(self, item_seq, item_seq_len):
        position_ids = self._get_position_ids(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        trm_output = self.linrec_encoder(
            input_emb, attention_mask=None, output_all_encoded_layers=True
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output

    def forward_with_streaming(self, item_seq, item_seq_len, user_ids, update_state=True):
        """Streaming: incremental KV update, per-user state per layer.
        update_state=False: read-only for predict, avoids double-update in T2T."""
        position_ids = self._get_position_ids(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        batch_size, seq_len = item_seq.size(0), item_seq.size(1)
        device = item_seq.device
        seq_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        start_idx = []
        for i in range(batch_size):
            uid = user_ids[i].item()
            if uid in self._streaming_state:
                start_idx.append(self._streaming_state[uid][1])
            else:
                start_idx.append(0)
        start_idx_tensor = torch.tensor(start_idx, device=device, dtype=torch.long).unsqueeze(1)
        valid_mask = (item_seq != 0)
        update_mask = valid_mask & (seq_idx >= start_idx_tensor)

        prev_KV_list = []
        for i in range(batch_size):
            uid = user_ids[i].item()
            if uid in self._streaming_state:
                prev_KV_list.append(self._streaming_state[uid][0])
            else:
                prev_KV_list.append(None)

        KV_batch_list = []
        for layer_idx in range(self.n_layers):
            layer_KV = []
            for i in range(batch_size):
                if prev_KV_list[i] is not None:
                    layer_KV.append(prev_KV_list[i][layer_idx])
                else:
                    d = self.hidden_size // self.n_heads
                    layer_KV.append(torch.zeros(
                        self.n_heads, d, d, device=device, dtype=input_emb.dtype
                    ))
            KV_batch_list.append(torch.stack(layer_KV, dim=0))

        trm_output, new_KV_list = self.linrec_encoder(
            input_emb,
            attention_mask=None,
            output_all_encoded_layers=True,
            prev_KV_list=KV_batch_list,
            update_mask=update_mask,
        )
        output = trm_output[-1]
        output = self.gather_indexes(output, item_seq_len - 1)

        if update_state:
            with torch.no_grad():
                for i in range(batch_size):
                    uid = user_ids[i].item()
                    new_len = item_seq_len[i].item()
                    if new_len > start_idx[i]:
                        stored = tuple(kv[i].detach().clone() for kv in new_KV_list)
                        self._streaming_state[uid] = (stored, new_len, device)

        return output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_ids = interaction[self.USER_ID]

        if self.streaming_mode:
            seq_output = self.forward_with_streaming(
                item_seq, item_seq_len, user_ids
            )
        else:
            seq_output = self.forward(item_seq, item_seq_len)

        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
            return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_ids = interaction[self.USER_ID]
        test_item = interaction[self.ITEM_ID]
        if self.streaming_mode:
            seq_output = self.forward_with_streaming(
                item_seq, item_seq_len, user_ids, update_state=False
            )
        else:
            seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_ids = interaction[self.USER_ID]
        if self.streaming_mode:
            seq_output = self.forward_with_streaming(
                item_seq, item_seq_len, user_ids, update_state=False
            )
        else:
            seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
