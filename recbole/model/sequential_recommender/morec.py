# -*- coding: utf-8 -*-
# MoRec: Momentum-based Delta Rule for Streaming Recommendation
#
# Replaces SGD-equivalent update in GDN with momentum SGD:
#   GDN (SGD):  S_t = β S_{t-1} + γ (v - S k) ⊗ k^T
#   MoRec:      m_t = μ m_{t-1} + γ (v - S k) ⊗ k^T;  S_t = S_{t-1} + m_t

import torch
from torch import nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import GatedDeltaLayerMomentum
from recbole.model.loss import BPRLoss


class MoRec(SequentialRecommender):
    r"""MoRec: Momentum-based Delta Rule for Streaming Recommendation.

    Same architecture as GDN but uses momentum SGD in the associative memory update:
    - m_t = μ m_{t-1} + γ (v - S k) ⊗ k^T
    - S_t = S_{t-1} + m_t

    Streaming state per user: (S_list, M_list, length, device).
    """

    def __init__(self, config, dataset):
        super(MoRec, self).__init__(config, dataset)

        self.embedding_size = config["embedding_size"]
        self.loss_type = config["loss_type"]
        self.dropout_prob = config["dropout_prob"]
        self.streaming_mode = (
            config["streaming_mode"] if config["streaming_mode"] is not None else False
        )
        self.n_layers = config["n_layers"] if config["n_layers"] is not None else 1
        self.num_heads = config["num_heads"] if config["num_heads"] is not None else 4
        self.conv_kernel_size = config["conv_kernel_size"] if config["conv_kernel_size"] is not None else 3
        self.ffn_ratio = config["ffn_ratio"] if config["ffn_ratio"] is not None else 4
        self.momentum = config["momentum"] if config["momentum"] is not None else 0.9

        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        self.layers = nn.ModuleList([
            GatedDeltaLayerMomentum(
                d_model=self.embedding_size,
                num_heads=self.num_heads,
                conv_kernel_size=self.conv_kernel_size,
                ffn_ratio=self.ffn_ratio,
                dropout=self.dropout_prob,
                momentum=self.momentum,
            )
            for _ in range(self.n_layers)
        ])
        self.output_proj = nn.Linear(self.embedding_size, self.embedding_size)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("loss_type must be in ['BPR', 'CE']")

        self.apply(self._init_weights)
        self._streaming_state = {}

        if self.streaming_mode:
            self.logger.info(
                "[MoRec] Streaming ON: momentum delta (μ=%.2f)" % self.momentum
            )
        else:
            self.logger.info("[MoRec] Streaming OFF: batch-independent forward")

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reset_streaming_state(self):
        self._streaming_state.clear()

    def _valid_mask(self, item_seq_emb):
        return item_seq_emb.abs().sum(dim=-1) > 1e-8

    def forward(self, item_seq, item_seq_len, prev_S_list=None, prev_M_list=None):
        """Standard forward. prev_S_list, prev_M_list: per-layer init (optional)."""
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)
        B, L, _ = item_seq_emb.size()
        device = item_seq_emb.device
        valid_mask = self._valid_mask(item_seq_emb)

        x = item_seq_emb
        S_list, M_list = [], []
        for i, layer in enumerate(self.layers):
            S_init = prev_S_list[i] if prev_S_list is not None else None
            M_init = prev_M_list[i] if prev_M_list is not None else None
            x, S, M = layer(
                x,
                S_init=S_init,
                M_init=M_init,
                valid_mask=valid_mask,
                return_S=True,
            )
            S_list.append(S)
            M_list.append(M)

        last_idx = (item_seq_len - 1).clamp(min=0)
        out = self.gather_indexes(x, last_idx)
        return self.output_proj(out)

    def forward_with_streaming(self, item_seq, item_seq_len, user_ids):
        """Streaming: per-user (S, M) state per layer."""
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)

        batch_size, seq_len, _ = item_seq_emb.size()
        device = item_seq_emb.device
        valid_mask = self._valid_mask(item_seq_emb)

        seq_idx = torch.arange(seq_len, device=device).unsqueeze(0)
        start_idx = []
        for i in range(batch_size):
            uid = user_ids[i].item()
            if uid in self._streaming_state:
                start_idx.append(self._streaming_state[uid][2])
            else:
                start_idx.append(0)
        start_idx_tensor = torch.tensor(start_idx, device=device, dtype=torch.long).unsqueeze(1)
        inc_mask = seq_idx >= start_idx_tensor
        update_mask = valid_mask & inc_mask

        S_batch_list, M_batch_list = [], []
        for layer_idx, layer in enumerate(self.layers):
            S_list, M_list = [], []
            for i in range(batch_size):
                uid = user_ids[i].item()
                if uid in self._streaming_state:
                    S_per_layer = self._streaming_state[uid][0]
                    M_per_layer = self._streaming_state[uid][1]
                    S_list.append(S_per_layer[layer_idx])
                    M_list.append(M_per_layer[layer_idx])
                else:
                    d_h = self.embedding_size // self.num_heads
                    S_list.append(torch.zeros(self.num_heads, d_h, d_h, device=device))
                    M_list.append(torch.zeros(self.num_heads, d_h, d_h, device=device))
            S_batch_list.append(torch.stack(S_list, dim=0))
            M_batch_list.append(torch.stack(M_list, dim=0))

        x = item_seq_emb
        for layer_idx, layer in enumerate(self.layers):
            x, S_new, M_new = layer(
                x,
                S_init=S_batch_list[layer_idx],
                M_init=M_batch_list[layer_idx],
                valid_mask=valid_mask,
                update_mask=update_mask,
                return_S=True,
            )
            S_batch_list[layer_idx] = S_new
            M_batch_list[layer_idx] = M_new

        last_idx = (item_seq_len - 1).clamp(min=0)
        out = self.gather_indexes(x, last_idx)
        out = self.output_proj(out)

        with torch.no_grad():
            for i in range(batch_size):
                uid = user_ids[i].item()
                new_len = item_seq_len[i].item()
                if new_len > start_idx[i]:
                    stored_S = tuple(s[i].detach().clone() for s in S_batch_list)
                    stored_M = tuple(m[i].detach().clone() for m in M_batch_list)
                    self._streaming_state[uid] = (stored_S, stored_M, new_len, device)

        return out

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_ids = interaction[self.USER_ID]

        if hasattr(self, "streaming_mode") and self.streaming_mode:
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
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)

        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_ids = interaction[self.USER_ID]
        if hasattr(self, "streaming_mode") and self.streaming_mode:
            seq_output = self.forward_with_streaming(
                item_seq, item_seq_len, user_ids
            )
        else:
            seq_output = self.forward(item_seq, item_seq_len)
        test_item = interaction[self.ITEM_ID]
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        user_ids = interaction[self.USER_ID]

        if hasattr(self, "streaming_mode") and self.streaming_mode:
            seq_output = self.forward_with_streaming(
                item_seq, item_seq_len, user_ids
            )
        else:
            seq_output = self.forward(item_seq, item_seq_len)

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )
        return scores
