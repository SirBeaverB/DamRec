# -*- coding: utf-8 -*-
# GDN: Gated Delta Networks for Sequential Recommendation
#
# Modular architecture aligned with Gated Delta Networks (ICLR 2025):
# - Q/K RMSNorm, dual gating (β decay, γ input)
# - Multi-head associative memory, causal 1D DWConv
# - Output gate + SwiGLU FFN
# - Stackable GatedDeltaLayer blocks

import torch
from torch import nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import GatedDeltaLayer
from recbole.model.loss import BPRLoss


class GDN(SequentialRecommender):
    r"""GDN: Gated Delta Networks for Sequential Recommendation.

    Modular architecture:
    - Causal 1D DWConv -> Q/K RMSNorm -> Multi-Head Delta Rule (dual β, γ gating)
    - Output gate + SwiGLU FFN per layer
    - S_t = β_t S_{t-1} + γ_t (v_t - S_{t-1} k_t) ⊗ k_t^T
    """

    def __init__(self, config, dataset):
        super(GDN, self).__init__(config, dataset)

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
        self.use_fla = config["use_fla"] if config["use_fla"] is not None else True

        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        self.layers = nn.ModuleList([
            GatedDeltaLayer(
                d_model=self.embedding_size,
                num_heads=self.num_heads,
                conv_kernel_size=self.conv_kernel_size,
                ffn_ratio=self.ffn_ratio,
                dropout=self.dropout_prob,
                use_fla=self.use_fla,
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
                "[GDN] Streaming ON: multi-head delta, dual gating, causal conv"
            )
        else:
            self.logger.info("[GDN] Streaming OFF: batch-independent forward")
        fla_status = "ON" if self.use_fla else "OFF"
        self.logger.info(f"[GDN] use_fla={fla_status} (FLA 加速; scale=1.0 已修复)")

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

    def forward(self, item_seq, item_seq_len, prev_S_list=None):
        """Standard forward. prev_S_list: list of [B,H,d_h,d_h] per layer (optional)."""
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)
        B, L, _ = item_seq_emb.size()
        device = item_seq_emb.device
        valid_mask = self._valid_mask(item_seq_emb)

        x = item_seq_emb
        S_list = []
        for i, layer in enumerate(self.layers):
            S_init = prev_S_list[i] if prev_S_list is not None else None
            x, S = layer(x, S_init=S_init, valid_mask=valid_mask, return_S=True)
            S_list.append(S)

        last_idx = (item_seq_len - 1).clamp(min=0)
        out = self.gather_indexes(x, last_idx)
        return self.output_proj(out)

    def forward_with_streaming(self, item_seq, item_seq_len, user_ids, update_state=True):
        """Streaming: incremental update only, per-user state per layer.
        update_state=False: read-only for predict, avoids double-update in T2T."""
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
                start_idx.append(self._streaming_state[uid][1])
            else:
                start_idx.append(0)
        start_idx_tensor = torch.tensor(start_idx, device=device, dtype=torch.long).unsqueeze(1)
        inc_mask = seq_idx >= start_idx_tensor
        update_mask = valid_mask & inc_mask

        S_batch_list = []
        for layer_idx, layer in enumerate(self.layers):
            S_list = []
            for i in range(batch_size):
                uid = user_ids[i].item()
                if uid in self._streaming_state:
                    S_per_layer = self._streaming_state[uid][0]
                    S_list.append(S_per_layer[layer_idx])
                else:
                    d_h = self.embedding_size // self.num_heads
                    S_list.append(torch.zeros(
                        self.num_heads, d_h, d_h, device=device
                    ))
            S_batch = torch.stack(S_list, dim=0)
            S_batch_list.append(S_batch)

        x = item_seq_emb
        for layer_idx, layer in enumerate(self.layers):
            x, S_new = layer(
                x,
                S_init=S_batch_list[layer_idx],
                valid_mask=valid_mask,
                update_mask=update_mask,
                return_S=True,
            )
            S_batch_list[layer_idx] = S_new

        last_idx = (item_seq_len - 1).clamp(min=0)
        out = self.gather_indexes(x, last_idx)
        out = self.output_proj(out)

        if update_state:
            with torch.no_grad():
                for i in range(batch_size):
                    uid = user_ids[i].item()
                    new_len = item_seq_len[i].item()
                    if new_len > start_idx[i]:
                        stored = tuple(s[i].detach().clone() for s in S_batch_list)
                        self._streaming_state[uid] = (stored, new_len, device)

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
                item_seq, item_seq_len, user_ids, update_state=False
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
                item_seq, item_seq_len, user_ids, update_state=False
            )
        else:
            seq_output = self.forward(item_seq, item_seq_len)

        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )
        return scores
