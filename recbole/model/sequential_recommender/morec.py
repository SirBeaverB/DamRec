# -*- coding: utf-8 -*-
# MoRec: Momentum-based Delta Rule for Streaming Recommendation
#
# 两种模式:
# - Token 级: m_t = μ m_{t-1} + γ (v - S k) ⊗ k^T; S_t = S_{t-1} + m_t (Python 循环)
# - Chunk 级: FLA 内部一阶 + Chunk 边界宏观动量，可复用 FLA 加速

import torch
from torch import nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import (
    GatedDeltaLayerMomentum,
    GatedDeltaLayerChunkMomentum,
)
from recbole.model.loss import BPRLoss

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False

DEBUG_T2T_STREAMING = False  # 设为 True 可开启算子/动量调试输出


class MoRec(SequentialRecommender):
    r"""MoRec: Momentum-based Delta Rule for Streaming Recommendation.

    - Token 级动量: m_t = μ m_{t-1} + γ (v - S k) ⊗ k^T; S_t = S_{t-1} + m_t
    - Chunk 级动量 (use_chunk_momentum): FLA + 宏观 M_new = μ*M + (1-μ)*ΔS, S_next = S_end + η*M
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
        self.momentum_eta = config["momentum_eta"] if config["momentum_eta"] is not None else 0.1
        use_chunk = config["use_chunk_momentum"] if config["use_chunk_momentum"] is not None else False
        self.use_chunk_momentum = use_chunk and _FLA_AVAILABLE

        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        layer_cls = GatedDeltaLayerChunkMomentum if self.use_chunk_momentum else GatedDeltaLayerMomentum
        layer_kw = dict(
            d_model=self.embedding_size,
            num_heads=self.num_heads,
            conv_kernel_size=self.conv_kernel_size,
            ffn_ratio=self.ffn_ratio,
            dropout=self.dropout_prob,
        )
        if self.use_chunk_momentum:
            layer_kw["momentum"] = self.momentum
            layer_kw["momentum_eta"] = self.momentum_eta
        else:
            layer_kw["momentum"] = self.momentum

        self.layers = nn.ModuleList([layer_cls(**layer_kw) for _ in range(self.n_layers)])
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
            mode = "chunk-level (FLA)" if self.use_chunk_momentum else "token-level"
            self.logger.info(
                "[MoRec] Streaming ON: %s momentum (μ=%.2f)" % (mode, self.momentum)
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

    def forward_with_streaming(self, item_seq, item_seq_len, user_ids, update_state=True):
        """Streaming: per-user (S, M) state per layer.
        update_state=False: read-only for predict, avoids double-update in T2T."""
        if DEBUG_T2T_STREAMING and not getattr(MoRec, "_t2t_debug_layer_printed", False):
            print(f"\n[DEBUG] MoRec Layer Type: {type(self.layers[0]).__name__}, FLA_AVAILABLE: {_FLA_AVAILABLE}, use_chunk_momentum: {self.use_chunk_momentum}\n")
            MoRec._t2t_debug_layer_printed = True

        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)

        batch_size, seq_len, _ = item_seq_emb.size()
        device = item_seq_emb.device
        valid_mask = self._valid_mask(item_seq_emb)

        update_mask = torch.zeros_like(item_seq_emb[:, :, 0], dtype=torch.bool)
        for i in range(batch_size):
            valid_len = item_seq_len[i].item()
            if valid_len == 0:
                continue
            if user_ids[i].item() in self._streaming_state:
                update_mask[i, valid_len - 1] = True
            else:
                update_mask[i, :valid_len] = True
        update_mask = update_mask & valid_mask

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

        if update_state:
            with torch.no_grad():
                for i in range(batch_size):
                    uid = user_ids[i].item()
                    new_len = item_seq_len[i].item()
                    stored_S = tuple(s[i].detach().clone() for s in S_batch_list)
                    stored_M = tuple(m[i].detach().clone() for m in M_batch_list)
                    if DEBUG_T2T_STREAMING:
                        cnt = getattr(MoRec, "_t2t_debug_m_count", 0)
                        if cnt < 50:
                            m_mean = sum(m.abs().mean().item() for m in stored_M) / len(stored_M)
                            print(f"[DEBUG] MoRec uid={uid} Step {new_len}, M_mean: {m_mean:.6f}")
                            MoRec._t2t_debug_m_count = cnt + 1
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
