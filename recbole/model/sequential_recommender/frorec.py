# -*- coding: utf-8 -*-
# FroRec: Frobenius Adam for Delta Rule in Streaming Recommendation
#
# F-Adam: 二阶矩 V 降维为标量，基于 Frobenius 范数的全局波动率，不破坏 M 的秩一外积方向。
# 标量除法，几何结构 100% 安全，自适应抗噪。

import torch
from torch import nn
from torch.nn.init import xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import GatedDeltaLayerChunkFroAdam
from recbole.model.loss import BPRLoss

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False

DEBUG_T2T_STREAMING = False  # 设为 True 可开启算子/动量调试输出


class FroRec(SequentialRecommender):
    r"""FroRec: Frobenius Adam for Delta Rule in Streaming Recommendation.

    F-Adam: V 为标量 [B,H,1,1]，S_next = S_end + η*M/(√V+ε)，保留 M 的秩一外积几何。
    需要 FLA + CUDA。
    """

    def __init__(self, config, dataset):
        super(FroRec, self).__init__(config, dataset)

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
        self.adam_beta1 = config["adam_beta1"] if config["adam_beta1"] is not None else 0.9
        self.adam_beta2 = config["adam_beta2"] if config["adam_beta2"] is not None else 0.999
        self.adam_eps = config["adam_eps"] if config["adam_eps"] is not None else 1e-8
        self.adam_eta = config["adam_eta"] if config["adam_eta"] is not None else 0.1

        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)

        self.layers = nn.ModuleList([
            GatedDeltaLayerChunkFroAdam(
                d_model=self.embedding_size,
                num_heads=self.num_heads,
                conv_kernel_size=self.conv_kernel_size,
                ffn_ratio=self.ffn_ratio,
                dropout=self.dropout_prob,
                adam_beta1=self.adam_beta1,
                adam_beta2=self.adam_beta2,
                adam_eps=self.adam_eps,
                adam_eta=self.adam_eta,
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

        self.logger.info(
            "[FroRec] F-Adam (β1=%.2f, β2=%.3f, η=%.2f), requires FLA+CUDA"
            % (self.adam_beta1, self.adam_beta2, self.adam_eta)
        )

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

    def forward(self, item_seq, item_seq_len, prev_S_list=None, prev_M_list=None, prev_V_list=None):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb = self.emb_dropout(item_seq_emb)
        B, L, _ = item_seq_emb.size()
        device = item_seq_emb.device
        valid_mask = self._valid_mask(item_seq_emb)

        x = item_seq_emb
        S_list, M_list, V_list = [], [], []
        for i, layer in enumerate(self.layers):
            S_init = prev_S_list[i] if prev_S_list is not None else None
            M_init = prev_M_list[i] if prev_M_list is not None else None
            V_init = prev_V_list[i] if prev_V_list is not None else None
            x, S, M, V, _ = layer(
                x,
                S_init=S_init,
                M_init=M_init,
                V_init=V_init,
                valid_mask=valid_mask,
                return_S=True,
            )
            S_list.append(S)
            M_list.append(M)
            V_list.append(V)

        last_idx = (item_seq_len - 1).clamp(min=0)
        out = self.gather_indexes(x, last_idx)
        return self.output_proj(out)

    def forward_with_streaming(self, item_seq, item_seq_len, user_ids, update_state=True):
        """update_state=False: read-only for predict, avoids double-update in T2T."""
        if DEBUG_T2T_STREAMING and not getattr(FroRec, "_t2t_debug_layer_printed", False):
            print(f"\n[DEBUG] FroRec Layer Type: {type(self.layers[0]).__name__}, FLA_AVAILABLE: {_FLA_AVAILABLE}\n")
            FroRec._t2t_debug_layer_printed = True

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

        S_batch_list, M_batch_list, V_batch_list = [], [], []
        step_batch_list = []
        for layer_idx, layer in enumerate(self.layers):
            S_list, M_list, V_list = [], [], []
            step_list = []
            for i in range(batch_size):
                uid = user_ids[i].item()
                if uid in self._streaming_state:
                    state = self._streaming_state[uid]
                    S_list.append(state[0][layer_idx])
                    M_list.append(state[1][layer_idx])
                    V_list.append(state[2][layer_idx])
                    if state[3] is not None:
                        step_list.append(state[3][layer_idx])
                    else:
                        step_list.append(torch.tensor(0.0, device=device, dtype=torch.float32))
                else:
                    d_h = self.embedding_size // self.num_heads
                    S_list.append(torch.zeros(self.num_heads, d_h, d_h, device=device))
                    M_list.append(torch.zeros(self.num_heads, d_h, d_h, device=device))
                    # F-Adam: V 形状为 [num_heads, 1, 1]
                    V_list.append(torch.zeros(self.num_heads, 1, 1, device=device))
                    step_list.append(torch.tensor(0.0, device=device, dtype=torch.float32))
            S_batch_list.append(torch.stack(S_list, dim=0))
            M_batch_list.append(torch.stack(M_list, dim=0))
            V_batch_list.append(torch.stack(V_list, dim=0))
            step_batch_list.append(torch.stack(step_list, dim=0))

        x = item_seq_emb
        for layer_idx, layer in enumerate(self.layers):
            ret = layer(
                x,
                S_init=S_batch_list[layer_idx],
                M_init=M_batch_list[layer_idx],
                V_init=V_batch_list[layer_idx],
                step_init=step_batch_list[layer_idx],
                valid_mask=valid_mask,
                update_mask=update_mask,
                return_S=True,
            )
            x, S_new, M_new, V_new, step_new = ret
            S_batch_list[layer_idx] = S_new
            M_batch_list[layer_idx] = M_new
            V_batch_list[layer_idx] = V_new
            step_batch_list[layer_idx] = step_new

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
                    stored_V = tuple(v[i].detach().clone() for v in V_batch_list)
                    stored_step = tuple(st[i].detach().clone() for st in step_batch_list)
                    if DEBUG_T2T_STREAMING:
                        cnt = getattr(FroRec, "_t2t_debug_m_count", 0)
                        if cnt < 50:
                            m_mean = sum(m.abs().mean().item() for m in stored_M) / len(stored_M)
                            print(f"[DEBUG] FroRec uid={uid} Step {new_len}, M_mean: {m_mean:.6f}")
                            FroRec._t2t_debug_m_count = cnt + 1
                    self._streaming_state[uid] = (stored_S, stored_M, stored_V, stored_step, new_len, device)

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
