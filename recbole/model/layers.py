# -*- coding: utf-8 -*-
# @Time   : 2020/6/27 16:40
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn
# @File   : layers.py

# UPDATE:
# @Time   : 2022/7/16, 2020/8/24 14:58, 2020/9/16, 2020/9/21, 2020/10/9, 2021/05/01
# @Author : Zhen Tian, Yujie Lu, Xingyu Pan, Zhichao Feng, Hui Wang, Xinyan Fan
# @Email  : chenyuwuxinn@gmail.com, yujielu1998@gmail.com, panxy@ruc.edu.cn, fzcbupt@gmail.com, hui.wang@ruc.edu.cn, xinyan.fan@ruc.edu.cn

"""
recbole.model.layers
#############################
Common Layers in recommender system
"""

import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
from torch.nn.init import normal_

from recbole.utils import FeatureType, FeatureSource


class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self,
        layers,
        dropout=0.0,
        activation="relu",
        bn=False,
        init_method=None,
        last_activation=True,
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)
        if self.activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)


def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "dice":
            activation = Dice(emb_dim)
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation


class FMEmbedding(nn.Module):
    r"""Embedding for token fields.

    Args:
        field_dims: list, the number of tokens in each token fields
        offsets: list, the dimension offset of each token field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FMEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        input_x = input_x + input_x.new_tensor(self.offsets).unsqueeze(0)
        output = self.embedding(input_x)
        return output


class FLEmbedding(nn.Module):
    r"""Embedding for float fields.

    Args:
        field_dims: list, the number of float in each float fields
        offsets: list, the dimension offset of each float field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,2)``.

    Return:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.
    """

    def __init__(self, field_dims, offsets, embed_dim):
        super(FLEmbedding, self).__init__()
        self.embedding = nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = offsets

    def forward(self, input_x):
        base, index = torch.split(input_x, [1, 1], dim=-1)
        index = index.squeeze(-1).long()
        index = index + index.new_tensor(self.offsets).unsqueeze(0)
        output = base * self.embedding(index)
        return output


class BaseFactorizationMachine(nn.Module):
    r"""Calculate FM result over the embeddings

    Args:
        reduce_sum: bool, whether to sum the result, default is True.

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

    Output
        output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.
    """

    def __init__(self, reduce_sum=True):
        super(BaseFactorizationMachine, self).__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1) ** 2
        sum_of_square = torch.sum(input_x**2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(
            in_features=in_dim, out_features=out_dim
        )

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1 # 拉普拉斯矩阵
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2


class AttLayer(nn.Module):
    """Calculate the attention signal(weight) according the input tensor.

    Args:
        infeatures (torch.FloatTensor): A 3D input tensor with shape of[batch_size, M, embed_dim].

    Returns:
        torch.FloatTensor: Attention weight of input. shape of [batch_size, M].
    """

    def __init__(self, in_dim, att_dim):
        super(AttLayer, self).__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.w = torch.nn.Linear(in_features=in_dim, out_features=att_dim, bias=False)
        self.h = nn.Parameter(torch.randn(att_dim), requires_grad=True)

    def forward(self, infeatures):
        att_signal = self.w(infeatures)  # [batch_size, M, att_dim]
        att_signal = fn.relu(att_signal)  # [batch_size, M, att_dim]

        att_signal = torch.mul(att_signal, self.h)  # [batch_size, M, att_dim]
        att_signal = torch.sum(att_signal, dim=2)  # [batch_size, M]
        att_signal = fn.softmax(att_signal, dim=1)  # [batch_size, M]

        return att_signal


class Dice(nn.Module):
    r"""Dice activation function

    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s

    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score


class SequenceAttLayer(nn.Module):
    """Attention Layer. Get the representation of each user in the batch.

    Args:
        queries (torch.Tensor): candidate ads, [B, H], H means embedding_size * feat_num
        keys (torch.Tensor): user_hist, [B, T, H]
        keys_length (torch.Tensor): mask, [B]

    Returns:
        torch.Tensor: result
    """

    def __init__(
        self,
        mask_mat,
        att_hidden_size=(80, 40),
        activation="sigmoid",
        softmax_stag=False,
        return_seq_weight=True,
    ):
        super(SequenceAttLayer, self).__init__()
        self.att_hidden_size = att_hidden_size
        self.activation = activation
        self.softmax_stag = softmax_stag
        self.return_seq_weight = return_seq_weight
        self.mask_mat = mask_mat
        self.att_mlp_layers = MLPLayers(
            self.att_hidden_size, activation=self.activation, bn=False
        )
        self.dense = nn.Linear(self.att_hidden_size[-1], 1)

    def forward(self, queries, keys, keys_length):
        embedding_size = queries.shape[-1]  # H
        hist_len = keys.shape[1]  # T
        queries = queries.repeat(1, hist_len)

        queries = queries.view(-1, hist_len, embedding_size)

        # MLP Layer
        input_tensor = torch.cat(
            [queries, keys, queries - keys, queries * keys], dim=-1
        )
        output = self.att_mlp_layers(input_tensor)
        output = torch.transpose(self.dense(output), -1, -2)

        # get mask
        output = output.squeeze(1)
        mask = self.mask_mat.repeat(output.size(0), 1)
        mask = mask >= keys_length.unsqueeze(1)

        # mask
        if self.softmax_stag:
            mask_value = -np.inf
        else:
            mask_value = 0.0

        output = output.masked_fill(mask=mask, value=torch.tensor(mask_value))
        output = output.unsqueeze(1)
        output = output / (embedding_size**0.5)

        # get the weight of each user's history list about the target item
        if self.softmax_stag:
            output = fn.softmax(output, dim=2)  # [B, 1, T]

        if not self.return_seq_weight:
            output = torch.matmul(output, keys)  # [B, 1, H]

        return output


class VanillaAttention(nn.Module):
    """
    Vanilla attention layer is implemented by linear layer.

    Args:
        input_tensor (torch.Tensor): the input of the attention layer

    Returns:
        hidden_states (torch.Tensor): the outputs of the attention layer
        weights (torch.Tensor): the attention weights

    """

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim), nn.ReLU(True), nn.Linear(attn_dim, 1)
        )

    def forward(self, input_tensor):
        # (B, Len, num, H) -> (B, Len, num, 1)
        energy = self.projection(input_tensor)
        weights = torch.softmax(energy.squeeze(-1), dim=-1)
        # (B, Len, num, H) * (B, Len, num, 1) -> (B, len, H)
        hidden_states = (input_tensor * weights.unsqueeze(-1)).sum(dim=-2)
        return hidden_states, weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor

    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 2, 1, 3)
        key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 2, 3, 1)
        value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 2, 1, 3)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)

        attention_scores = attention_scores / self.sqrt_attention_head_size
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.

    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer

    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer

    """

    def __init__(
        self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps
    ):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": self.gelu,
            "relu": fn.relu,
            "swish": self.swish,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return ACT2FN[act]

    def gelu(self, x):
        """Implementation of the gelu activation function.

        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results)::

            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

        Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.

    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r"""One TransformerEncoder consists of several TransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12

    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.

        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class LinRecMultiHeadAttention(nn.Module):
    r"""LinRec: L2-Normalized Linear Attention for sequential recommendation (SIGIR 2023).

    Formula: A'(Q,K,V) = ρ1(elu(Q)) @ (ρ2(elu(K))^T @ V)
    - ρ1: row-wise L2 norm for Q: Q_i / (sqrt(d) * ||Q_i||_2)
    - ρ2: column-wise L2 norm for K: K_j / (sqrt(N) * ||K_j||_2)
    - Causal: strictly uses K[:t+1], V[:t+1] at position t (no future leakage).

    Streaming: uses row-wise K norm to enable O(1) recurrence: S_t = S_{t-1} + (k'_t)^T @ v_t.
    """

    def __init__(
        self,
        n_heads,
        hidden_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(LinRecMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.eps = 1e-8

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def _transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x

    def _rho1(self, Q):
        """Row-wise L2 norm: Q_i / (sqrt(d) * ||Q_i||_2)"""
        d = Q.size(-1)
        norm = Q.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return Q / (math.sqrt(d) * norm)

    def _rho2(self, K):
        """Column-wise L2 norm: K_j / (sqrt(N) * ||K_j||_2)"""
        N = K.size(-2)
        norm = K.norm(dim=-2, keepdim=True).clamp(min=self.eps)
        return K / (math.sqrt(N) * norm)

    def forward(self, input_tensor, attention_mask=None, prev_KV=None, update_mask=None):
        """Forward with optional streaming state.

        prev_KV: [B, n_heads, d, d] from previous run (streaming).
        update_mask: [B, L] bool, True = update KV at this position (streaming).
        Returns (hidden_states, new_KV) if prev_KV/update_mask given, else hidden_states.
        """
        streaming = prev_KV is not None and update_mask is not None
        B, L, _ = input_tensor.size()
        mixed_query = self.query(input_tensor)
        mixed_key = self.key(input_tensor)
        mixed_value = self.value(input_tensor)

        Q = self._transpose_for_scores(mixed_query)  # [B, L, n_heads, d]
        K = self._transpose_for_scores(mixed_key)
        V = self._transpose_for_scores(mixed_value)

        if streaming:
            return self._forward_streaming(
                input_tensor, Q, K, V, prev_KV, update_mask
            )

        outputs = []
        for t in range(L):
            Q_t = Q[:, t, :, :]
            K_t = K[:, : t + 1, :, :]
            V_t = V[:, : t + 1, :, :]

            Q_t = fn.elu(Q_t)
            K_t = fn.elu(K_t)
            V_t = fn.elu(V_t)

            Q_t = self._rho1(Q_t)
            N_t = t + 1
            norm_K = K_t.norm(dim=1, keepdim=True).clamp(min=self.eps)
            K_t = K_t / (math.sqrt(N_t) * norm_K)

            KV = torch.einsum("btnc,btnv->bncv", K_t, V_t)
            out_t = torch.einsum("bnd,bncv->bnv", Q_t, KV)
            outputs.append(out_t)

        context_layer = torch.stack(outputs, dim=1)
        return self._output_proj(input_tensor, context_layer)

    def _forward_streaming(self, input_tensor, Q, K, V, prev_KV, update_mask):
        """Streaming: O(1) per-token via row-wise K norm recurrence S_t = S_{t-1} + (k'_t)^T @ v_t.
        update_mask: [B, L] bool, True = update KV at (b,t)."""
        B, L = Q.size(0), Q.size(1)
        d = self.attention_head_size
        device = Q.device
        KV = prev_KV if prev_KV is not None else torch.zeros(
            B, self.num_attention_heads, d, d, device=device, dtype=Q.dtype
        )
        outputs = []
        for t in range(L):
            Q_t = fn.elu(Q[:, t, :, :])
            K_t = fn.elu(K[:, t, :, :])
            V_t = fn.elu(V[:, t, :, :])
            k_norm = K_t.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            k_prime = K_t / k_norm
            delta = torch.einsum("bnd,bnv->bndv", k_prime, V_t)
            inc = update_mask[:, t].view(B, 1, 1, 1).to(delta.dtype)
            KV = KV + inc * delta
            Q_t = self._rho1(Q_t)
            out_t = torch.einsum("bnd,bncv->bnv", Q_t, KV)
            outputs.append(out_t)
        context_layer = torch.stack(outputs, dim=1)
        return self._output_proj(input_tensor, context_layer), KV

    def _output_proj(self, input_tensor, context_layer):
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.attn_dropout(hidden_states)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LinRecLayer(nn.Module):
    """One LinRec layer: LinRec attention (causal, no target leakage) + feed-forward."""

    def __init__(
        self,
        n_heads,
        hidden_size,
        inner_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(LinRecLayer, self).__init__()
        self.attention = LinRecMultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, attention_mask=None, prev_KV=None, update_mask=None):
        if prev_KV is not None and update_mask is not None:
            attn_out, new_KV = self.attention(
                hidden_states, attention_mask, prev_KV=prev_KV, update_mask=update_mask
            )
            return self.feed_forward(attn_out), new_KV
        attn_out = self.attention(hidden_states, attention_mask)
        return self.feed_forward(attn_out)


class LinRecEncoder(nn.Module):
    """LinRec encoder: stack of LinRec layers. Supports streaming state."""

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(LinRecEncoder, self).__init__()
        layer = LinRecLayer(
            n_heads,
            hidden_size,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_all_encoded_layers=True,
        prev_KV_list=None,
        update_mask=None,
    ):
        streaming = prev_KV_list is not None and update_mask is not None
        KV_list = prev_KV_list if prev_KV_list is not None else [None] * len(self.layer)
        all_encoder_layers = []
        new_KV_list = []
        for layer_idx, layer_module in enumerate(self.layer):
            if streaming:
                hidden_states, new_KV = layer_module(
                    hidden_states,
                    attention_mask,
                    prev_KV=KV_list[layer_idx],
                    update_mask=update_mask,
                )
                new_KV_list.append(new_KV)
            else:
                hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        if streaming:
            return all_encoder_layers, new_KV_list
        return all_encoder_layers


class ItemToInterestAggregation(nn.Module):
    def __init__(self, seq_len, hidden_size, k_interests=5):
        super().__init__()
        self.k_interests = k_interests  # k latent interests
        self.theta = nn.Parameter(torch.randn([hidden_size, k_interests]))

    def forward(self, input_tensor):  # [B, L, d] -> [B, k, d]
        D_matrix = torch.matmul(input_tensor, self.theta)  # [B, L, k]
        D_matrix = nn.Softmax(dim=-2)(D_matrix)
        result = torch.einsum("nij, nik -> nkj", input_tensor, D_matrix)  # #[B, k, d]

        return result


class LightMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        hidden_dropout_prob,
        attn_dropout_prob,
        layer_norm_eps,
    ):
        super(LightMultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # initialization for low-rank decomposed self-attention
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attpooling_key = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )
        self.attpooling_value = ItemToInterestAggregation(
            seq_len, hidden_size, k_interests
        )

        # initialization for decoupled position encoding
        self.attn_scale_factor = 2
        self.pos_q_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_k_linear = nn.Linear(hidden_size, self.all_head_size)
        self.pos_scaling = (
            float(self.attention_head_size * self.attn_scale_factor) ** -0.5
        )
        self.pos_ln = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):  # transfor to multihead
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, pos_emb):
        # linear map
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # low-rank decomposed self-attention: relation of items
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(self.attpooling_key(mixed_key_layer))
        value_layer = self.transpose_for_scores(
            self.attpooling_value(mixed_value_layer)
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-2)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer_item = torch.matmul(attention_probs, value_layer)

        # decoupled position encoding: relation of positions
        value_layer_pos = self.transpose_for_scores(mixed_value_layer)
        pos_emb = self.pos_ln(pos_emb).unsqueeze(0)
        pos_query_layer = (
            self.transpose_for_scores(self.pos_q_linear(pos_emb)) * self.pos_scaling
        )
        pos_key_layer = self.transpose_for_scores(self.pos_k_linear(pos_emb))

        abs_pos_bias = torch.matmul(pos_query_layer, pos_key_layer.transpose(-1, -2))
        abs_pos_bias = abs_pos_bias / math.sqrt(self.attention_head_size)
        abs_pos_bias = nn.Softmax(dim=-2)(abs_pos_bias)

        context_layer_pos = torch.matmul(abs_pos_bias, value_layer_pos)

        context_layer = context_layer_item + context_layer_pos

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class LightTransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.

    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer

    Returns:
        feedforward_output (torch.Tensor): the output of the point-wise feed-forward sublayer, is the output of the transformer layer
    """

    def __init__(
        self,
        n_heads,
        k_interests,
        hidden_size,
        seq_len,
        intermediate_size,
        hidden_dropout_prob,
        attn_dropout_prob,
        hidden_act,
        layer_norm_eps,
    ):
        super(LightTransformerLayer, self).__init__()
        self.multi_head_attention = LightMultiHeadAttention(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            hidden_dropout_prob,
            attn_dropout_prob,
            layer_norm_eps,
        )
        self.feed_forward = FeedForward(
            hidden_size,
            intermediate_size,
            hidden_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )

    def forward(self, hidden_states, pos_emb):
        attention_output = self.multi_head_attention(hidden_states, pos_emb)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class LightTransformerEncoder(nn.Module):
    r"""One LightTransformerEncoder consists of several LightTransformerLayers.

    Args:
        n_layers(num): num of transformer layers in transformer encoder. Default: 2
        n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        hidden_size(num): the input and output hidden size. Default: 64
        inner_size(num): the dimensionality in feed-forward layer. Default: 256
        hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        hidden_act(str): activation function in feed-forward layer. Default: 'gelu'.
            candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        k_interests=5,
        hidden_size=64,
        seq_len=50,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act="gelu",
        layer_norm_eps=1e-12,
    ):
        super(LightTransformerEncoder, self).__init__()
        layer = LightTransformerLayer(
            n_heads,
            k_interests,
            hidden_size,
            seq_len,
            inner_size,
            hidden_dropout_prob,
            attn_dropout_prob,
            hidden_act,
            layer_norm_eps,
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, pos_emb, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TrandformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output

        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer layers' output,
            otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, pos_emb)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class ContextSeqEmbAbstractLayer(nn.Module):
    """For Deep Interest Network and feature-rich sequential recommender systems, return features embedding matrices."""

    def __init__(self):
        super(ContextSeqEmbAbstractLayer, self).__init__()
        self.token_field_offsets = {}
        self.float_field_offsets = {}
        self.token_embedding_table = nn.ModuleDict()
        self.float_embedding_table = nn.ModuleDict()
        self.token_seq_embedding_table = nn.ModuleDict()
        self.float_seq_embedding_table = nn.ModuleDict()

        self.token_field_names = None
        self.token_field_dims = None
        self.float_field_names = None
        self.float_field_dims = None
        self.token_seq_field_names = None
        self.token_seq_field_dims = None
        self.float_seq_field_names = None
        self.float_seq_field_dims = None
        self.num_feature_field = None

    def get_fields_name_dim(self):
        """get user feature field and item feature field."""
        self.token_field_names = {type: [] for type in self.types}
        self.token_field_dims = {type: [] for type in self.types}
        self.float_field_names = {type: [] for type in self.types}
        self.float_field_dims = {type: [] for type in self.types}
        self.token_seq_field_names = {type: [] for type in self.types}
        self.token_seq_field_dims = {type: [] for type in self.types}
        self.num_feature_field = {type: 0 for type in self.types}
        self.float_seq_field_names = {type: [] for type in self.types}
        self.float_seq_field_dims = {type: [] for type in self.types}

        for type in self.types:
            for field_name in self.field_names[type]:
                if self.dataset.field2type[field_name] == FeatureType.TOKEN:
                    self.token_field_names[type].append(field_name)
                    self.token_field_dims[type].append(self.dataset.num(field_name))
                elif self.dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                    self.token_seq_field_names[type].append(field_name)
                    self.token_seq_field_dims[type].append(self.dataset.num(field_name))
                elif (
                    self.dataset.field2type[field_name] == FeatureType.FLOAT
                    and field_name in self.dataset.config["numerical_features"]
                ):
                    self.float_field_names[type].append(field_name)
                    self.float_field_dims[type].append(self.dataset.num(field_name))
                elif (
                    self.dataset.field2type[field_name] == FeatureType.FLOAT_SEQ
                    and field_name in self.dataset.config["numerical_features"]
                ):
                    self.float_seq_field_names[type].append(field_name)
                    self.float_seq_field_dims[type].append(self.dataset.num(field_name))
                else:
                    continue
                self.num_feature_field[type] += 1

    def get_embedding(self):
        """get embedding of all features."""
        for type in self.types:
            if len(self.token_field_dims[type]) > 0:
                self.token_field_offsets[type] = np.array(
                    (0, *np.cumsum(self.token_field_dims[type])[:-1]), dtype=np.long
                )
                self.token_embedding_table[type] = FMEmbedding(
                    self.token_field_dims[type],
                    self.token_field_offsets[type],
                    self.embedding_size,
                ).to(self.device)
            if len(self.float_field_dims[type]) > 0:
                self.float_field_offsets[type] = np.array(
                    (0, *np.cumsum(self.float_field_dims[type])[:-1]), dtype=np.long
                )
                self.float_embedding_table[type] = FLEmbedding(
                    self.float_field_dims[type],
                    self.float_field_offsets[type],
                    self.embedding_size,
                ).to(self.device)
            if len(self.token_seq_field_dims) > 0:
                self.token_seq_embedding_table[type] = nn.ModuleList()
                for token_seq_field_dim in self.token_seq_field_dims[type]:
                    self.token_seq_embedding_table[type].append(
                        nn.Embedding(token_seq_field_dim, self.embedding_size).to(
                            self.device
                        )
                    )
            if len(self.float_seq_field_dims) > 0:
                self.float_seq_embedding_table[type] = nn.ModuleList()
                for float_seq_field_dim in self.float_seq_field_dims[type]:
                    self.float_seq_embedding_table[type].append(
                        nn.Embedding(float_seq_field_dim, self.embedding_size).to(
                            self.device
                        )
                    )

    def embed_float_fields(self, float_fields, type, embed=True):
        """Get the embedding of float fields.
        In the following three functions("embed_float_fields" "embed_token_fields" "embed_token_seq_fields")
        when the type is user, [batch_size, max_item_length] should be recognised as [batch_size]

        Args:
            float_fields(torch.Tensor): [batch_size, max_item_length, num_float_field]
            type(str): user or item
            embed(bool): embed or not

        Returns:
            torch.Tensor: float fields embedding. [batch_size, max_item_length, num_float_field, embed_dim]

        """
        if float_fields is None:
            return None

        if type == "item":
            embedding_shape = float_fields.shape[:-1] + (-1,)
            float_fields = float_fields.reshape(
                -1, float_fields.shape[-2], float_fields.shape[-1]
            )
            float_embedding = self.float_embedding_table[type](float_fields)
            float_embedding = float_embedding.view(embedding_shape)
        else:
            float_embedding = self.float_embedding_table[type](float_fields)

        return float_embedding

    def embed_token_fields(self, token_fields, type):
        """Get the embedding of token fields

        Args:
            token_fields(torch.Tensor): input, [batch_size, max_item_length, num_token_field]
            type(str): user or item

        Returns:
            torch.Tensor: token fields embedding, [batch_size, max_item_length, num_token_field, embed_dim]

        """
        if token_fields is None:
            return None
        # [batch_size, max_item_length, num_token_field, embed_dim]
        if type == "item":
            embedding_shape = token_fields.shape + (-1,)
            token_fields = token_fields.reshape(-1, token_fields.shape[-1])
            token_embedding = self.token_embedding_table[type](token_fields)
            token_embedding = token_embedding.view(embedding_shape)
        else:
            token_embedding = self.token_embedding_table[type](token_fields)
        return token_embedding

    def embed_float_seq_fields(self, float_seq_fields, type):
        """Embed the float sequence feature columns

        Args:
            float_seq_fields (torch.FloatTensor): The input tensor. shape of [batch_size, seq_len, 2]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of float sequence columns.
        """
        fields_result = []
        for i, float_seq_field in enumerate(float_seq_fields):
            embedding_table = self.float_seq_embedding_table[type][i]
            base, index = torch.split(float_seq_field, [1, 1], dim=-1)
            index = index.squeeze(-1)
            mask = index != 0
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=-1, keepdim=True)
            float_seq_embedding = base * embedding_table(index.long())
            mask = mask.unsqueeze(-1).expand_as(float_seq_embedding)
            if self.pooling_mode == "max":
                masked_float_seq_embedding = float_seq_embedding - (1 - mask) * 1e9
                result = torch.max(masked_float_seq_embedding, dim=-2, keepdim=True)
                result = result.values
            elif self.pooling_mode == "sum":
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=-2, keepdim=True)
            else:
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(masked_float_seq_embedding, dim=-2)
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)
                result = result.unsqueeze(-2)

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(fields_result, dim=-2)

    def embed_token_seq_fields(self, token_seq_fields, type):
        """Get the embedding of token_seq fields.

        Args:
            token_seq_fields(torch.Tensor): input, [batch_size, max_item_length, seq_len]`
            type(str): user or item
            mode(str): mean/max/sum

        Returns:
            torch.Tensor: result [batch_size, max_item_length, num_token_seq_field, embed_dim]

        """
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[type][i]
            mask = token_seq_field != 0  # [batch_size, max_item_length, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(
                mask, dim=-1, keepdim=True
            )  # [batch_size, max_item_length, 1]
            token_seq_embedding = embedding_table(
                token_seq_field
            )  # [batch_size, max_item_length, seq_len, embed_dim]
            mask = mask.unsqueeze(-1).expand_as(token_seq_embedding)
            if self.pooling_mode == "max":
                masked_token_seq_embedding = token_seq_embedding - (1 - mask) * 1e9
                result = torch.max(
                    masked_token_seq_embedding, dim=-2, keepdim=True
                )  # [batch_size, max_item_length, 1, embed_dim]
                result = result.values
            elif self.pooling_mode == "sum":
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=-2, keepdim=True
                )  # [batch_size, max_item_length, 1, embed_dim]
            else:
                masked_token_seq_embedding = token_seq_embedding * mask.float()
                result = torch.sum(
                    masked_token_seq_embedding, dim=-2
                )  # [batch_size, max_item_length, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(
                    result, value_cnt + eps
                )  # [batch_size, max_item_length, embed_dim]
                result = result.unsqueeze(
                    -2
                )  # [batch_size, max_item_length, 1, embed_dim]

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.cat(
                fields_result, dim=-2
            )  # [batch_size, max_item_length, num_token_seq_field, embed_dim]

    def embed_input_fields(self, user_idx, item_idx):
        """Get the embedding of user_idx and item_idx

        Args:
            user_idx(torch.Tensor): interaction['user_id']
            item_idx(torch.Tensor): interaction['item_id_list']

        Returns:
            dict: embedding of user feature and item feature

        """
        user_item_feat = {"user": self.user_feat, "item": self.item_feat}
        user_item_idx = {"user": user_idx, "item": item_idx}
        float_fields_embedding = {}
        float_seq_fields_embedding = {}
        token_fields_embedding = {}
        token_seq_fields_embedding = {}
        sparse_embedding = {}
        dense_embedding = {}

        for type in self.types:
            float_fields = []
            for field_name in self.float_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_fields.append(
                    feature
                    if len(feature.shape) == (3 + (type == "item"))
                    else feature.unsqueeze(-2)
                )
            if len(float_fields) > 0:
                float_fields = torch.cat(
                    float_fields, dim=-1
                )  # [batch_size, max_item_length, num_float_field]
            else:
                float_fields = None
            float_fields_embedding[type] = self.embed_float_fields(float_fields, type)

            float_seq_fields = []
            for field_name in self.float_seq_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                float_seq_fields.append(feature)
            # [batch_size, max_item_length, num_token_seq_field, embed_dim] or None
            float_seq_fields_embedding[type] = self.embed_float_seq_fields(
                float_seq_fields, type
            )

            if float_fields_embedding[type] is None:
                dense_embedding[type] = float_seq_fields_embedding[type]
            else:
                if float_seq_fields_embedding[type] is None:
                    dense_embedding[type] = float_fields_embedding[type]
                else:
                    dense_embedding[type] = torch.cat(
                        [
                            float_fields_embedding[type],
                            float_seq_fields_embedding[type],
                        ],
                        dim=-2,
                    )

            token_fields = []
            for field_name in self.token_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_fields.append(feature.unsqueeze(-1))
            if len(token_fields) > 0:
                token_fields = torch.cat(
                    token_fields, dim=-1
                )  # [batch_size, max_item_length, num_token_field]
            else:
                token_fields = None
            # [batch_size, max_item_length, num_token_field, embed_dim] or None
            token_fields_embedding[type] = self.embed_token_fields(token_fields, type)

            token_seq_fields = []
            for field_name in self.token_seq_field_names[type]:
                feature = user_item_feat[type][field_name][user_item_idx[type]]
                token_seq_fields.append(feature)
            # [batch_size, max_item_length, num_token_seq_field, embed_dim] or None
            token_seq_fields_embedding[type] = self.embed_token_seq_fields(
                token_seq_fields, type
            )

            if token_fields_embedding[type] is None:
                sparse_embedding[type] = token_seq_fields_embedding[type]
            else:
                if token_seq_fields_embedding[type] is None:
                    sparse_embedding[type] = token_fields_embedding[type]
                else:
                    sparse_embedding[type] = torch.cat(
                        [
                            token_fields_embedding[type],
                            token_seq_fields_embedding[type],
                        ],
                        dim=-2,
                    )

        # sparse_embedding[type]
        # shape: [batch_size, max_item_length, num_token_seq_field+num_token_field, embed_dim] or None
        # dense_embedding[type]
        # shape: [batch_size, max_item_length, num_float_field]
        #     or [batch_size, max_item_length, num_float_field, embed_dim] or None
        return sparse_embedding, dense_embedding

    def forward(self, user_idx, item_idx):
        return self.embed_input_fields(user_idx, item_idx)


class ContextSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For Deep Interest Network, return all features (including user features and item features) embedding matrices."""

    def __init__(self, dataset, embedding_size, pooling_mode, device):
        super(ContextSeqEmbLayer, self).__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = self.dataset.get_user_feature().to(self.device)
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {
            "user": list(self.user_feat.interaction.keys()),
            "item": list(self.item_feat.interaction.keys()),
        }

        self.types = ["user", "item"]
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ["mean", "max", "sum"]
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class FeatureSeqEmbLayer(ContextSeqEmbAbstractLayer):
    """For feature-rich sequential recommenders, return item features embedding matrices according to
    selected features."""

    def __init__(
        self, dataset, embedding_size, selected_features, pooling_mode, device
    ):
        super(FeatureSeqEmbLayer, self).__init__()

        self.device = device
        self.embedding_size = embedding_size
        self.dataset = dataset
        self.user_feat = None
        self.item_feat = self.dataset.get_item_feature().to(self.device)

        self.field_names = {"item": selected_features}

        self.types = ["item"]
        self.pooling_mode = pooling_mode
        try:
            assert self.pooling_mode in ["mean", "max", "sum"]
        except AssertionError:
            raise AssertionError("Make sure 'pooling_mode' in ['mean', 'max', 'sum']!")
        self.get_fields_name_dim()
        self.get_embedding()


class CNNLayers(nn.Module):
    r"""CNNLayers

    Args:
        - channels(list): a list contains the channels of each layer in cnn layers
        - kernel(list): a list contains the kernels of each layer in cnn layers
        - strides(list): a list contains the channels of each layer in cnn layers
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'
                      candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

        .. math::
            H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                      \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                      \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Examples::

        >>> m = CNNLayers([1, 32, 32], [2,2], [2,2], 'relu')
        >>> input = torch.randn(128, 1, 64, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 32, 16, 16])
    """

    def __init__(self, channels, kernels, strides, activation="relu", init_method=None):
        super(CNNLayers, self).__init__()
        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.activation = activation
        self.init_method = init_method
        self.num_of_nets = len(self.channels) - 1

        if len(kernels) != len(strides) or self.num_of_nets != (len(kernels)):
            raise RuntimeError("channels, kernels and strides don't match\n")

        cnn_modules = []

        for i in range(self.num_of_nets):
            cnn_modules.append(
                nn.Conv2d(
                    self.channels[i],
                    self.channels[i + 1],
                    self.kernels[i],
                    stride=self.strides[i],
                )
            )
            if self.activation.lower() == "sigmoid":
                cnn_modules.append(nn.Sigmoid())
            elif self.activation.lower() == "tanh":
                cnn_modules.append(nn.Tanh())
            elif self.activation.lower() == "relu":
                cnn_modules.append(nn.ReLU())
            elif self.activation.lower() == "leakyrelu":
                cnn_modules.append(nn.LeakyReLU())
            elif self.activation.lower() == "none":
                pass

        self.cnn_layers = nn.Sequential(*cnn_modules)

        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Conv2d):
            if self.init_method == "norm":
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.cnn_layers(input_feature)


class FMFirstOrderLinear(nn.Module):
    """Calculate the first order score of the input features.
    This class is a member of ContextRecommender, you can call it easily when inherit ContextRecommender.

    """

    def __init__(self, config, dataset, output_dim=1):
        super(FMFirstOrderLinear, self).__init__()
        self.field_names = dataset.fields(
            source=[
                FeatureSource.INTERACTION,
                FeatureSource.USER,
                FeatureSource.USER_ID,
                FeatureSource.ITEM,
                FeatureSource.ITEM_ID,
            ]
        )
        self.LABEL = config["LABEL_FIELD"]
        self.device = config["device"]
        self.numerical_features = config["numerical_features"]
        self.token_field_names = []
        self.token_field_dims = []
        self.float_field_names = []
        self.float_field_dims = []
        self.token_seq_field_names = []
        self.token_seq_field_dims = []
        self.float_seq_field_names = []
        self.float_seq_field_dims = []

        for field_name in self.field_names:
            if field_name == self.LABEL:
                continue
            if dataset.field2type[field_name] == FeatureType.TOKEN:
                self.token_field_names.append(field_name)
                self.token_field_dims.append(dataset.num(field_name))
            elif dataset.field2type[field_name] == FeatureType.TOKEN_SEQ:
                self.token_seq_field_names.append(field_name)
                self.token_seq_field_dims.append(dataset.num(field_name))
            elif (
                dataset.field2type[field_name] == FeatureType.FLOAT
                and field_name in self.numerical_features
            ):
                self.float_field_names.append(field_name)
                self.float_field_dims.append(dataset.num(field_name))
            elif (
                dataset.field2type[field_name] == FeatureType.FLOAT_SEQ
                and field_name in self.numerical_features
            ):
                self.float_seq_field_names.append(field_name)
                self.float_seq_field_dims.append(dataset.num(field_name))

        if len(self.token_field_dims) > 0:
            self.token_field_offsets = np.array(
                (0, *np.cumsum(self.token_field_dims)[:-1]), dtype=np.long
            )
            self.token_embedding_table = FMEmbedding(
                self.token_field_dims, self.token_field_offsets, output_dim
            )
        if len(self.float_field_dims) > 0:
            self.float_field_offsets = np.array(
                (0, *np.cumsum(self.float_field_dims)[:-1]), dtype=np.long
            )
            self.float_embedding_table = FLEmbedding(
                self.float_field_dims, self.float_field_offsets, output_dim
            )
        if len(self.token_seq_field_dims) > 0:
            self.token_seq_embedding_table = nn.ModuleList()
            for token_seq_field_dim in self.token_seq_field_dims:
                self.token_seq_embedding_table.append(
                    nn.Embedding(token_seq_field_dim, output_dim)
                )
        if len(self.float_seq_field_dims) > 0:
            self.float_seq_embedding_table = nn.ModuleList()
            for float_seq_field_dim in self.float_seq_field_dims:
                self.float_seq_embedding_table.append(
                    nn.Embedding(float_seq_field_dim, output_dim)
                )

        self.bias = nn.Parameter(torch.zeros((output_dim,)), requires_grad=True)

    def embed_float_fields(self, float_fields):
        """Embed the float feature columns

        Args:
            float_fields (torch.FloatTensor): The input dense tensor. shape of [batch_size, num_float_field, 2]
            embed (bool): Return the embedding of columns or just the columns itself. Defaults to ``True``.

        Returns:
            torch.FloatTensor: The result embedding tensor of float columns.
        """
        # input Tensor shape : [batch_size, num_float_field]
        if float_fields is None:
            return None
        # [batch_size, num_float_field, embed_dim]
        float_embedding = self.float_embedding_table(float_fields)

        # [batch_size, 1, output_dim]
        float_embedding = torch.sum(float_embedding, dim=1, keepdim=True)
        return float_embedding

    def embed_float_seq_fields(self, float_seq_fields, mode="mean"):
        """Embed the float sequence feature columns

        Args:
            float_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len, 2]
            mode (str): How to aggregate the embedding of feature in this field. default=mean

        Returns:
            torch.FloatTensor: The result embedding tensor of float sequence columns.
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, float_seq_field in enumerate(float_seq_fields):
            embedding_table = self.float_seq_embedding_table[i]
            base, index = torch.split(float_seq_field, [1, 1], dim=-1)
            index = index.squeeze(-1)
            mask = index != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            float_seq_embedding = base * embedding_table(
                index.long()
            )  # [batch_size, seq_len, embed_dim]

            mask = mask.unsqueeze(2).expand_as(
                float_seq_embedding
            )  # [batch_size, seq_len, embed_dim]
            if mode == "max":
                masked_float_seq_embedding = (
                    float_seq_embedding - (1 - mask) * 1e9
                )  # [batch_size, seq_len, embed_dim]
                result = torch.max(
                    masked_float_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            elif mode == "sum":
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(
                    masked_float_seq_embedding, dim=1, keepdim=True
                )  # [batch_size, 1, embed_dim]
            else:
                masked_float_seq_embedding = float_seq_embedding * mask.float()
                result = torch.sum(
                    masked_float_seq_embedding, dim=1
                )  # [batch_size, embed_dim]
                eps = torch.FloatTensor([1e-8]).to(self.device)
                result = torch.div(result, value_cnt + eps)  # [batch_size, embed_dim]
                result = result.unsqueeze(1)  # [batch_size, 1, embed_dim]
            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.sum(
                torch.cat(fields_result, dim=1), dim=1, keepdim=True
            )  # [batch_size, num_token_seq_field, embed_dim]

    def embed_token_fields(self, token_fields):
        """Calculate the first order score of token feature columns

        Args:
            token_fields (torch.LongTensor): The input tensor. shape of [batch_size, num_token_field]

        Returns:
            torch.FloatTensor: The first order score of token feature columns
        """
        # input Tensor shape : [batch_size, num_token_field]
        if token_fields is None:
            return None
        # [batch_size, num_token_field, embed_dim]
        token_embedding = self.token_embedding_table(token_fields)
        # [batch_size, 1, output_dim]
        token_embedding = torch.sum(token_embedding, dim=1, keepdim=True)

        return token_embedding

    def embed_token_seq_fields(self, token_seq_fields):
        """Calculate the first order score of token sequence feature columns

        Args:
            token_seq_fields (torch.LongTensor): The input tensor. shape of [batch_size, seq_len]

        Returns:
            torch.FloatTensor: The first order score of token sequence feature columns
        """
        # input is a list of Tensor shape of [batch_size, seq_len]
        fields_result = []
        for i, token_seq_field in enumerate(token_seq_fields):
            embedding_table = self.token_seq_embedding_table[i]
            mask = token_seq_field != 0  # [batch_size, seq_len]
            mask = mask.float()
            value_cnt = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]

            token_seq_embedding = embedding_table(
                token_seq_field
            )  # [batch_size, seq_len, output_dim]

            mask = mask.unsqueeze(2).expand_as(
                token_seq_embedding
            )  # [batch_size, seq_len, output_dim]
            masked_token_seq_embedding = token_seq_embedding * mask.float()
            result = torch.sum(
                masked_token_seq_embedding, dim=1, keepdim=True
            )  # [batch_size, 1, output_dim]

            fields_result.append(result)
        if len(fields_result) == 0:
            return None
        else:
            return torch.sum(
                torch.cat(fields_result, dim=1), dim=1, keepdim=True
            )  # [batch_size, 1, output_dim]

    def forward(self, interaction):
        total_fields_embedding = []
        float_fields = []
        for field_name in self.float_field_names:
            if len(interaction[field_name].shape) == 3:
                float_fields.append(interaction[field_name])
            else:
                float_fields.append(interaction[field_name].unsqueeze(1))

        if len(float_fields) > 0:
            float_fields = torch.cat(float_fields, dim=1)
        else:
            float_fields = None

        float_fields_embedding = self.embed_float_fields(float_fields)

        if float_fields_embedding is not None:
            total_fields_embedding.append(float_fields_embedding)

        float_seq_fields = []
        for field_name in self.float_seq_field_names:
            float_seq_fields.append(interaction[field_name])

        float_seq_fields_embedding = self.embed_float_seq_fields(float_seq_fields)

        if float_seq_fields_embedding is not None:
            total_fields_embedding.append(float_seq_fields_embedding)

        token_fields = []
        for field_name in self.token_field_names:
            token_fields.append(interaction[field_name].unsqueeze(1))
        if len(token_fields) > 0:
            token_fields = torch.cat(
                token_fields, dim=1
            )  # [batch_size, num_token_field]
        else:
            token_fields = None
        # [batch_size, 1, output_dim] or None
        token_fields_embedding = self.embed_token_fields(token_fields)
        if token_fields_embedding is not None:
            total_fields_embedding.append(token_fields_embedding)

        token_seq_fields = []
        for field_name in self.token_seq_field_names:
            token_seq_fields.append(interaction[field_name])
        # [batch_size, 1, output_dim] or None
        token_seq_fields_embedding = self.embed_token_seq_fields(token_seq_fields)
        if token_seq_fields_embedding is not None:
            total_fields_embedding.append(token_seq_fields_embedding)

        return (
            torch.sum(torch.cat(total_fields_embedding, dim=1), dim=1) + self.bias
        )  # [batch_size, output_dim]


class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)


# --- Gated Delta Layer (GDN / ICLR 2025) ---

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    _FLA_AVAILABLE = True
except ImportError:
    _FLA_AVAILABLE = False

# 设置 DAMREC_DEBUG_FLA=1 时，首个前向仅 chunk0 打印 FLA 前后张量范围（定位爆炸环节）
_DEBUG_FLA = os.environ.get("DAMREC_DEBUG_FLA", "").lower() in ("1", "true", "yes")


def l2_norm(x, eps=1e-5):
    """L2 norm: x / sqrt(sum(x^2) + eps). Aligned with original GatedDeltaNet."""
    norm = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) + eps).clamp(min=1e-4)
    return x / norm


class CausalDepthwiseConv1d(nn.Module):
    """Causal 1D depthwise convolution. No future info leak."""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, padding=0)

    def forward(self, x):
        # x: [B, L, d] -> [B, d, L]
        x = x.transpose(1, 2)
        x = fn.pad(x, (self.padding, 0), mode="constant", value=0)
        x = self.conv(x)
        return x.transpose(1, 2)  # [B, L, d]


class GatedDeltaLayer(nn.Module):
    r"""Single Gated Delta layer. Aligned with NVlabs GatedDeltaNet:
    QKV proj -> Conv on Q,K,V -> L2 norm -> Multi-Head Delta -> Output Gate -> FFN.

    Delta rule: S_t = β_t S_{t-1} + γ_t (v_t - S_{t-1} k_t) ⊗ k_t^T
    """

    _fla_path_logged = False  # 仅首次 forward 时打印

    def __init__(
        self,
        d_model,
        num_heads=4,
        conv_kernel_size=3,
        ffn_ratio=4,
        dropout=0.1,
        use_fla=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self._use_fla = (use_fla if use_fla is not None else _FLA_AVAILABLE) and _FLA_AVAILABLE

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.beta_gate = nn.Linear(d_model, 1)
        self.gamma_gate = nn.Linear(d_model, 1)
        self.out_gate = nn.Linear(d_model, 1)

        # SwiGLU FFN: (W_gate x) * silu(W_up x) @ W_down
        ffn_hidden = d_model * ffn_ratio
        self.ffn_gate = nn.Linear(d_model, ffn_hidden)
        self.ffn_up = nn.Linear(d_model, ffn_hidden)
        self.ffn_down = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        # 遗忘门偏置初始化：sigmoid(1)≈0.73，避免初始 β=0.5 导致记忆瞬间衰减 (LSTM/Mamba 常用 trick)
        nn.init.constant_(self.beta_gate.bias, 1.0)

    def _delta_step_multihead(self, q, k, v, S, beta, gamma):
        """Rank-1 update per head: S_t = β S + γ (v - S k) ⊗ k^T. No scaling (original formula)."""
        Sk = torch.einsum("bhij,bhj->bhi", S, k)
        residual = v - Sk
        update = torch.einsum("bhi,bhj->bhij", residual, k)
        S_new = beta * S + gamma * update
        return S_new

    def forward(
        self,
        x,
        S_init=None,
        valid_mask=None,
        update_mask=None,
        return_S=False,
    ):
        """Forward over sequence. x: [B, L, d]. Returns [B, L, d].
        update_mask: if given (streaming), only update S where True; else use valid_mask.
        Note: FLA chunk path may differ slightly from Python loop (chunk parallelism vs sequential).
        """
        B, L, d = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        q = l2_norm(q)
        k = l2_norm(k)

        q = q.view(B, L, self.num_heads, self.d_head)  # [B,L,H,d_h]
        k = k.view(B, L, self.num_heads, self.d_head)
        v = v.view(B, L, self.num_heads, self.d_head)

        beta_gates = torch.sigmoid(self.beta_gate(x))  # [B, L, 1]
        gamma_gates = torch.sigmoid(self.gamma_gate(x))
        out_gates = torch.sigmoid(self.out_gate(x))

        # [Streaming 兼容 FLA] 对不更新的位置：k,v 用极小值替代 0，避免 FLA 内部除零 NaN
        vmask = valid_mask if valid_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        mask_for_update = (vmask & update_mask) if update_mask is not None else vmask
        if valid_mask is not None or update_mask is not None:
            mask_H = mask_for_update.view(B, L, 1, 1)
            eps_kv = 1e-8
            k = torch.where(mask_H.expand_as(k).bool(), k, torch.full_like(k, eps_kv))
            v = torch.where(mask_H.expand_as(v).bool(), v, torch.full_like(v, eps_kv))
            beta_gates = torch.where(mask_for_update.unsqueeze(-1), beta_gates, torch.ones_like(beta_gates))

        use_fla = self._use_fla and x.is_cuda
        if use_fla:
            if not GatedDeltaLayer._fla_path_logged:
                import logging
                logging.getLogger("recbole").info(
                    "[GatedDeltaLayer] FLA chunk path active, scale=1.0 (与 Python 循环对齐)"
                )
                GatedDeltaLayer._fla_path_logged = True
            g = torch.log(beta_gates.expand(-1, -1, self.num_heads).clamp(min=1e-8))
            beta_fla = gamma_gates.expand(-1, -1, self.num_heads)
            h0 = S_init
            # scale=1.0: 与 Python 循环一致，FLA 默认 1/sqrt(d_head) 会导致输出缩小，recall 异常
            o_fla, S = chunk_gated_delta_rule(
                q=q, k=k, v=v,
                g=g, beta=beta_fla,
                scale=1.0,
                initial_state=h0,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
            out_seq = o_fla.reshape(B, L, d)
        else:
            q = q.transpose(1, 2)  # [B,H,L,d_h]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if S_init is None:
                S = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device)
            else:
                S = S_init
            outputs = []
            for t in range(L):
                q_t = q[:, :, t, :]
                k_t = k[:, :, t, :]
                v_t = v[:, :, t, :]
                beta_t = beta_gates[:, t, :].view(B, 1, 1, 1)
                gamma_t = gamma_gates[:, t, :].view(B, 1, 1, 1)
                S_new = self._delta_step_multihead(q_t, k_t, v_t, S, beta_t, gamma_t)
                mask_t = mask_for_update[:, t].view(B, 1, 1, 1)
                S = torch.where(mask_t, S_new, S)
                out_t = torch.einsum("bhij,bhj->bhi", S, q_t)
                outputs.append(out_t)
            out_seq = torch.stack(outputs, dim=2).permute(0, 2, 1, 3).reshape(B, L, d)
        out_seq = out_seq * out_gates + x  # residual with input
        ffn = self.ffn_gate(out_seq) * fn.silu(self.ffn_up(out_seq))
        out_seq = out_seq + self.dropout(self.ffn_down(ffn))

        if return_S:
            return out_seq, S
        return out_seq


class GatedDeltaLayerMomentum(nn.Module):
    r"""Momentum variant of Gated Delta layer: SGD → Momentum SGD.
    m_t = μ m_{t-1} + γ (v - S k) ⊗ k^T
    S_t = S_{t-1} + m_t
    No FLA path (momentum requires per-step velocity); streaming stores (S, M) per user.
    """

    def __init__(
        self,
        d_model,
        num_heads=4,
        conv_kernel_size=3,
        ffn_ratio=4,
        dropout=0.1,
        momentum=0.9,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.momentum = momentum

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.gamma_gate = nn.Linear(d_model, 1)
        self.out_gate = nn.Linear(d_model, 1)

        ffn_hidden = d_model * ffn_ratio
        self.ffn_gate = nn.Linear(d_model, ffn_hidden)
        self.ffn_up = nn.Linear(d_model, ffn_hidden)
        self.ffn_down = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def _momentum_step(self, q, k, v, S, M, gamma):
        """m_new = μ*m + γ*(v-Sk)⊗k^T; S_new = S + m_new."""
        Sk = torch.einsum("bhij,bhj->bhi", S, k)
        delta = torch.einsum("bhi,bhj->bhij", v - Sk, k)
        M_new = self.momentum * M + gamma * delta
        S_new = S + M_new
        return S_new, M_new

    def forward(
        self,
        x,
        S_init=None,
        M_init=None,
        valid_mask=None,
        update_mask=None,
        return_S=False,
    ):
        """Forward. Returns (out, S, M) when return_S=True."""
        B, L, d = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        q = l2_norm(q)
        k = l2_norm(k)

        q = q.view(B, L, self.num_heads, self.d_head)
        k = k.view(B, L, self.num_heads, self.d_head)
        v = v.view(B, L, self.num_heads, self.d_head)

        gamma_gates = torch.sigmoid(self.gamma_gate(x))
        out_gates = torch.sigmoid(self.out_gate(x))

        vmask = valid_mask if valid_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        mask_for_update = (vmask & update_mask) if update_mask is not None else vmask
        if valid_mask is not None or update_mask is not None:
            mask_H = mask_for_update.view(B, L, 1, 1).to(k.dtype)
            k = k * mask_H
            v = v * mask_H

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if S_init is None:
            S = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device)
        else:
            S = S_init
        if M_init is None:
            M = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device)
        else:
            M = M_init

        outputs = []
        for t in range(L):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            gamma_t = gamma_gates[:, t, :].view(B, 1, 1, 1)
            S_new, M_new = self._momentum_step(q_t, k_t, v_t, S, M, gamma_t)
            mask_t = mask_for_update[:, t].view(B, 1, 1, 1)
            S = torch.where(mask_t, S_new, S)
            M = torch.where(mask_t, M_new, M)
            out_t = torch.einsum("bhij,bhj->bhi", S, q_t)
            outputs.append(out_t)

        out_seq = torch.stack(outputs, dim=2).permute(0, 2, 1, 3).reshape(B, L, d)
        out_seq = out_seq * out_gates + x
        ffn = self.ffn_gate(out_seq) * fn.silu(self.ffn_up(out_seq))
        out_seq = out_seq + self.dropout(self.ffn_down(ffn))

        if return_S:
            return out_seq, S, M
        return out_seq


class GatedDeltaLayerNesterov(nn.Module):
    r"""Nesterov variant of Gated Delta layer: Momentum → Nesterov Momentum.
    双门控 (β 衰减, γ 输入): M = μ*M + γ*Δ; M_nesterov = μ*M + γ*Δ; S_t = β*S_{t-1} + M_nesterov
    提前看一步的等效更新方向，β 衰减历史记忆防止流式下数值膨胀与兴趣漂移。
    """

    def __init__(
        self,
        d_model,
        num_heads=4,
        conv_kernel_size=3,
        ffn_ratio=4,
        dropout=0.1,
        momentum=0.9,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.momentum = momentum

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.beta_gate = nn.Linear(d_model, 1)
        self.gamma_gate = nn.Linear(d_model, 1)
        self.out_gate = nn.Linear(d_model, 1)

        ffn_hidden = d_model * ffn_ratio
        self.ffn_gate = nn.Linear(d_model, ffn_hidden)
        self.ffn_up = nn.Linear(d_model, ffn_hidden)
        self.ffn_down = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.constant_(self.beta_gate.bias, 1.0)

    def _nesterov_step(self, q, k, v, S, M, beta, gamma):
        """Nesterov with decay: M = μ*M + γ*Δ; M_nesterov = μ*M + γ*Δ; S_new = β*S + M_nesterov."""
        Sk = torch.einsum("bhij,bhj->bhi", S, k)
        delta = torch.einsum("bhi,bhj->bhij", v - Sk, k)
        M = self.momentum * M + gamma * delta
        M_nesterov = self.momentum * M + gamma * delta
        # 加入 beta 衰减历史记忆
        S_new = beta * S + M_nesterov
        return S_new, M

    def forward(
        self,
        x,
        S_init=None,
        M_init=None,
        valid_mask=None,
        update_mask=None,
        return_S=False,
    ):
        """Forward. Returns (out, S, M) when return_S=True."""
        B, L, d = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        q = l2_norm(q)
        k = l2_norm(k)

        q = q.view(B, L, self.num_heads, self.d_head)
        k = k.view(B, L, self.num_heads, self.d_head)
        v = v.view(B, L, self.num_heads, self.d_head)

        beta_gates = torch.sigmoid(self.beta_gate(x))
        gamma_gates = torch.sigmoid(self.gamma_gate(x))
        out_gates = torch.sigmoid(self.out_gate(x))

        vmask = valid_mask if valid_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        mask_for_update = (vmask & update_mask) if update_mask is not None else vmask
        if valid_mask is not None or update_mask is not None:
            mask_H = mask_for_update.view(B, L, 1, 1).to(k.dtype)
            k = k * mask_H
            v = v * mask_H
            # 在不更新的位置，强制 beta=1 (100% 保留记忆)，与 FLA 版本对齐
            beta_gates = torch.where(mask_for_update.unsqueeze(-1), beta_gates, torch.ones_like(beta_gates))

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if S_init is None:
            S = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device)
        else:
            S = S_init
        if M_init is None:
            M = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device)
        else:
            M = M_init

        outputs = []
        for t in range(L):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            beta_t = beta_gates[:, t, :].view(B, 1, 1, 1)
            gamma_t = gamma_gates[:, t, :].view(B, 1, 1, 1)
            S_new, M_new = self._nesterov_step(q_t, k_t, v_t, S, M, beta_t, gamma_t)
            mask_t = mask_for_update[:, t].view(B, 1, 1, 1)
            S = torch.where(mask_t, S_new, S)
            M = torch.where(mask_t, M_new, M)
            out_t = torch.einsum("bhij,bhj->bhi", S, q_t)
            outputs.append(out_t)

        out_seq = torch.stack(outputs, dim=2).permute(0, 2, 1, 3).reshape(B, L, d)
        out_seq = out_seq * out_gates + x
        ffn = self.ffn_gate(out_seq) * fn.silu(self.ffn_up(out_seq))
        out_seq = out_seq + self.dropout(self.ffn_down(ffn))

        if return_S:
            return out_seq, S, M
        return out_seq


class GatedDeltaLayerDamRec(nn.Module):
    r"""DamRec token-level layer per instruct.md §5–6, §11.
    S̃=αS, r=S̃k−v, V_r/V_k rank-1 EMA of (r⊙r),(k⊙k), P=√(V_r V_k^T / (1−ρ^t))+ε,
    S=αS+β((−r)k^T ⊘ P). Second-moment updates are stop-gradient."""

    def __init__(
        self,
        d_model,
        num_heads=4,
        conv_kernel_size=3,
        ffn_ratio=4,
        dropout=0.1,
        damrec_rho=0.99,
        damrec_eps=1e-8,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rho = damrec_rho
        self.eps = damrec_eps

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.alpha_gate = nn.Linear(d_model, 1)
        self.beta_gate = nn.Linear(d_model, 1)
        self.out_gate = nn.Linear(d_model, 1)
        self.out_norm = nn.GroupNorm(num_groups=num_heads, num_channels=d_model)

        ffn_hidden = d_model * ffn_ratio
        self.ffn_gate = nn.Linear(d_model, ffn_hidden)
        self.ffn_up = nn.Linear(d_model, ffn_hidden)
        self.ffn_down = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def _damrec_step(self, k, v, S, V_r, V_k, alpha, beta, step_1based):
        """V_r,V_k: [B,H,d_h]; step_1based: scalar tensor for bias (1−ρ^t)."""
        S_tilde = alpha * S
        Sk = torch.einsum("bhij,bhj->bhi", S_tilde, k)
        r = Sk - v
        rr = (r * r).detach()
        kk = (k * k).detach()
        with torch.no_grad():
            V_r_new = self.rho * V_r + (1.0 - self.rho) * rr
            V_k_new = self.rho * V_k + (1.0 - self.rho) * kk
        bc = (1.0 - torch.pow(self.rho, step_1based)).clamp(min=1e-8)
        bc_b = bc.view(-1, 1, 1, 1)
        # P[i,j] = sqrt(V_r[i]*V_k[j]/bc) + ε, broadcast [B,H,d_h,d_h] (paper eq. 25)
        P = (torch.sqrt(
            ((V_r_new.unsqueeze(-1) * V_k_new.unsqueeze(-2)) / bc_b).clamp(min=0.0)
        ) + self.eps).clamp(min=1e-6)
        G = torch.einsum("bhi,bhj->bhij", r, k)
        update_scaled = G / P
        update_scaled = torch.nan_to_num(update_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        # instruct §11: update = (-r)⊗k / P = -G/P; S = αS + β * update_scaled
        update_scaled = -update_scaled
        S_new = alpha * S + beta * update_scaled
        return S_new, V_r_new, V_k_new

    def forward(
        self,
        x,
        S_init=None,
        V_r_init=None,
        V_k_init=None,
        step_init=None,
        valid_mask=None,
        update_mask=None,
        return_S=False,
    ):
        B, L, d = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        q = l2_norm(q)
        k = l2_norm(k)

        q = q.view(B, L, self.num_heads, self.d_head)
        k = k.view(B, L, self.num_heads, self.d_head)
        v = v.view(B, L, self.num_heads, self.d_head)

        alpha_gates = torch.sigmoid(self.alpha_gate(x))
        beta_gates = torch.sigmoid(self.beta_gate(x))
        out_gates = torch.sigmoid(self.out_gate(x))

        vmask = valid_mask if valid_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        mask_for_update = (vmask & update_mask) if update_mask is not None else vmask
        if valid_mask is not None or update_mask is not None:
            mask_H = mask_for_update.view(B, L, 1, 1).to(k.dtype)
            k = k * mask_H
            v = v * mask_H

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if S_init is None:
            S = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device, dtype=x.dtype)
        else:
            S = S_init
        if V_r_init is None:
            V_r = torch.zeros(B, self.num_heads, self.d_head, device=device, dtype=x.dtype)
        else:
            V_r = V_r_init
        if V_k_init is None:
            V_k = torch.zeros(B, self.num_heads, self.d_head, device=device, dtype=x.dtype)
        else:
            V_k = V_k_init

        if step_init is None:
            step_base = torch.zeros(B, device=device, dtype=x.dtype)
        else:
            step_base = step_init.to(device=device, dtype=x.dtype)

        outputs = []
        for t in range(L):
            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            alpha_t = alpha_gates[:, t, :].view(B, 1, 1, 1)
            beta_t = beta_gates[:, t, :].view(B, 1, 1, 1)
            step_1based = step_base + (t + 1)
            S_new, V_r_new, V_k_new = self._damrec_step(k_t, v_t, S, V_r, V_k, alpha_t, beta_t, step_1based)
            mask_t = mask_for_update[:, t].view(B, 1, 1, 1)
            mask_v = mask_for_update[:, t].view(B, 1, 1)
            S = torch.where(mask_t, S_new, S)
            V_r = torch.where(mask_v, V_r_new, V_r)
            V_k = torch.where(mask_v, V_k_new, V_k)
            out_t = torch.einsum("bhij,bhj->bhi", S, q_t)
            outputs.append(out_t)

        out_seq = torch.stack(outputs, dim=2).permute(0, 2, 1, 3).reshape(B, L, d)
        # Paper §3.1 hidden-state regularization: per-token GroupNorm on readout (num_heads groups)
        out_seq = self.out_norm(
            out_seq.reshape(-1, d).unsqueeze(-1)
        ).squeeze(-1).reshape(B, L, d)
        out_seq = out_seq * out_gates + x
        ffn = self.ffn_gate(out_seq) * fn.silu(self.ffn_up(out_seq))
        out_seq = out_seq + self.dropout(self.ffn_down(ffn))

        if return_S:
            return out_seq, S, V_r, V_k
        return out_seq


# Backward-compatible alias (DamRec 现与 instruct.md 一致，非 Adam）
GatedDeltaLayerAdam = GatedDeltaLayerDamRec


class GatedDeltaLayerChunkMomentum(nn.Module):
    r"""MoRec chunk-level layer (消融用). FLA 内部一阶 + Chunk 边界宏观动量。
    - 微观 (Intra-Chunk): FLA chunk_gated_delta_rule，纯 GDN
    - 宏观 (Inter-Chunk): ΔS = S_end - S_start, M_new = μ*M + (1-μ)*ΔS, S_next = S_end + η*M_new
    """

    CHUNK_SIZE = 16  # 必须 ≤ MAX_ITEM_LIST_LENGTH(50)，否则 L≤50 时只有 1 chunk，宏观动量成死代码
    _logged = False

    def __init__(
        self,
        d_model,
        num_heads=4,
        conv_kernel_size=3,
        ffn_ratio=4,
        dropout=0.1,
        momentum=0.9,
        momentum_eta=0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.momentum = momentum
        self.momentum_eta = momentum_eta
        self._use_fla = _FLA_AVAILABLE

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.beta_gate = nn.Linear(d_model, 1)
        self.gamma_gate = nn.Linear(d_model, 1)
        self.out_gate = nn.Linear(d_model, 1)

        ffn_hidden = d_model * ffn_ratio
        self.ffn_gate = nn.Linear(d_model, ffn_hidden)
        self.ffn_up = nn.Linear(d_model, ffn_hidden)
        self.ffn_down = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.constant_(self.beta_gate.bias, 1.0)

    def forward(
        self,
        x,
        S_init=None,
        M_init=None,
        valid_mask=None,
        update_mask=None,
        return_S=False,
    ):
        B, L, d = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        q = l2_norm(q)
        k = l2_norm(k)

        q = q.view(B, L, self.num_heads, self.d_head)
        k = k.view(B, L, self.num_heads, self.d_head)
        v = v.view(B, L, self.num_heads, self.d_head)

        beta_gates = torch.sigmoid(self.beta_gate(x))
        gamma_gates = torch.sigmoid(self.gamma_gate(x))
        out_gates = torch.sigmoid(self.out_gate(x))

        vmask = valid_mask if valid_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        mask_for_update = (vmask & update_mask) if update_mask is not None else vmask

        if not self._use_fla or not x.is_cuda:
            raise RuntimeError("GatedDeltaLayerChunkMomentum requires FLA and CUDA")

        if not GatedDeltaLayerChunkMomentum._logged:
            import logging
            logging.getLogger("recbole").info(
                "[MoRec ChunkMomentum] FLA + chunk-level momentum (μ=%.2f, η=%.2f)"
                % (self.momentum, self.momentum_eta)
            )
            GatedDeltaLayerChunkMomentum._logged = True

        C = self.CHUNK_SIZE
        outputs = []
        S = S_init
        M = M_init
        if S is None:
            S = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device, dtype=x.dtype)
        if M is None:
            M = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device, dtype=x.dtype)

        for start in range(0, L, C):
            end = min(start + C, L)
            q_c = q[:, start:end, :, :]
            k_c = k[:, start:end, :, :]
            v_c = v[:, start:end, :, :]
            mask_c = mask_for_update[:, start:end]

            mask_H = mask_c.view(B, end - start, 1, 1).to(k_c.dtype)
            eps_kv = 1e-8
            k_c = torch.where(mask_H.expand_as(k_c).bool(), k_c, torch.full_like(k_c, eps_kv))
            v_c = torch.where(mask_H.expand_as(v_c).bool(), v_c, torch.full_like(v_c, eps_kv))
            beta_c = torch.where(mask_c.unsqueeze(-1), beta_gates[:, start:end, :], torch.ones_like(beta_gates[:, start:end, :]))

            g_c = torch.log(beta_c.expand(-1, -1, self.num_heads).clamp(min=1e-8))
            beta_fla = gamma_gates[:, start:end, :].expand(-1, -1, self.num_heads)

            S_start = S
            o_c, S_end = chunk_gated_delta_rule(
                q=q_c, k=k_c, v=v_c,
                g=g_c, beta=beta_fla,
                scale=1.0,
                initial_state=S,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
            delta_S = S_end - S_start
            delta_S = torch.nan_to_num(delta_S, nan=0.0, posinf=0.0, neginf=0.0)
            M = self.momentum * M + (1.0 - self.momentum) * delta_S
            S = S_end + self.momentum_eta * M

            outputs.append(o_c)

        out_seq = torch.cat(outputs, dim=1).reshape(B, L, d)
        out_seq = out_seq * out_gates + x
        ffn = self.ffn_gate(out_seq) * fn.silu(self.ffn_up(out_seq))
        out_seq = out_seq + self.dropout(self.ffn_down(ffn))

        if return_S:
            return out_seq, S, M
        return out_seq


class GatedDeltaLayerChunkNesterov(nn.Module):
    r"""NestRec chunk-level layer. FLA 内部一阶 + Chunk 边界 Nesterov 动量。
    - 微观 (Intra-Chunk): FLA chunk_gated_delta_rule，纯 GDN
    - 宏观 (Inter-Chunk): Nesterov 动量，提前看一步的等效更新方向
      M = μ*M + (1-μ)*ΔS; M_nesterov = μ*M + (1-μ)*ΔS; S_next = S_end + η*M_nesterov
    """

    CHUNK_SIZE = 16  # 必须 ≤ MAX_ITEM_LIST_LENGTH(50)，否则宏观 Nesterov 成死代码
    _logged = False

    def __init__(
        self,
        d_model,
        num_heads=4,
        conv_kernel_size=3,
        ffn_ratio=4,
        dropout=0.1,
        momentum=0.9,
        momentum_eta=0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        assert momentum_eta > 0, f"momentum_eta={momentum_eta} 会退化成纯 GDN，必须 > 0"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.momentum = momentum
        self.momentum_eta = momentum_eta
        self._use_fla = _FLA_AVAILABLE

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.beta_gate = nn.Linear(d_model, 1)
        self.gamma_gate = nn.Linear(d_model, 1)
        self.out_gate = nn.Linear(d_model, 1)

        ffn_hidden = d_model * ffn_ratio
        self.ffn_gate = nn.Linear(d_model, ffn_hidden)
        self.ffn_up = nn.Linear(d_model, ffn_hidden)
        self.ffn_down = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.constant_(self.beta_gate.bias, 1.0)

    def forward(
        self,
        x,
        S_init=None,
        M_init=None,
        valid_mask=None,
        update_mask=None,
        return_S=False,
    ):
        B, L, d = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        q = l2_norm(q)
        k = l2_norm(k)

        q = q.view(B, L, self.num_heads, self.d_head)
        k = k.view(B, L, self.num_heads, self.d_head)
        v = v.view(B, L, self.num_heads, self.d_head)

        beta_gates = torch.sigmoid(self.beta_gate(x))
        gamma_gates = torch.sigmoid(self.gamma_gate(x))
        out_gates = torch.sigmoid(self.out_gate(x))

        vmask = valid_mask if valid_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        mask_for_update = (vmask & update_mask) if update_mask is not None else vmask

        if not self._use_fla or not x.is_cuda:
            raise RuntimeError("GatedDeltaLayerChunkNesterov requires FLA and CUDA")

        if not GatedDeltaLayerChunkNesterov._logged:
            import logging
            logging.getLogger("recbole").info(
                "[NestRec ChunkNesterov] FLA + chunk-level Nesterov (μ=%.2f, η=%.2f)"
                % (self.momentum, self.momentum_eta)
            )
            GatedDeltaLayerChunkNesterov._logged = True

        C = self.CHUNK_SIZE
        outputs = []
        S = S_init
        M = M_init
        if S is None:
            S = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device, dtype=x.dtype)
        if M is None:
            M = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device, dtype=x.dtype)

        for start in range(0, L, C):
            end = min(start + C, L)
            q_c = q[:, start:end, :, :]
            k_c = k[:, start:end, :, :]
            v_c = v[:, start:end, :, :]
            mask_c = mask_for_update[:, start:end]

            mask_H = mask_c.view(B, end - start, 1, 1).to(k_c.dtype)
            eps_kv = 1e-8
            k_c = torch.where(mask_H.expand_as(k_c).bool(), k_c, torch.full_like(k_c, eps_kv))
            v_c = torch.where(mask_H.expand_as(v_c).bool(), v_c, torch.full_like(v_c, eps_kv))
            beta_c = torch.where(mask_c.unsqueeze(-1), beta_gates[:, start:end, :], torch.ones_like(beta_gates[:, start:end, :]))

            g_c = torch.log(beta_c.expand(-1, -1, self.num_heads).clamp(min=1e-8))
            beta_fla = gamma_gates[:, start:end, :].expand(-1, -1, self.num_heads)

            S_start = S
            o_c, S_end = chunk_gated_delta_rule(
                q=q_c, k=k_c, v=v_c,
                g=g_c, beta=beta_fla,
                scale=1.0,
                initial_state=S,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
            delta_S = S_end - S_start
            delta_S = torch.nan_to_num(delta_S, nan=0.0, posinf=0.0, neginf=0.0)

            # 1. 计算当前的动量 (EMA)
            M = self.momentum * M + (1.0 - self.momentum) * delta_S
            # 2. Nesterov 核心：提前看一步的等效更新方向
            M_nesterov = self.momentum * M + (1.0 - self.momentum) * delta_S
            # 3. 将 Nesterov 动量注入状态演化
            S = S_end + self.momentum_eta * M_nesterov

            outputs.append(o_c)

        out_seq = torch.cat(outputs, dim=1).reshape(B, L, d)
        out_seq = out_seq * out_gates + x
        ffn = self.ffn_gate(out_seq) * fn.silu(self.ffn_up(out_seq))
        out_seq = out_seq + self.dropout(self.ffn_down(ffn))

        if return_S:
            return out_seq, S, M
        return out_seq


class GatedDeltaLayerChunkDamRec(nn.Module):
    r"""DamRec chunk：instruct.md §7 + 补充「预条件吸收与 FLA」。

    FLA 路径：用秩一向量 ``s_r,s_k`` 将 ``P_{nC}`` 吸收进 ``k,v,S``，在缩放空间调用 ``chunk_gated_delta_rule``，
    输出按方案 A 用 ``1/s_r`` 修正；块间 ``V_r,V_k`` 按 (38)(39) 在 ``no_grad`` 中更新（残差用块首 ``S`` 近似）。
    非 FLA：块内逐步 ``_intra_step``（固定 ``P`` 矩阵）。
    """

    _logged = False
    _debug_fla_logged = False
    _debug_fla_meta_once = False

    def __init__(
        self,
        d_model,
        num_heads=4,
        conv_kernel_size=3,
        ffn_ratio=4,
        dropout=0.1,
        damrec_rho=0.99,
        damrec_eps=1e-8,
        damrec_chunk_size=16,
        use_fla_intrachunk=True,
        damrec_scale_max=2.0,
        damrec_scale_min=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rho = damrec_rho
        self.eps = damrec_eps
        self.chunk_size = damrec_chunk_size
        self.use_fla_intrachunk = use_fla_intrachunk
        self.damrec_scale_max = float(damrec_scale_max) if damrec_scale_max is not None else 2.0
        if damrec_scale_min is not None:
            self.damrec_scale_min = float(damrec_scale_min)
        else:
            self.damrec_scale_min = 1.0 / self.damrec_scale_max
        assert self.damrec_scale_min > 0 and self.damrec_scale_max >= self.damrec_scale_min, (
            f"damrec_scale_min/max invalid: {self.damrec_scale_min}, {self.damrec_scale_max}"
        )

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.alpha_gate = nn.Linear(d_model, 1)
        self.beta_gate = nn.Linear(d_model, 1)
        self.out_gate = nn.Linear(d_model, 1)
        self.out_norm = nn.GroupNorm(num_groups=num_heads, num_channels=d_model)

        ffn_hidden = d_model * ffn_ratio
        self.ffn_gate = nn.Linear(d_model, ffn_hidden)
        self.ffn_up = nn.Linear(d_model, ffn_hidden)
        self.ffn_down = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self._debug_vready_printed = False
        self._debug_vready_post_printed = False

    def _intra_step(self, k, v, S, P_fixed, alpha, beta):
        """Fixed P; S = αS + β * (−(r k^T) ⊘ P)."""
        S_tilde = alpha * S
        Sk = torch.einsum("bhij,bhj->bhi", S_tilde, k)
        r = Sk - v
        G = torch.einsum("bhi,bhj->bhij", r, k)
        upd = -(G / P_fixed)
        upd = torch.nan_to_num(upd, nan=0.0, posinf=0.0, neginf=0.0)
        return alpha * S + beta * upd, r

    def forward(
        self,
        x,
        S_init=None,
        V_r_init=None,
        V_k_init=None,
        step_init=None,
        valid_mask=None,
        update_mask=None,
        return_S=False,
    ):
        B, L, d = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        q = l2_norm(q)
        k = l2_norm(k)

        q = q.view(B, L, self.num_heads, self.d_head)
        k = k.view(B, L, self.num_heads, self.d_head)
        v = v.view(B, L, self.num_heads, self.d_head)

        alpha_gates = torch.sigmoid(self.alpha_gate(x))
        beta_gates = torch.sigmoid(self.beta_gate(x))
        out_gates = torch.sigmoid(self.out_gate(x))

        vmask = valid_mask if valid_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        mask_for_update = (vmask & update_mask) if update_mask is not None else vmask

        use_fla = (
            self.use_fla_intrachunk
            and _FLA_AVAILABLE
            and x.is_cuda
        )

        if not GatedDeltaLayerChunkDamRec._logged:
            import logging
            extra = ""
            if use_fla:
                extra = " | FLA: P→s_r,s_k 吸收 + 缩放空间 GDN + q 方案 A（见 instruct 补充）"
            logging.getLogger("recbole").info(
                "[DamRec Chunk] instruct §7: fixed-P + block V (ρ=%.3f, C=%d, FLA=%s)%s"
                % (self.rho, self.chunk_size, "on" if use_fla else "off", extra)
            )
            GatedDeltaLayerChunkDamRec._logged = True

        C = self.chunk_size
        S = S_init
        V_r = V_r_init
        V_k = V_k_init
        if S is None:
            S = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device, dtype=x.dtype)
        if V_r is None:
            V_r = torch.zeros(B, self.num_heads, self.d_head, device=device, dtype=x.dtype)
        if V_k is None:
            V_k = torch.zeros(B, self.num_heads, self.d_head, device=device, dtype=x.dtype)
        if step_init is None:
            base_step = torch.zeros(B, device=device, dtype=x.dtype)
        else:
            base_step = step_init.to(device=device, dtype=x.dtype)

        outputs = []
        num_chunks = (L + C - 1) // C

        for start in range(0, L, C):
            n = start // C
            end = min(start + C, L)
            clen = end - start
            if (
                _DEBUG_FLA
                and start == 0
                and not GatedDeltaLayerChunkDamRec._debug_fla_meta_once
            ):
                GatedDeltaLayerChunkDamRec._debug_fla_meta_once = True
                print(
                    f"[DamRec FLA DEBUG] first chunk start={start} n={n} L={L} C={C} "
                    f"use_fla={use_fla} x.is_cuda={x.is_cuda} _FLA_AVAILABLE={_FLA_AVAILABLE}"
                )
            q_c = q[:, start:end, :, :]
            k_c = k[:, start:end, :, :]
            v_c = v[:, start:end, :, :]
            mask_c = mask_for_update[:, start:end]

            bias_corr = max(1.0 - (self.rho ** (float(n * C) + 1.0)), 1e-8)
            S_start = S

            if use_fla:
                Vr = V_r.clamp(min=0.0)
                Vk = V_k.clamp(min=0.0)
                # V 不足时关闭吸收；就绪后 s_* 限制在 [damrec_scale_min, damrec_scale_max]（config 可调）
                v_ready = (Vr.abs().max() > 1e-4).item() and (Vk.abs().max() > 1e-4).item()
                smin, smax = self.damrec_scale_min, self.damrec_scale_max
                inv_lo, inv_hi = 1.0 / smax, 1.0 / smin
                den_floor = smin * smin
                if v_ready:
                    sqrt_eps = max(self.eps ** 0.5, 1e-4)
                    s_r = 1.0 / (torch.sqrt(Vr / bias_corr) + sqrt_eps)
                    s_k = 1.0 / (torch.sqrt(Vk / bias_corr) + sqrt_eps)
                    s_r = s_r.clamp(min=smin, max=smax)
                    s_k = s_k.clamp(min=smin, max=smax)
                else:
                    s_r = torch.ones_like(Vr)
                    s_k = torch.ones_like(Vk)
                S_scaled = S_start * s_r.unsqueeze(-1) * s_k.unsqueeze(-2)
                S_scaled = torch.nan_to_num(S_scaled, nan=0.0, posinf=0.0, neginf=0.0).clamp(
                    min=-10.0, max=10.0
                )
                k_scaled = k_c * s_k.unsqueeze(1)
                v_scaled = v_c * s_r.unsqueeze(1)
                sk_u = s_k.unsqueeze(1).clamp(min=smin, max=smax)
                q_adj = q_c / sk_u
                q_adj = torch.nan_to_num(q_adj, nan=0.0, posinf=0.0, neginf=0.0).clamp(
                    min=-20.0, max=20.0
                )

                if (
                    _DEBUG_FLA
                    and v_ready
                    and not self._debug_vready_printed
                ):
                    self._debug_vready_printed = True
                    with torch.no_grad():
                        print(f"[DEBUG FIRST v_ready=True] chunk start={start} n={n}")
                        print(
                            f"  V_r: min={V_r.min().item():.6e} max={V_r.max().item():.6e} "
                            f"absmax={V_r.abs().max().item():.6e}"
                        )
                        print(
                            f"  V_k: min={V_k.min().item():.6e} max={V_k.max().item():.6e} "
                            f"absmax={V_k.abs().max().item():.6e}"
                        )
                        print(f"  bias_corr={bias_corr:.6e}")
                        print(
                            f"  V_r/bias_corr: max={(V_r / bias_corr).abs().max().item():.6e}"
                        )
                        print(
                            f"  sqrt(V_r/bc): max={torch.sqrt((Vr / bias_corr).clamp(min=0)).abs().max().item():.6e}"
                        )
                        print(
                            f"  s_r: min={s_r.min().item():.6e} max={s_r.max().item():.6e}"
                        )
                        print(
                            f"  s_k: min={s_k.min().item():.6e} max={s_k.max().item():.6e}"
                        )
                        print(
                            f"  S_start absmax={S_start.abs().max().item():.6e}"
                        )
                        print(
                            f"  S_scaled absmax={S_scaled.abs().max().item():.6e}"
                        )
                        print(
                            f"  k_scaled absmax={k_scaled.abs().max().item():.6e}"
                        )
                        print(
                            f"  v_scaled absmax={v_scaled.abs().max().item():.6e}"
                        )

                mask_H = mask_c.view(B, clen, 1, 1).to(k_scaled.dtype)
                # 与 GatedDeltaLayer FLA 一致：占位处 k,v 用小常数；β 勿用 1e-8（FLA 对 (I+A)^{-1} 反向会病态致 NaN）
                eps_kv = 1e-4
                k_f = torch.where(mask_H.expand_as(k_scaled).bool(), k_scaled, torch.full_like(k_scaled, eps_kv))
                v_f = torch.where(mask_H.expand_as(v_scaled).bool(), v_scaled, torch.full_like(v_scaled, eps_kv))
                q_in = torch.where(mask_H.expand_as(q_adj).bool(), q_adj, torch.zeros_like(q_adj))
                if (
                    _DEBUG_FLA
                    and start == 0
                    and not GatedDeltaLayerChunkDamRec._debug_fla_logged
                ):
                    with torch.no_grad():
                        print(
                            f"[DamRec FLA DEBUG chunk start={start} n={n}] v_ready={v_ready} "
                            f"Vr_absmax={Vr.abs().max().item():.6e} Vk_absmax={Vk.abs().max().item():.6e}"
                        )
                        print(
                            f"  s_r: min={s_r.min().item():.4f} max={s_r.max().item():.4f} | "
                            f"s_k: min={s_k.min().item():.4f} max={s_k.max().item():.4f}"
                        )
                        print(
                            f"  S_start: min={S_start.min().item():.4f} max={S_start.max().item():.4f} "
                            f"absmax={S_start.abs().max().item():.4f}"
                        )
                        print(
                            f"  S_scaled (after clamp): min={S_scaled.min().item():.4f} max={S_scaled.max().item():.4f} "
                            f"absmax={S_scaled.abs().max().item():.4f}"
                        )
                        print(
                            f"  k_scaled absmax={k_scaled.abs().max().item():.4f} | "
                            f"v_scaled absmax={v_scaled.abs().max().item():.4f} | "
                            f"q_adj absmax={q_adj.abs().max().item():.4f} | "
                            f"q_in absmax={q_in.abs().max().item():.4f}"
                        )
                ag = alpha_gates[:, start:end, :]
                bg = beta_gates[:, start:end, :]
                ag = torch.where(mask_c.unsqueeze(-1), ag, torch.ones_like(ag))
                bg = torch.where(mask_c.unsqueeze(-1), bg, torch.ones_like(bg))
                g_c = torch.log(ag.expand(-1, -1, self.num_heads).clamp(min=1e-4))
                g_c = g_c.clamp(min=-25.0, max=0.0)
                # β 过小会使 FLA 三角求解反向病态；略抬高下限（与占位 β=1 一致）
                beta_fla = bg.expand(-1, -1, self.num_heads).clamp(min=1e-2, max=1.0)
                # FLA 自定义 CUDA 反向在 fp16/bf16 下易 NaN；此处强制 fp32 且关闭 autocast
                _fl_dt = torch.float32
                q_fl = q_in.to(_fl_dt).clamp(min=-20.0, max=20.0)
                k_fl = k_f.to(_fl_dt).clamp(min=-20.0, max=20.0)
                v_fl = v_f.to(_fl_dt).clamp(min=-20.0, max=20.0)
                g_fl = g_c.to(_fl_dt)
                b_fl = beta_fla.to(_fl_dt)
                s0_fl = S_scaled.to(_fl_dt)
                assert not torch.isnan(S_scaled).any(), "S_scaled has NaN before FLA"
                assert not torch.isinf(S_scaled).any(), "S_scaled has Inf before FLA"
                assert not torch.isnan(k_f).any(), "k_f has NaN before FLA"
                assert not torch.isnan(v_f).any(), "v_f has NaN before FLA"
                _cuda_tf32 = None
                _cudnn_tf32 = None
                if device.type == "cuda":
                    _cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
                    torch.backends.cuda.matmul.allow_tf32 = False
                    if hasattr(torch.backends.cudnn, "allow_tf32"):
                        _cudnn_tf32 = torch.backends.cudnn.allow_tf32
                        torch.backends.cudnn.allow_tf32 = False
                try:
                    with torch.amp.autocast(device_type=device.type, enabled=False):
                        o_fla, S_end = chunk_gated_delta_rule(
                            q=q_fl,
                            k=k_fl,
                            v=v_fl,
                            g=g_fl,
                            beta=b_fl,
                            scale=1.0,
                            initial_state=s0_fl,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=False,
                        )
                finally:
                    if _cuda_tf32 is not None:
                        torch.backends.cuda.matmul.allow_tf32 = _cuda_tf32
                    if _cudnn_tf32 is not None:
                        torch.backends.cudnn.allow_tf32 = _cudnn_tf32
                o_fla = o_fla.to(x.dtype)
                S_end = S_end.to(x.dtype)
                assert not torch.isnan(o_fla).any(), "o_fla has NaN after FLA forward"
                assert not torch.isnan(S_end).any(), "S_end has NaN after FLA forward"
                # FLA 文档为 [B,T,H,V]；若实现为 [B,H,T,V] 则先 permute
                if (
                    o_fla.dim() == 4
                    and o_fla.shape[1] == self.num_heads
                    and o_fla.shape[2] == clen
                ):
                    o_fla = o_fla.permute(0, 2, 1, 3).contiguous()
                if (
                    _DEBUG_FLA
                    and self._debug_vready_printed
                    and not self._debug_vready_post_printed
                ):
                    self._debug_vready_post_printed = True
                    with torch.no_grad():
                        den_vr = (s_r.unsqueeze(-1) * s_k.unsqueeze(-2)).clamp(
                            min=1e-12
                        )
                        S_back_vr = S_end / den_vr
                        print(
                            f"  [FIRST v_ready=True after FLA] S_end absmax={S_end.abs().max().item():.6e}"
                        )
                        print(
                            f"  o_fla absmax={o_fla.abs().max().item():.6e}"
                        )
                        print(
                            f"  den (min=1e-12): min={den_vr.min().item():.6e} max={den_vr.max().item():.6e}"
                        )
                        print(
                            f"  S_back absmax={S_back_vr.abs().max().item():.6e}"
                        )
                        o_check = o_fla.reshape(B, clen, self.num_heads, self.d_head)
                        o_check = o_check * (1.0 / s_r).unsqueeze(1).clamp(
                            min=inv_lo, max=inv_hi
                        )
                        print(
                            f"  o after 1/s_r absmax={o_check.abs().max().item():.6e}"
                        )
                if (
                    _DEBUG_FLA
                    and start == 0
                    and not GatedDeltaLayerChunkDamRec._debug_fla_logged
                ):
                    with torch.no_grad():
                        print(
                            f"  [after FLA] o_fla absmax={o_fla.abs().max().item():.4f} | "
                            f"S_end absmax={S_end.abs().max().item():.4f}"
                        )
                # den = s_r s_k，下界 smin² 与 s_* 的 clamp 一致（config 可调）
                den = (s_r.unsqueeze(-1) * s_k.unsqueeze(-2)).clamp(min=den_floor)
                if (
                    _DEBUG_FLA
                    and start == 0
                    and not GatedDeltaLayerChunkDamRec._debug_fla_logged
                ):
                    with torch.no_grad():
                        S_back = S_end / den
                        print(
                            f"  S_back (S_end/den inverse) absmax={S_back.abs().max().item():.4f} "
                            f"den absmax={den.abs().max().item():.4f}"
                        )
                S = torch.nan_to_num(S_end / den, nan=0.0, posinf=0.0, neginf=0.0)
                o_nd = o_fla.reshape(B, clen, self.num_heads, self.d_head)
                o_nd = o_nd * (1.0 / s_r).unsqueeze(1).clamp(min=inv_lo, max=inv_hi)
                if (
                    _DEBUG_FLA
                    and start == 0
                    and not GatedDeltaLayerChunkDamRec._debug_fla_logged
                ):
                    with torch.no_grad():
                        print(
                            f"  o_nd after ×(1/s_r) absmax={o_nd.abs().max().item():.4f} | "
                            f"S (nan_to_num) absmax={S.abs().max().item():.4f}"
                        )
                    GatedDeltaLayerChunkDamRec._debug_fla_logged = True
                o_fla = torch.nan_to_num(o_nd.reshape(B, clen, d), nan=0.0, posinf=0.0, neginf=0.0)
                for t in range(clen):
                    outputs.append(o_fla[:, t, :])

                with torch.no_grad():
                    r_list = []
                    for t in range(clen):
                        k_t = k_c[:, t, :, :]
                        v_t = v_c[:, t, :, :]
                        alpha_t = alpha_gates[:, start + t, :].view(B, 1, 1)
                        Sk = torch.einsum("bhij,bhj->bhi", S_start.detach(), k_t)
                        r_t = alpha_t * Sk - v_t
                        m = mask_c[:, t].view(B, 1, 1)
                        r_list.append(torch.where(m, r_t, torch.zeros_like(r_t)))
                    k_list = [
                        torch.where(
                            mask_c[:, t].view(B, 1, 1),
                            k_c[:, t, :, :],
                            torch.zeros_like(k_c[:, t, :, :]),
                        )
                        for t in range(clen)
                    ]
            else:
                # P[i,j] = sqrt(V_r[i]*V_k[j]/bc) + ε (paper eq. 25)
                P = (torch.sqrt(
                    ((V_r.unsqueeze(-1) * V_k.unsqueeze(-2)) / bias_corr).clamp(min=0.0)
                ) + self.eps).clamp(min=1e-6)
                r_list = []
                k_list = []
                for t in range(clen):
                    q_t = q_c[:, t, :, :]
                    k_t = k_c[:, t, :, :]
                    v_t = v_c[:, t, :, :]
                    alpha_t = alpha_gates[:, start + t, :].view(B, 1, 1, 1)
                    beta_t = beta_gates[:, start + t, :].view(B, 1, 1, 1)
                    m = mask_c[:, t].view(B, 1, 1, 1)
                    S_new, r_t = self._intra_step(k_t, v_t, S, P, alpha_t, beta_t)
                    S = torch.where(m, S_new, S)
                    r_list.append(torch.where(m, r_t, torch.zeros_like(r_t)))
                    k_list.append(torch.where(m, k_t, torch.zeros_like(k_t)))
                    out_t = torch.einsum("bhij,bhj->bhi", S, q_t)
                    outputs.append(out_t.reshape(B, d))

            # Block V update (38)(39); τ 权重 ρ^{clen-1-j}
            r_stack = torch.stack(r_list, dim=1).detach()
            k_stack = torch.stack(k_list, dim=1).detach()
            idx = torch.arange(clen, device=device, dtype=x.dtype)
            w = self.rho ** (clen - 1 - idx)
            w = w.view(1, clen, 1, 1)
            weighted_rr = ((r_stack ** 2) * w).sum(dim=1)
            weighted_kk = ((k_stack ** 2) * w).sum(dim=1)
            with torch.no_grad():
                V_r = (self.rho ** clen) * V_r + (1.0 - self.rho) * weighted_rr
                V_k = (self.rho ** clen) * V_k + (1.0 - self.rho) * weighted_kk

        final_step = base_step + float(num_chunks)
        out_seq = torch.stack(outputs, dim=1).reshape(B, L, d)
        # Paper §3.1 hidden-state regularization: per-token GroupNorm on readout (num_heads groups)
        out_seq = self.out_norm(
            out_seq.reshape(-1, d).unsqueeze(-1)
        ).squeeze(-1).reshape(B, L, d)
        out_seq = out_seq * out_gates + x
        ffn = self.ffn_gate(out_seq) * fn.silu(self.ffn_up(out_seq))
        out_seq = out_seq + self.dropout(self.ffn_down(ffn))

        if return_S:
            return out_seq, S, V_r, V_k, final_step
        return out_seq


GatedDeltaLayerChunkAdam = GatedDeltaLayerChunkDamRec


class GatedDeltaLayerChunkFroAdam(nn.Module):
    r"""FroRec chunk-level layer. F-Adam: 二阶矩 V 降维为标量 [B,H,1,1]，保留 M 的秩一外积方向。
    - 微观 (Intra-Chunk): FLA chunk_gated_delta_rule，纯 GDN
    - 宏观 (Inter-Chunk): M = β1*M + (1-β1)*ΔS; V = β2*V + (1-β2)*mean(ΔS²); S = S_end + η*M/(√V+ε)
    V 为标量广播，不改变 M 的方向，几何结构 100% 安全。
    """

    CHUNK_SIZE = 16  # 必须 ≤ MAX_ITEM_LIST_LENGTH(50)，否则宏观 F-Adam 成死代码
    _logged = False

    def __init__(
        self,
        d_model,
        num_heads=4,
        conv_kernel_size=3,
        ffn_ratio=4,
        dropout=0.1,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        adam_eta=0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.beta1 = adam_beta1
        self.beta2 = adam_beta2
        self.eps = adam_eps
        self.eta = adam_eta
        self._use_fla = _FLA_AVAILABLE

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(d_model, conv_kernel_size)
        self.beta_gate = nn.Linear(d_model, 1)
        self.gamma_gate = nn.Linear(d_model, 1)
        self.out_gate = nn.Linear(d_model, 1)

        ffn_hidden = d_model * ffn_ratio
        self.ffn_gate = nn.Linear(d_model, ffn_hidden)
        self.ffn_up = nn.Linear(d_model, ffn_hidden)
        self.ffn_down = nn.Linear(ffn_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        nn.init.constant_(self.beta_gate.bias, 1.0)

    def forward(
        self,
        x,
        S_init=None,
        M_init=None,
        V_init=None,
        step_init=None,
        valid_mask=None,
        update_mask=None,
        return_S=False,
    ):
        B, L, d = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        q = l2_norm(q)
        k = l2_norm(k)

        q = q.view(B, L, self.num_heads, self.d_head)
        k = k.view(B, L, self.num_heads, self.d_head)
        v = v.view(B, L, self.num_heads, self.d_head)

        beta_gates = torch.sigmoid(self.beta_gate(x))
        gamma_gates = torch.sigmoid(self.gamma_gate(x))
        out_gates = torch.sigmoid(self.out_gate(x))

        vmask = valid_mask if valid_mask is not None else torch.ones(B, L, dtype=torch.bool, device=device)
        mask_for_update = (vmask & update_mask) if update_mask is not None else vmask

        if not self._use_fla or not x.is_cuda:
            raise RuntimeError("GatedDeltaLayerChunkFroAdam requires FLA and CUDA")

        if not GatedDeltaLayerChunkFroAdam._logged:
            import logging
            logging.getLogger("recbole").info(
                "[FroRec ChunkFroAdam] FLA + F-Adam (β1=%.2f, β2=%.3f, η=%.2f)"
                % (self.beta1, self.beta2, self.eta)
            )
            GatedDeltaLayerChunkFroAdam._logged = True

        C = self.CHUNK_SIZE
        outputs = []
        S = S_init
        M = M_init
        V = V_init
        if S is None:
            S = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device, dtype=x.dtype)
        if M is None:
            M = torch.zeros(B, self.num_heads, self.d_head, self.d_head, device=device, dtype=x.dtype)
        if V is None:
            V = torch.zeros(B, self.num_heads, 1, 1, device=device, dtype=x.dtype)
        if step_init is None:
            base_step = torch.zeros(B, device=device, dtype=torch.float32)
        else:
            base_step = step_init

        for chunk_idx, start in enumerate(range(0, L, C)):
            chunk_step = base_step + (chunk_idx + 1)
            end = min(start + C, L)
            q_c = q[:, start:end, :, :]
            k_c = k[:, start:end, :, :]
            v_c = v[:, start:end, :, :]
            mask_c = mask_for_update[:, start:end]

            mask_H = mask_c.view(B, end - start, 1, 1).to(k_c.dtype)
            eps_kv = 1e-8
            k_c = torch.where(mask_H.expand_as(k_c).bool(), k_c, torch.full_like(k_c, eps_kv))
            v_c = torch.where(mask_H.expand_as(v_c).bool(), v_c, torch.full_like(v_c, eps_kv))
            beta_c = torch.where(mask_c.unsqueeze(-1), beta_gates[:, start:end, :], torch.ones_like(beta_gates[:, start:end, :]))

            g_c = torch.log(beta_c.expand(-1, -1, self.num_heads).clamp(min=1e-8))
            beta_fla = gamma_gates[:, start:end, :].expand(-1, -1, self.num_heads)

            S_start = S
            o_c, S_end = chunk_gated_delta_rule(
                q=q_c, k=k_c, v=v_c,
                g=g_c, beta=beta_fla,
                scale=1.0,
                initial_state=S,
                output_final_state=True,
                use_qk_l2norm_in_kernel=False,
            )
            delta_S = S_end - S_start
            delta_S = torch.nan_to_num(delta_S, nan=0.0, posinf=0.0, neginf=0.0)

            # 1. 一阶动量 M 保持矩阵形式 [B, H, d_h, d_h]
            M = self.beta1 * M + (1.0 - self.beta1) * delta_S

            # 2. [F-Adam] 二阶矩 V 降维为标量 [B, H, 1, 1]
            delta_sq_mean = (delta_S ** 2).mean(dim=(-1, -2), keepdim=True)
            V = self.beta2 * V + (1.0 - self.beta2) * delta_sq_mean
            V = V.clamp(min=0.0)

            # 3. 偏差校正
            step_view = chunk_step.view(B, 1, 1, 1)
            bc1 = (1.0 - torch.pow(self.beta1, step_view)).clamp(min=1e-8)
            bc2 = (1.0 - torch.pow(self.beta2, step_view)).clamp(min=1e-8)
            M_hat = M / bc1
            V_hat = V / bc2

            # 4. 标量除法，几何结构 100% 安全
            denom = (torch.sqrt(V_hat) + self.eps).clamp(min=1e-6)
            S = S_end + self.eta * (M_hat / denom)

            outputs.append(o_c)

        final_step = base_step + len(list(range(0, L, C)))
        out_seq = torch.cat(outputs, dim=1).reshape(B, L, d)
        out_seq = out_seq * out_gates + x
        ffn = self.ffn_gate(out_seq) * fn.silu(self.ffn_up(out_seq))
        out_seq = out_seq + self.dropout(self.ffn_down(ffn))

        if return_S:
            return out_seq, S, M, V, final_step
        return out_seq
