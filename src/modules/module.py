#pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from modules.sparse_moe import MoE
from modules.config import MoEConfig

import sys
import pdb



class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        # embed_dim can be decomposed into num_heads x head_dim?
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None
        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # src_len = bsz * num_heads
        src_len = k.size(1)
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                # print(attn_weights.shape)
                # print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # embed_dim = self.num_heads * self.head_dim
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class TransformerCrossEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerCrossEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, args, embed_dim, num_heads, layers, device, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, q_seq_len_1=None, q_seq_len_2=None, num_modalities=3):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device=device
        self.num_modalities=3

        # self.q_seq_len_1=q_seq_len_1 
        # # seq_len_1 is tt_max, the longest sequence length, which is 48 for 48 hrs
        # self.q_seq_len_2=q_seq_len_2
        # self.num_modalities = num_modalities
        # # self.intermediate=intermediate
        # self.embed_positions_q_1=nn.Embedding(self.q_seq_len_1, embed_dim, padding_idx=0)
        # nn.init.normal_(self.embed_positions_q_1.weight, std=0.02)

        # if self.q_seq_len_2 != None:
        #     self.embed_positions_q_2=nn.Embedding(self.q_seq_len_2,embed_dim,padding_idx=0)
        #     nn.init.normal_(self.embed_positions_q_2.weight, std=0.02)
        #     self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1, self.embed_positions_q_2])
        # else:
        #     self.embed_positions_q=nn.ModuleList([self.embed_positions_q_1 for _ in range(num_modalities)])

        self.attn_mask = attn_mask
        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerCrossEncoderLayer(args,
                                                    embed_dim,
                                                    device=device,
                                                    num_heads=num_heads,
                                                    attn_dropout=attn_dropout,
                                                    relu_dropout=relu_dropout,
                                                    res_dropout=res_dropout,
                                                    attn_mask=attn_mask,
                                                    num_modalities=num_modalities)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])

    def forward(self, x_in_list, modality):
        """
        Args:
            x_in_list (list of FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the list of last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`

        """

        # x_in_list contains ts and clinical notes tensors
        x_list = x_in_list
        lengths, positions = [], []
        for i in range(self.num_modalities):
            lengths.append(x_list[i].size(0))
        x_list = [self.embed_scale * x_in for x_in in x_in_list]
        # if self.q_seq_len_1 is not None:
        #     for length in lengths:
        #         positions.append(torch.tensor(torch.arange(length),dtype=torch.long).to(self.device))
        #     x_list = [l(position_x).unsqueeze(0).transpose(0,1) + x for l, x, position_x in zip(self.embed_positions_q, x_list, positions)]
        #       # Add positional embedding
        #     x_list = [F.dropout(x, p=self.dropout, training=self.training) for x in x_list]
        # encoder layers
        layer_id=1
        layer_routing_log = []
        for layer in self.layers:
            x_list, routing_log = layer(x_list, modality, layer_id)[:2] #proj_x_txt, proj_x_ts
            # routing_log = layer(x_list, modality, layer_id)[1]
            # len(x_list) = 2
            # x_list[0].shape = torch.Size([48, 1, 128])
            layer_id+=1
            layer_routing_log.extend(routing_log)
            if x_list is None:
                return None

        if self.normalize:
            x_list=[l(x) for l, x in zip(self.layer_norm, x_list)]
        return x_list, layer_routing_log


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, hp, embed_dim, device, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1, 
                 attn_mask=False, num_modalities=3):
        super().__init__()
        self.args = hp
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.pre_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        self.self_attns = nn.ModuleList([MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        ) for _ in range(num_modalities)])

        self.post_self_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])
        self.pre_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        # Depthwise Separable Convolution setup
        self.conv1_depth = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, groups=1)
            for _ in range(num_modalities)
        ])
        self.conv1_point = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)  # Keep same feature size
            for _ in range(num_modalities)
        ])

        # self.conv2_depth = nn.ModuleList([
        #     nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1, groups=1)
        #     for _ in range(num_modalities)
        # ])
        # self.conv2_point = nn.ModuleList([
        #     nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)  # Keep same feature size
        #     for _ in range(num_modalities)
        # ])

        # Adaptive Average Pooling to ensure output remains (batch, 128)
        self.global_avg_pool = nn.ModuleList([
            nn.AdaptiveAvgPool1d(128) for _ in range(num_modalities)
        ])

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=res_dropout)

        # self.cross_attn_1 = MultiheadAttention(
        #     embed_dim=self.embed_dim,
        #     num_heads=self.num_heads,
        #     attn_dropout=attn_dropout
        # )

        # self.cross_attn_2 = MultiheadAttention(
        #     embed_dim=self.embed_dim,
        #     num_heads=self.num_heads,
        #     attn_dropout=attn_dropout
        # )

        self.post_encoder_attn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])

        self.attn_mask = attn_mask

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])
        self.fc1 = nn.ModuleList([nn.Linear(self.embed_dim, 4 * self.embed_dim) for _ in range(num_modalities)])  # The "Add & Norm" part in the paper
        self.fc2 = nn.ModuleList([nn.Linear(4 * self.embed_dim, self.embed_dim) for _ in range(num_modalities)])
        self.pre_ffn_layer_norm = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(num_modalities)])
        
        moe_config = MoEConfig(            
            num_experts=16,
            moe_input_size=128*3,
            moe_hidden_size=768,
            moe_output_size=768 + hp.d_vout + hp.d_aout,
            top_k=2,
            router_type='permod',
            num_modalities=3,
            gating='laplace')
        self.moe = MoE(moe_config)
        self.moe = self.moe.to(device)
        
    def forward(self, x_list, modality, layer_id):
        """
        Args:
            x (List of Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
        Returns:
            list of encoded output of shape `(batch, src_len, embed_dim)`
        """
        #TODO: figure out how many layers of this is required?
        residual = x_list
        # seq_len, bs = x_list[0].shape[0], x_list[0].shape[1]
        bs, seq_len = x_list[0].shape[0], x_list[0].shape[1]

        # Reshape for Conv1D (batch, 1, 128)
        x_list = [x.unsqueeze(1) for x in x_list]

        # Apply Depthwise Separable Conv1D for each modality
        for i in range(len(x_list)):
            x = self.conv1_depth[i](x_list[i])
            x = self.conv1_point[i](x)
            x = self.relu(x)
            x = self.dropout(x)

            # x = self.conv2_depth[i](x)
            # x = self.conv2_point[i](x)
            # x = self.relu(x)
            # x = self.dropout(x)

            # Apply Global Average Pooling to ensure fixed output shape
            x = self.global_avg_pool[i](x)

            x_list[i] = x.squeeze(1)  # Restore to shape (batch, 128)

        # Residual Connection
        x_list = [res + x for res, x in zip(residual, x_list)]
        x_list = [l(x) for l, x in zip(self.pre_encoder_attn_layer_norm, x_list)]
        # if self.args.cross_method in ["moe", "hme"]:
        x_mod_in = [torch.reshape(x, (bs, -1)) for x in x_list]
        embd_len_list = [0] + list(np.cumsum([x.shape[1] for x in x_mod_in]))
        embeddings = torch.concat(x_mod_in, dim=1)
        if torch.isnan(embeddings).any():
            return None
        # just replace this with hierarchical moe
        moe_out, balance_loss, routing_info = self.moe(x_mod_in, modalities=modality)
        x_mod_out = [moe_out[:, embd_len_list[i]:embd_len_list[i + 1]] for i in range(len(embd_len_list) - 1)]
        # moe_out.shape = [32, 800], x_mod_out.shape = [3 of [32, 128]]
        x_allmod_output = [torch.reshape(x, (bs, -1)) for x in x_mod_out]
        # x_allmod_output.shape = [3 of [32, 128]]
        # Should reshape be shape of [3, 32, 128]?
        moe_output = [F.dropout(x, p=self.res_dropout, training=self.training) for x in x_allmod_output]
        # moe_output.shape = [3 of [32, 128]], should also be [3, 32, 128]?
        x_list = [r + x for r, x in zip(residual, moe_output)]

        routing_log = []
        for idx, mod in routing_info:
            routing_log.append([layer_id, mod, idx.detach().cpu().tolist()])

        # pay attention to how the text and patch embeddings are concated in LIMOE
        # LIMOE just concat? add modality type embeddings
        # sparse attention combined with dense attention
        # if self.args.cross_method == "self_cross":
        #     assert self.num_modalities == 2, 'Input modality should be 2 if using cross attention method.'
        #     x_txt, x_ts = x_list #proj_x_txt, proj_x_ts
        #     x_ts_to_txt, _ = self.cross_attn_1(query=x_txt, key=x_ts, value=x_ts)
        #     x_txt_to_ts, _ = self.cross_attn_2(query=x_ts, key=x_txt, value=x_txt)

        #     x_ts_to_txt = F.dropout(x_ts_to_txt, p=self.res_dropout, training=self.training)
        #     x_txt_to_ts = F.dropout(x_txt_to_ts, p=self.res_dropout, training=self.training)
        #     x_list = [r+ x for r, x in zip(residual, (x_ts_to_txt, x_txt_to_ts))]

        # FNN
        residual = x_list
        x_list = [l(x) for l, x in zip(self.pre_ffn_layer_norm, x_list)]
        x_list = [F.relu(l(x)) for l, x in zip(self.fc1, x_list)]
        x_list = [F.dropout(x, p=self.relu_dropout, training=self.training) for x in x_list]
        x_list = [l(x) for l, x in zip(self.fc2, x_list)]
        x_list = [F.dropout(x, p=self.res_dropout, training=self.training) for x in x_list]
        x_list = [r + x  for r, x in zip(residual, x_list) ]

        return x_list, routing_log


