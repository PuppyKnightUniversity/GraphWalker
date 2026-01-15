# NOTE (EAG-RL): change by EAG-RL
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from utils.numerical.sequence_handler import get_last_visit, normalize_mask


class FinalAttentionQKV(nn.Module):
    def __init__(
        self,
        attention_input_dim: int,
        attention_hidden_dim: int,
        attention_type: str = "add",
        dropout: float = 0.5,
    ):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(
            torch.zeros(
                1,
            )
        )
        self.b_out = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(
            torch.zeros(
                1,
            )
        )

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # NOTE (EAG-RL): x.size() = [Batch, lab_dim + 1, hidden_dim]
        (
            batch_size,
            time_step,
            lab_dim,
        ) = x.size()  # batch_size * lab_dim + 1 * hidden_dim(i)
        input_q = self.W_q(x[:, -1, :])  # b h
        input_k = self.W_k(x)  # b t h
        input_v = self.W_v(x)  # b t h

        if self.attention_type == "add":  # B*T*I  @ H*I

            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "mul":
            # NOTE (EAG-RL): this is the real choice
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1))  # B*h 1
            e = torch.matmul(input_k, q)  # b t 1
            e = e.squeeze(-1)  # Remove last dimension only: b t
            # Ensure e maintains correct shape even when batch_size=1
            if e.dim() == 1:
                e = e.unsqueeze(0)  # [time_step] -> [1, time_step]

        elif self.attention_type == "concat":
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        else:
            raise ValueError("Unknown attention type: {}, please use add, mul, concat".format(self.attention_type))

        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # B*I

        return v, a


class PositionwiseFeedForward(nn.Module):  # new added
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x)))), None


class PositionalEncoding(nn.Module):  # new added / not use anymore
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pos = self.pe[:, : x.size(1)].clone().requires_grad_(False)
        x = x + pos
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, self.d_k * self.h) for _ in range(3)])
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)  # b h t d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # b h t t
        if mask is not None:  # 1 1 t t
            scores = scores.masked_fill(mask == 0, -1e9)  # b h t t 下三角
        p_attn = torch.softmax(scores, dim=-1)  # b h t t
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  # b h t v (d_k)

    def cov(self, m, y=None):
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]

        # Avoid division by zero when batch size is 1
        denominator = x.size(1) - 1
        if denominator <= 0:
            # Return zero covariance matrix when we can't compute proper covariance
            return torch.zeros(x.size(0), x.size(0), device=x.device, dtype=x.dtype)

        cov = (1 / denominator) * x.mm(x.t())
        return cov

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 1 1 t t

        nbatches = query.size(0)  # b
        feature_dim = query.size(1)  # i+1

        # x size -> # batch_size * d_input * hidden_dim

        # d_model => h * d_k
        query, key, value = [
            layer(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for layer, x in zip(self.linears, (query, key, value))
        ]  # b num_head d_input d_k

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)  # b num_head d_input d_v (d_k)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # batch_size * d_input * hidden_dim

        # DeCov
        DeCov_contexts = x.transpose(0, 1).transpose(1, 2)  # I+1 H B
        Covs = self.cov(DeCov_contexts[0, :, :])
        DeCov_loss = 0.5 * (torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2)
        for i in range(feature_dim - 1):
            Covs = self.cov(DeCov_contexts[i + 1, :, :])
            DeCov_loss += 0.5 * (torch.norm(Covs, p="fro") ** 2 - torch.norm(torch.diag(Covs)) ** 2)

        return self.final_linear(x), DeCov_loss


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]), returned_value[1]


class SingleAttention(nn.Module):
    def __init__(
        self,
        attention_input_dim,
        attention_hidden_dim,
        attention_type="add",
        time_aware=False,
    ):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.time_aware = time_aware

        # batch_time = torch.arange(0, batch_mask.size()[1], dtype=torch.float32).reshape(1, batch_mask.size()[1], 1)
        # batch_time = batch_time.repeat(batch_mask.size()[0], 1, 1)

        if attention_type == "add":
            if self.time_aware:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.bh = nn.Parameter(
                torch.zeros(
                    attention_hidden_dim,
                )
            )
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wd, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "mul":
            self.Wa = nn.Parameter(torch.randn(attention_input_dim, attention_input_dim))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == "concat":
            if self.time_aware:
                self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim + 1, attention_hidden_dim))
            else:
                self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(
                torch.zeros(
                    1,
                )
            )

            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == "new":
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))

            self.rate = nn.Parameter(torch.zeros(1) + 0.8)
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        else:
            raise RuntimeError("Wrong attention type. Please use 'add', 'mul', 'concat' or 'new'.")

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x, mask, device):

        (
            batch_size,
            time_step,
            lab_dim,
        ) = x.size()  # batch_size * time_step * hidden_dim(i)

        time_decays = (
            torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device=device)
        )  # 1*t*1
        b_time_decays = time_decays.repeat(batch_size, 1, 1) + 1  # b t 1

        if self.attention_type == "add":  # B*T*I  @ H*I
            last_visit = get_last_visit(x, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            if self.time_aware is True:
                k = torch.matmul(x, self.Wx)  # b t h
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)  # b t h
            else:
                k = torch.matmul(x, self.Wx)  # b t h
            h = q + k + self.bh  # b t h
            if self.time_aware:
                h += time_hidden
            h = self.tanh(h)  # B*T*H
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        elif self.attention_type == "mul":
            last_visit = get_last_visit(x, mask)
            e = torch.matmul(last_visit, self.Wa)  # b i
            e = torch.matmul(e.unsqueeze(1), x.permute(0, 2, 1)).reshape(batch_size, time_step) + self.ba  # b t
        elif self.attention_type == "concat":
            last_visit = get_last_visit(x, mask)
            q = last_visit.unsqueeze(1).repeat(1, time_step, 1)  # b t i
            k = x
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            if self.time_aware:
                c = torch.cat((c, b_time_decays), dim=-1)  # B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == "new":
            last_visit = get_last_visit(x, mask)
            q = torch.matmul(last_visit, self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            k = torch.matmul(x, self.Wx)  # b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).reshape(batch_size, time_step)  # b t
            denominator = self.sigmoid(self.rate) * (
                torch.log(2.72 + (1 - self.sigmoid(dot_product))) * (b_time_decays.reshape(batch_size, time_step))
            )
            e = self.relu(self.sigmoid(dot_product) / (denominator))  # b * t
        else:
            raise ValueError("Wrong attention type. Plase use 'add', 'mul', 'concat' or 'new'.")

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        a = self.softmax(e)  # B*T
        v = torch.matmul(a.unsqueeze(1), x).reshape(batch_size, lab_dim)  # B*I

        return v, a


class ConCareLayer(nn.Module):
    """ConCare layer.

    Paper: Liantao Ma et al. Concare: Personalized clinical feature embedding via capturing the healthcare context. AAAI 2020.

    This layer is used in the ConCare model. But it can also be used as a
    standalone layer.

    Args:
        lab_dim: dynamic feature size.
        demo_dim: static feature size, if 0, then no static feature is used.
        hidden_dim: hidden dimension of the channel-wise GRU, default 128.
        transformer_hidden: hidden dimension of the transformer, default 128.
        num_head: number of heads in the transformer, default 4.
        pe_hidden: hidden dimension of the positional encoding, default 64.
        dropout: dropout rate, default 0.5.

    Examples:
        >>> from pyhealth.models import ConCare
        >>> x = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = ConCare(64)
        >>> c, _ = layer(x)
        >>> c.shape
        torch.Size([3, 128])
    """

    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: int = 0.5,
    ):
        super(ConCareLayer, self).__init__()

        # hyperparameters
        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim  # d_model
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        # self.output_dim = output_dim
        self.dropout = dropout
        self.demo_dim = demo_dim

        # layers
        self.PositionalEncoding = PositionalEncoding(self.transformer_hidden, dropout=0, max_len=400)

        self.GRUs = nn.ModuleList([nn.GRU(1, self.hidden_dim, batch_first=True) for _ in range(self.lab_dim)])
        self.LastStepAttentions = nn.ModuleList(
            [
                SingleAttention(
                    self.hidden_dim,
                    8,
                    attention_type="new",
                    time_aware=True,
                )
                for _ in range(self.lab_dim)
            ]
        )

        self.FinalAttentionQKV = FinalAttentionQKV(
            self.hidden_dim,
            self.hidden_dim,
            attention_type="mul",
            dropout=self.dropout,
        )

        self.MultiHeadedAttention = MultiHeadedAttention(self.num_head, self.transformer_hidden, dropout=self.dropout)
        self.SublayerConnection = SublayerConnection(self.transformer_hidden, dropout=self.dropout)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.transformer_hidden, self.pe_hidden, dropout=0.1)

        if self.demo_dim > 0:
            self.demo_proj_main = nn.Linear(self.demo_dim, self.hidden_dim)

        self.dropout = nn.Dropout(p=self.dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.tensor, ...]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, lab_dim].
            static: a tensor of shape [batch size, demo_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.
            return_attention: if True, return attention scores along with the output.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
            decov_loss: DeCov regularization loss.
            attention_scores (optional): dictionary containing attention scores if return_attention=True.
        """
        # rnn will only apply dropout between layers
        # NOTE (EAG-RL): demo dim to hidden_dim
        if self.demo_dim > 0:
            demo_main = self.tanh(self.demo_proj_main(static)).unsqueeze(1)  # b hidden_dim

        batch_size = x.size(0)
        feature_dim = x.size(2)

        if self.transformer_hidden % self.num_head != 0:
            raise ValueError("transformer_hidden must be divisible by num_head")

        # Normalize mask
        if mask is not None:
            mask = normalize_mask(mask, x.shape[1])

        # Store attention scores if requested
        attention_scores = {} if return_attention else None
        feature_time_attentions = [] if return_attention else None

        # forward
        GRU_embeded_input = self.GRUs[0](
            x[:, :, 0].unsqueeze(-1).to(device=x.device),
            torch.zeros(batch_size, self.hidden_dim).to(device=x.device).unsqueeze(0),
        )[
            0
        ]  # b t h

        feature_output, time_attention = self.LastStepAttentions[0](GRU_embeded_input, mask, x.device)
        Attention_embeded_input = feature_output.unsqueeze(1)  # b 1 h

        if return_attention:
            feature_time_attentions.append(time_attention)  # [batch, time_step]

        for i in range(feature_dim - 1):
            embeded_input = self.GRUs[i + 1](
                x[:, :, i + 1].unsqueeze(-1),
                torch.zeros(batch_size, self.hidden_dim).to(device=x.device).unsqueeze(0),
            )[
                0
            ]  # b t h
            feature_output, time_attention = self.LastStepAttentions[i + 1](embeded_input, mask, x.device)
            embeded_input = feature_output.unsqueeze(1)  # b 1 h
            Attention_embeded_input = torch.cat((Attention_embeded_input, embeded_input), 1)  # b i h

            if return_attention:
                feature_time_attentions.append(time_attention)  # [batch, time_step]

        # NOTE (EAG-RL) concat demo to attention_embeded_input
        if self.demo_dim > 0:
            Attention_embeded_input = torch.cat((Attention_embeded_input, demo_main), 1)  # b i+1 h
        posi_input = self.dropout(Attention_embeded_input)  # batch_size * d_input+1 * hidden_dim

        contexts = self.SublayerConnection(
            posi_input,
            lambda _: self.MultiHeadedAttention(posi_input, posi_input, posi_input, None),
        )  # # batch_size * d_input * hidden_dim

        DeCov_loss = contexts[1]
        contexts = contexts[0]

        # Store multi-head attention scores if requested
        if return_attention:
            attention_scores["multihead_attention"] = (
                self.MultiHeadedAttention.attn
            )  # [batch, num_head, feature_dim, feature_dim]
            attention_scores["feature_time_attentions"] = torch.stack(
                feature_time_attentions, dim=1
            )  # [batch, feature_dim, time_step]

        contexts = self.SublayerConnection(contexts, lambda _: self.PositionwiseFeedForward(contexts))[0]

        weighted_contexts, final_attention = self.FinalAttentionQKV(contexts)
        # NOTE (EAG-RL): weighted_contexts.size() = [hidden_dim]
        # NOTE (EAG-RL): final_attention.size() = [batch, lab_dim + 1]

        if return_attention:
            attention_scores["final_attention"] = final_attention  # [batch, feature_dim (+1 if demo)]

        if return_attention:
            return weighted_contexts, DeCov_loss, attention_scores
        else:
            return weighted_contexts, DeCov_loss


class ConCare(nn.Module):
    def __init__(
        self,
        lab_dim: int,
        demo_dim: int = 0,
        hidden_dim: int = 128,
        num_head: int = 4,
        pe_hidden: int = 64,
        dropout: int = 0.0,
        **kwargs,
    ):
        super(ConCare, self).__init__()

        # hyperparameters
        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim  # d_model
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        # self.output_dim = output_dim
        self.dropout = nn.Dropout(p=dropout)
        self.demo_dim = demo_dim
        self.concare_layer = ConCareLayer(
            lab_dim=lab_dim, demo_dim=demo_dim, hidden_dim=hidden_dim, num_head=num_head, pe_hidden=pe_hidden, dropout=dropout
        )

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.tensor, ...]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, lab_dim].
            static: a tensor of shape [batch size, demo_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.
            return_attention: if True, return attention scores along with the output.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
            decov_loss: DeCov regularization loss.
            attention_scores (optional): dictionary containing attention scores if return_attention=True.
        """
        
        # If static is None and demo_dim > 0, try to split from x
        if static is None and self.demo_dim > 0:
            # Check if x has enough dimensions
            # Expect x to be [Batch, Time, demo_dim + lab_dim]
            if x.shape[2] == self.demo_dim + self.lab_dim:
                static = x[:, 0, :self.demo_dim]
                x = x[:, :, self.demo_dim:]
            else:
                # If dimensions don't match expectation for split, maybe static is missing
                # But we can't proceed without static if demo_dim > 0
                raise ValueError(f"Expected x to have {self.demo_dim + self.lab_dim} features or static provided. Got x shape {x.shape}")

        # x: [Batch, Time, lab_dim] [1, 8, 42]
        # static: [Batch, demo_dim] [1, 2]
        # mask: [Batch, Time] [1, 8]

        if return_attention:
            out, decov_loss, attention_scores = self.concare_layer(x, static, mask, return_attention=True)
            out = self.dropout(out)
            return out, decov_loss, attention_scores
        else:
            out, decov_loss = self.concare_layer(x, static, mask, return_attention=False)
            out = self.dropout(out)
            return out, decov_loss

    def get_attention_scores(
        self, x: torch.tensor, static: Optional[torch.tensor] = None, mask: Optional[torch.tensor] = None
    ) -> Dict[str, torch.tensor]:
        """Extract attention scores for input data.

        Args:
            x: a tensor of shape [batch size, sequence len, lab_dim].
            static: a tensor of shape [batch size, demo_dim].
            mask: an optional tensor of shape [batch size, sequence len].

        Returns:
            Dictionary containing:
            - 'final_attention': Feature-level attention scores [batch, lab_dim (+1 if demo)]
            - 'feature_time_attentions': Time-step attention for each feature [batch, lab_dim, time_step]
            - 'multihead_attention': Multi-head self-attention [batch, num_head, feature_dim, feature_dim]
        """
        self.eval()
        with torch.no_grad():
            _, _, attention_scores = self.forward(x, static, mask, return_attention=True)
        return attention_scores

    def analyze_feature_importance(
        self,
        x: torch.tensor,
        static: torch.tensor,
        mask: torch.tensor,
        feature_names: list,
    ) -> Dict[str, torch.tensor]:
        """Analyze feature importance for batch data.

        Args:
            x: Input tensor [batch_size, seq_len, lab_dim] (must be 3D)
            static: Static features [batch_size, demo_dim] (must be 2D)
            mask: Attention mask [batch_size, seq_len] (must be 2D)
            feature_names: List of feature names (required)

        Returns:
            Dictionary containing analyzed attention scores and feature attention dicts
        """
        # Assert input format requirements
        assert x.dim() == 3, f"x must be 3D tensor [batch_size, seq_len, lab_dim], got {x.dim()}D"
        assert (
            static is not None and static.dim() == 2
        ), f"static must be 2D tensor [batch_size, demo_dim], got {static.dim() if static is not None else None}D"
        assert (
            mask is not None and mask.dim() == 2
        ), f"mask must be 2D tensor [batch_size, seq_len], got {mask.dim() if mask is not None else None}D"
        assert feature_names is not None and len(feature_names) > 0, "feature_names must be provided and non-empty"
        assert self.demo_dim > 0, "Model must have demo features (demo_dim > 0)"

        # Get attention scores
        attention_scores = self.get_attention_scores(x, static, mask)
        final_attention = attention_scores["final_attention"]  # [batch, feature_dim]

        # Separate lab and demo features (we know demo_dim > 0 from assert)
        lab_attention = final_attention[:, :-1]  # [batch, lab_dim] - lab features
        demo_attention = final_attention[:, -1:]  # [batch, 1] - demo features

        results = {
            "lab_feature_attention": lab_attention,
            "demo_feature_attention": demo_attention,
            "feature_time_attentions": attention_scores["feature_time_attentions"],
            "multihead_attention": attention_scores["multihead_attention"],
        }

        # Create feature attention dictionary for each sample in the batch
        batch_size = lab_attention.shape[0]
        feature_attention_dicts = []

        for batch_idx in range(batch_size):
            feature_attention_dict = {}

            # Map lab feature names to attention scores
            lab_scores = lab_attention[batch_idx].cpu().numpy()  # [lab_dim]

            # Use first lab_dim feature names for lab features
            lab_feature_names = feature_names[: lab_scores.shape[0]]
            for name, score in zip(lab_feature_names, lab_scores):
                feature_attention_dict[name] = float(score)

            # Add demo features (we know demo_attention exists from assert)
            demo_scores = demo_attention[batch_idx].cpu().numpy()  # [1]
            # Demo attention is always 1-dimensional, so use single "demo" key
            feature_attention_dict["demo"] = float(demo_scores[0])

            feature_attention_dicts.append(feature_attention_dict)

        # Always return list for batch processing
        results["feature_attention_dicts"] = feature_attention_dicts

        return results
