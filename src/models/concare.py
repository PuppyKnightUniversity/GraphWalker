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

        (
            batch_size,
            time_step,
            lab_dim,
        ) = x.size()
        input_q = self.W_q(x[:, -1, :])
        input_k = self.W_k(x)
        input_v = self.W_v(x)

        if self.attention_type == "add":

            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim))
            h = q + input_k + self.b_in
            h = self.tanh(h)
            e = self.W_out(h)
            e = torch.reshape(e, (batch_size, time_step))

        elif self.attention_type == "mul":
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1))
            e = torch.matmul(input_k, q)
            e = e.squeeze(-1)
            if e.dim() == 1:
                e = e.unsqueeze(0)

        elif self.attention_type == "concat":
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)
            k = input_k
            c = torch.cat((q, k), dim=-1)
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba
            e = torch.reshape(e, (batch_size, time_step))
        else:
            raise ValueError("Unknown attention type: {}, please use add, mul, concat".format(self.attention_type))

        a = self.softmax(e)
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()

        return v, a


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x)))), None


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, self.d_k * self.h) for _ in range(3)])
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def cov(self, m, y=None):
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]

        denominator = x.size(1) - 1
        if denominator <= 0:
            return torch.zeros(x.size(0), x.size(0), device=x.device, dtype=x.dtype)

        cov = (1 / denominator) * x.mm(x.t())
        return cov

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        feature_dim = query.size(1)


        query, key, value = [
            layer(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for layer, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        DeCov_contexts = x.transpose(0, 1).transpose(1, 2)
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

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
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


        if attention_type == "add":
            if self.time_aware:
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
        ) = x.size()

        time_decays = (
            torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(device=device)
        )
        b_time_decays = time_decays.repeat(batch_size, 1, 1) + 1

        if self.attention_type == "add":
            last_visit = get_last_visit(x, mask)
            q = torch.matmul(last_visit, self.Wt)
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))
            if self.time_aware is True:
                k = torch.matmul(x, self.Wx)
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)
            else:
                k = torch.matmul(x, self.Wx)
            h = q + k + self.bh
            if self.time_aware:
                h += time_hidden
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba
            e = torch.reshape(e, (batch_size, time_step))
        elif self.attention_type == "mul":
            last_visit = get_last_visit(x, mask)
            e = torch.matmul(last_visit, self.Wa)
            e = torch.matmul(e.unsqueeze(1), x.permute(0, 2, 1)).reshape(batch_size, time_step) + self.ba
        elif self.attention_type == "concat":
            last_visit = get_last_visit(x, mask)
            q = last_visit.unsqueeze(1).repeat(1, time_step, 1)
            k = x
            c = torch.cat((q, k), dim=-1)
            if self.time_aware:
                c = torch.cat((c, b_time_decays), dim=-1)
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba
            e = torch.reshape(e, (batch_size, time_step))

        elif self.attention_type == "new":
            last_visit = get_last_visit(x, mask)
            q = torch.matmul(last_visit, self.Wt)
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))
            k = torch.matmul(x, self.Wx)
            dot_product = torch.matmul(q, k.transpose(1, 2)).reshape(batch_size, time_step)
            denominator = self.sigmoid(self.rate) * (
                torch.log(2.72 + (1 - self.sigmoid(dot_product))) * (b_time_decays.reshape(batch_size, time_step))
            )
            e = self.relu(self.sigmoid(dot_product) / (denominator))
        else:
            raise ValueError("Wrong attention type. Plase use 'add', 'mul', 'concat' or 'new'.")

        if mask is not None:
            e = e.masked_fill(mask == 0, -1e9)
        a = self.softmax(e)
        v = torch.matmul(a.unsqueeze(1), x).reshape(batch_size, lab_dim)

        return v, a


class ConCareLayer(nn.Module):

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

        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
        self.dropout = dropout
        self.demo_dim = demo_dim

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
        if self.demo_dim > 0:
            demo_main = self.tanh(self.demo_proj_main(static)).unsqueeze(1)

        batch_size = x.size(0)
        feature_dim = x.size(2)

        if self.transformer_hidden % self.num_head != 0:
            raise ValueError("transformer_hidden must be divisible by num_head")

        if mask is not None:
            mask = normalize_mask(mask, x.shape[1])

        attention_scores = {} if return_attention else None
        feature_time_attentions = [] if return_attention else None

        GRU_embeded_input = self.GRUs[0](
            x[:, :, 0].unsqueeze(-1).to(device=x.device),
            torch.zeros(batch_size, self.hidden_dim).to(device=x.device).unsqueeze(0),
        )[
            0
        ]

        feature_output, time_attention = self.LastStepAttentions[0](GRU_embeded_input, mask, x.device)
        Attention_embeded_input = feature_output.unsqueeze(1)

        if return_attention:
            feature_time_attentions.append(time_attention)

        for i in range(feature_dim - 1):
            embeded_input = self.GRUs[i + 1](
                x[:, :, i + 1].unsqueeze(-1),
                torch.zeros(batch_size, self.hidden_dim).to(device=x.device).unsqueeze(0),
            )[
                0
            ]
            feature_output, time_attention = self.LastStepAttentions[i + 1](embeded_input, mask, x.device)
            embeded_input = feature_output.unsqueeze(1)
            Attention_embeded_input = torch.cat((Attention_embeded_input, embeded_input), 1)

            if return_attention:
                feature_time_attentions.append(time_attention)

        if self.demo_dim > 0:
            Attention_embeded_input = torch.cat((Attention_embeded_input, demo_main), 1)
        posi_input = self.dropout(Attention_embeded_input)

        contexts = self.SublayerConnection(
            posi_input,
            lambda _: self.MultiHeadedAttention(posi_input, posi_input, posi_input, None),
        )

        DeCov_loss = contexts[1]
        contexts = contexts[0]

        if return_attention:
            attention_scores["multihead_attention"] = (
                self.MultiHeadedAttention.attn
            )
            attention_scores["feature_time_attentions"] = torch.stack(
                feature_time_attentions, dim=1
            )

        contexts = self.SublayerConnection(contexts, lambda _: self.PositionwiseFeedForward(contexts))[0]

        weighted_contexts, final_attention = self.FinalAttentionQKV(contexts)

        if return_attention:
            attention_scores["final_attention"] = final_attention

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

        self.lab_dim = lab_dim
        self.hidden_dim = hidden_dim
        self.transformer_hidden = hidden_dim
        self.num_head = num_head
        self.pe_hidden = pe_hidden
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
        
        if static is None and self.demo_dim > 0:
            if x.shape[2] == self.demo_dim + self.lab_dim:
                static = x[:, 0, :self.demo_dim]
                x = x[:, :, self.demo_dim:]
            else:
                raise ValueError(f"Expected x to have {self.demo_dim + self.lab_dim} features or static provided. Got x shape {x.shape}")


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
        assert x.dim() == 3, f"x must be 3D tensor [batch_size, seq_len, lab_dim], got {x.dim()}D"
        assert (
            static is not None and static.dim() == 2
        ), f"static must be 2D tensor [batch_size, demo_dim], got {static.dim() if static is not None else None}D"
        assert (
            mask is not None and mask.dim() == 2
        ), f"mask must be 2D tensor [batch_size, seq_len], got {mask.dim() if mask is not None else None}D"
        assert feature_names is not None and len(feature_names) > 0, "feature_names must be provided and non-empty"
        assert self.demo_dim > 0, "Model must have demo features (demo_dim > 0)"

        attention_scores = self.get_attention_scores(x, static, mask)
        final_attention = attention_scores["final_attention"]

        lab_attention = final_attention[:, :-1]
        demo_attention = final_attention[:, -1:]

        results = {
            "lab_feature_attention": lab_attention,
            "demo_feature_attention": demo_attention,
            "feature_time_attentions": attention_scores["feature_time_attentions"],
            "multihead_attention": attention_scores["multihead_attention"],
        }

        batch_size = lab_attention.shape[0]
        feature_attention_dicts = []

        for batch_idx in range(batch_size):
            feature_attention_dict = {}

            lab_scores = lab_attention[batch_idx].cpu().numpy()

            lab_feature_names = feature_names[: lab_scores.shape[0]]
            for name, score in zip(lab_feature_names, lab_scores):
                feature_attention_dict[name] = float(score)

            demo_scores = demo_attention[batch_idx].cpu().numpy()
            feature_attention_dict["demo"] = float(demo_scores[0])

            feature_attention_dicts.append(feature_attention_dict)

        results["feature_attention_dicts"] = feature_attention_dicts

        return results
