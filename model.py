"""GATFormer: graph attention branch over a label-free kNN sample graph (GQDO-selected pooled features)
+ Transformer encoder branch over frame-level fused features, gated fusion, disease prototype learning.

Test-time graph construction: edges are built ONLY from feature similarity (cosine kNN) among the
batch samples and a fixed bank of training anchors. No label enters inference. The Disease Relationship
Graph is used only in the training losses (graph-consistency term) and in the GQDO fitness.
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


# --------------------------------------------------------------------------- graph attention layer (dense, batch graph)
class GATLayer(nn.Module):
    def __init__(self, din, dout, heads=8, dropout=0.3, concat=True):
        super().__init__()
        self.h, self.dk, self.concat = heads, dout // heads if concat else dout, concat
        self.W = nn.Linear(din, self.h * self.dk, bias=False)
        self.a_src = nn.Parameter(torch.empty(self.h, self.dk)); self.a_dst = nn.Parameter(torch.empty(self.h, self.dk))
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        """x: (N, din); adj: (N, N) bool/float mask with self loops. Returns (N, dout) and attention (h, N, N)."""
        N = x.shape[0]
        Wh = self.W(x).view(N, self.h, self.dk)                       # (N, h, dk)
        es = (Wh * self.a_src).sum(-1); ed = (Wh * self.a_dst).sum(-1)  # (N, h)
        e = F.leaky_relu(es.T.unsqueeze(2) + ed.T.unsqueeze(1), 0.2)   # (h, N, N)
        e = e.masked_fill(~adj.bool().unsqueeze(0), float("-inf"))
        att = torch.softmax(e, -1); att = self.drop(att)
        out = torch.einsum("hij,jhd->ihd", att, Wh)                    # (N, h, dk)
        out = out.reshape(N, -1) if self.concat else out.mean(1)
        return out, att


def knn_adj(z, k=8):
    """Label-free cosine kNN graph with self loops, symmetrized. z: (N, d)."""
    zn = F.normalize(z, dim=1); sim = zn @ zn.T
    sim.fill_diagonal_(-2)
    k = min(k, z.shape[0] - 1)
    idx = sim.topk(k, dim=1).indices
    adj = torch.zeros_like(sim, dtype=torch.bool); adj.scatter_(1, idx, True)
    adj = adj | adj.T; adj.fill_diagonal_(True)
    return adj


# --------------------------------------------------------------------------- transformer branch
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d); pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x): return x + self.pe[: x.shape[1]].unsqueeze(0)


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__(); self.q = nn.Linear(d, 1)

    def forward(self, h):                      # h: (B, T, d)
        w = torch.softmax(self.q(h).squeeze(-1), 1)   # (B, T)
        return (w.unsqueeze(-1) * h).sum(1), w


# --------------------------------------------------------------------------- GATFormer
class GATFormer(nn.Module):
    def __init__(self, d_vec, d_frame, n_classes, d=128, gat_heads=8, tf_layers=3, tf_heads=8, dropout=0.3,
                 use_gat=True, use_tf=True, knn=8):
        super().__init__()
        assert use_gat or use_tf
        self.use_gat, self.use_tf, self.knn, self.d = use_gat, use_tf, knn, d
        if use_gat:
            self.g_in = nn.Sequential(nn.Linear(d_vec, d), nn.LayerNorm(d), nn.ELU(), nn.Dropout(dropout))
            self.gat1 = GATLayer(d, d, gat_heads, dropout); self.gat2 = GATLayer(d, d, gat_heads, dropout)
            self.g_norm = nn.LayerNorm(d)
        if use_tf:
            self.t_in = nn.Sequential(nn.Linear(d_frame, d), nn.LayerNorm(d))
            self.pe = PositionalEncoding(d)
            layer = nn.TransformerEncoderLayer(d, tf_heads, 4 * d, dropout, batch_first=True, activation="gelu", norm_first=True)
            self.tf = nn.TransformerEncoder(layer, tf_layers); self.pool = AttnPool(d); self.t_norm = nn.LayerNorm(d)
        if use_gat and use_tf:
            self.gate = nn.Linear(2 * d, d)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes))
        self.n_classes = n_classes

    def forward(self, vec, seq, anchors_vec=None, anchors_seq=None, return_aux=False):
        """vec: (B, d_vec) selected pooled features; seq: (B, T, d_frame).
        anchors_*: optional training anchors appended to the graph at inference (label-free)."""
        B = vec.shape[0]; aux = {}
        if self.use_gat:
            if anchors_vec is not None:
                v_all = torch.cat([vec, anchors_vec], 0)
            else:
                v_all = vec
            g0 = self.g_in(v_all)
            adj = knn_adj(v_all, self.knn)
            g1, att1 = self.gat1(g0, adj); g1 = F.elu(g1) + g0
            g2, att2 = self.gat2(g1, adj); g2 = self.g_norm(F.elu(g2) + g1)
            z_g = g2[:B]; aux["gat_att"] = att2[:, :B, :].detach()
        if self.use_tf:
            h = self.pe(self.t_in(seq)); h = self.tf(h)
            z_t, w = self.pool(h); z_t = self.t_norm(z_t); aux["tf_att"] = w.detach()
        if self.use_gat and self.use_tf:
            g = torch.sigmoid(self.gate(torch.cat([z_g, z_t], 1)))
            z = g * z_g + (1 - g) * z_t; aux["gate"] = g.detach()
        elif self.use_gat: z = z_g
        else: z = z_t
        logits = self.head(z)
        return (logits, z, aux) if return_aux else (logits, z)


# --------------------------------------------------------------------------- losses
class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0):
        super().__init__(); self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32)); self.gamma = gamma

    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1); p = logp.exp()
        return (-self.alpha[y] * (1 - p) ** self.gamma * logp).mean()


class PrototypeBank(nn.Module):
    """EMA class prototypes (Eq. 43) and the corrected prototype loss (Eq. 44):
       L = -1/N sum_i log [ exp(-||h_i - p_{y_i}||^2 / tau) / sum_c exp(-||h_i - p_c||^2 / tau) ]"""
    def __init__(self, C, d, momentum=0.9, tau=0.1):
        super().__init__(); self.register_buffer("P", torch.zeros(C, d)); self.register_buffer("init", torch.zeros(C, dtype=torch.bool))
        self.m, self.tau = momentum, tau

    @torch.no_grad()
    def update(self, z, y):
        z = F.normalize(z, dim=1)
        for c in y.unique():
            m = z[y == c].mean(0)
            if self.init[c]: self.P[c] = self.m * self.P[c] + (1 - self.m) * m
            else: self.P[c] = m; self.init[c] = True

    def loss(self, z, y):
        z = F.normalize(z, dim=1); P = F.normalize(self.P, dim=1)
        d2 = torch.cdist(z, P) ** 2                              # (B, C) squared distances to every prototype
        logits = -d2 / self.tau
        return F.cross_entropy(logits, y)                        # = -log softmax over all classes (p_c in the denominator)

    def graph_consistency(self, z, y, A_ref):
        """Graph-consistency regulariser (Eq. 46): the similarity structure of the class centroids of the current batch
        (which the momentum prototypes track) is pulled towards the reference Disease Relationship Graph.
        Computed on the batch embeddings so that the term carries gradient; classes absent from the batch are skipped."""
        present = torch.unique(y)
        if len(present) < 2: return z.sum() * 0.0
        cent = F.normalize(torch.stack([z[y == c].mean(0) for c in present]), dim=1)      # (K, d)
        S = cent @ cent.T
        S = S.masked_fill(torch.eye(len(present), dtype=torch.bool, device=S.device), float("-inf"))
        S = torch.softmax(S / 0.1, 1)
        A = A_ref[present][:, present]; A = A / (A.sum(1, keepdim=True) + 1e-12)
        return ((A - S) ** 2).sum() / len(present)
