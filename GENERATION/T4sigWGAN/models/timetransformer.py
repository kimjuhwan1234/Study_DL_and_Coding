import torch
import torch.nn as nn


class TimesFormerLayer(nn.Module):
    def __init__(self, dim, num_heads, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding='same', dilation=dilation)
        self.relu = nn.ReLU()
        self.norm_tcn = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.mha1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        self.ff_conv = nn.Conv1d(dim, dim, kernel_size=1)
        self.ff_relu = nn.ReLU()
        self.ff_norm = nn.LayerNorm(dim)

        self.cross1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, tcn_input, trans_input):  # (B, T, D)
        # Temporal Convolution
        tcn_out = tcn_input.transpose(1, 2)  # (B, D, T)
        tcn_out = self.conv(tcn_out)  # (B, D, T)
        tcn_out = self.relu(tcn_out)
        tcn_out = tcn_out.transpose(1, 2)  # (B, T, D)
        tcn_out = self.norm_tcn(tcn_out)
        tcn_out = self.dropout(tcn_out)

        # Transformer self-attention
        x, _ = self.mha1(trans_input, trans_input, trans_input)
        x = self.norm1(x + trans_input)

        # Feedforward
        x_ff = self.ff_conv(x.transpose(1, 2))  # (B, D, T)
        x_ff = self.ff_relu(x_ff)
        x_ff = x_ff.transpose(1, 2)  # (B, T, D)
        x_ff = self.ff_norm(x_ff)
        trans_out = x_ff + x

        # Cross attention
        ca1, _ = self.cross1(trans_out, tcn_out, tcn_out)
        chnl_trans = self.norm2(ca1 + trans_out)

        ca2, _ = self.cross2(tcn_out, trans_out, trans_out)
        chnl_tcn = self.norm3(ca2 + tcn_out)

        return chnl_tcn, chnl_trans


class TimesFormerDecoder(nn.Module):
    def __init__(self, ts_shape, num_heads, k_size, dilations, dropout):
        super().__init__()
        ts_len, ts_dim = ts_shape

        self.timesformer_blocks = nn.ModuleList([
            TimesFormerLayer(ts_dim, num_heads, k_size, d, dropout)
            for d in dilations
        ])

        self.final = nn.Sequential(
            nn.Flatten(),  # (B, ts_len, ts_dim * 2) → (B, ts_len * ts_dim * 2)
            nn.Linear(ts_len * ts_dim * 2, ts_len * 10),  # (B, ...) → (B, ts_len * 10)
            nn.Unflatten(1, (ts_len, 10))  # → (B, ts_len, 10)
        )

    def forward(self, x):
        res = x
        tcn, trans = res, res
        for layer in self.timesformer_blocks:
            tcn, trans = layer(tcn, trans)

        x = torch.cat([tcn, trans], dim=-1)
        return self.final(x)
