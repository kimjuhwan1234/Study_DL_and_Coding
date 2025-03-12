import torch
import torch.nn as nn


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, nonlinearity='tanh', bias=True, device=None, dtype=None):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device if device else torch.device("cpu")
        self.dtype = dtype if dtype else torch.float32

        assert nonlinearity in ['tanh', 'relu'], "Nonlinearity must be either 'tanh' or 'relu'"
        self.activation = torch.tanh if nonlinearity == 'tanh' else torch.relu

        self.weight_ih = nn.Parameter(torch.randn(hidden_size, input_size, dtype=self.dtype, device=self.device))
        self.weight_hh = nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=self.dtype, device=self.device))

        if bias:
            self.bias_ih = nn.Parameter(torch.randn(hidden_size, dtype=self.dtype, device=self.device))
            self.bias_hh = nn.Parameter(torch.randn(hidden_size, dtype=self.dtype, device=self.device))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

    def forward(self, x, h_prev):
        x = x.to(self.weight_ih.dtype)

        return self.activation(
            x @ self.weight_ih.T + (self.bias_ih if self.bias_ih is not None else 0) +
            h_prev @ self.weight_hh.T + (self.bias_hh if self.bias_hh is not None else 0)
        )


class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False,
                 device=None, dtype=None):
        super(RNNLayer, self).__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.device = device if device else torch.device("cpu")
        self.dtype = dtype if dtype else torch.float32

        self.cells = nn.ModuleList([
            RNNCell(input_size if layer == 0 else hidden_size, hidden_size, nonlinearity, bias, device, dtype)
            for layer in range(num_layers)
        ])

    def forward(self, x, hx=None):
        if self.batch_first:
            x = x.transpose(0, 1)  # (seq_len, batch, input_size)
        seq_len, batch_size, _ = x.size()

        if hx is None:
            hx = torch.zeros(self.num_layers, batch_size, self.cells[0].hidden_size, dtype=self.dtype,
                             device=self.device)

        h_t = hx.clone()
        output = []

        for t in range(seq_len):
            new_h = []
            h_t_input = x[t]
            for layer in range(self.num_layers):
                h_t_l = self.cells[layer](h_t_input, h_t[layer])
                new_h.append(h_t_l)
                h_t_input = h_t_l
            h_t = torch.stack(new_h)
            output.append(h_t[-1])

        output = torch.stack(output)  # (seq_len, batch, hidden_size)

        if self.batch_first:
            output = output.transpose(0, 1)  # (batch, seq_len, hidden_size)

        return output, h_t
