import torch.nn as nn
from modules import RNNLayer
import torch.nn.functional as F


class RNNModel(nn.Module):
    def __init__(self, args, dtype=None):
        super(RNNModel, self).__init__()
        self.rnn = RNNLayer(args.input_size, args.hidden_size, args.num_layers, args.nonlinearity, args.bias,
                            args.batch_first, args.device, dtype)
        self.output_layer = nn.Linear(args.hidden_size, 1)  # 예측값을 위한 출력 레이어

    def forward(self, x, gt=None, hx=None):
        rnn_out, h_t = self.rnn(x, hx)
        output = self.output_layer(rnn_out[:, -1, :])  # 마지막 타임스텝의 출력 사용

        if gt != None:
            loss = F.mse_loss(output, gt)
            return output, loss

        return output, h_t
