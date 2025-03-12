import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import LSTM


class LSTMModel(nn.Module):
    def __init__(self, args):
        super(LSTMModel, self).__init__()
        self.lstm = LSTM(
            args.input_size,  # 입력 특성의 크기 (예: 10개 feature)
            args.hidden_size,  # LSTM의 hidden state 크기
            num_layers=args.num_layers,  # LSTM 레이어 개수 (기본값=1)
            bias=args.bias,  # bias 사용 여부 (기본값=True)
            batch_first=args.batch_first,  # 입력 텐서의 배치 차원 우선 여부
            bidirectional=args.bidirectional  # 양방향 LSTM 사용 여부
        )

        self.fc = nn.Linear(args.hidden_size * 2, 1)  # 최종 출력 레이어

    def forward(self, x, gt=None):
        x = x.to(torch.float32)

        lstm_out, _ = self.lstm(x)  # LSTM 실행
        output = self.fc(lstm_out[:, -1, :])  # 마지막 타임스텝의 출력 사용
        if gt != None:
            loss = F.mse_loss(output, gt)
            return output, loss

        return output
