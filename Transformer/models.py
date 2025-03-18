import torch.nn as nn
from modules import Transformer


class TransformerModel(nn.Module):
    def __init__(self, args):
        super(TransformerModel, self).__init__()

        # 입력 임베딩 레이어
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)

        # Transformer 정의
        self.transformer = Transformer(
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            num_decoder_layers=args.num_decoder_layers,
            batch_first=False  # Transformer는 (seq_len, batch, d_model) 형식을 기대
        )

        self.fc = nn.Linear(args.d_model, 1)  # Transformer 출력에서 예측값 생성

    def forward(self, input_ids, attention_mask=None, gt=None):
        # 1️⃣ 임베딩 변환
        embedded = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        embedded = embedded.permute(1, 0, 2)  # (seq_len, batch_size, d_model)

        # 2️⃣ attention_mask → PyTorch 형식 변환
        src_key_padding_mask = attention_mask == 0 if attention_mask is not None else None

        # 3️⃣ Transformer 인코더 적용
        transformer_out = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)

        # 4️⃣ 마지막 토큰의 출력 사용
        output = self.fc(transformer_out[-1])  # (batch_size, 1)

        # 5️⃣ Loss 계산 (선택적)
        if gt is not None:
            gt = gt.float()  # (batch_size, 1)로 reshape
            loss = nn.BCEWithLogitsLoss()(output, gt)
            return output, loss

        return output
