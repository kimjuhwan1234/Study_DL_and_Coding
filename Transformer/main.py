import os
import torch
import warnings
from parser import args
from models import TransformerModel
from datasets import load_dataset
from transformers import AutoTokenizer
from dataset import IMDBDataset
from trainers import FinetuneTrainer
from torch.utils.data import DataLoader, random_split
from utils import EarlyStopping, check_path, set_seed, save_args_txt, retrainer

warnings.filterwarnings("ignore")


def main():
    # initial setting
    set_seed(args.seed)
    check_path(args.output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda" if args.cuda_condition else "cpu")

    # log location
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    save_args_txt(args, args.log_file)

    # load data
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    # 데이터셋 변환
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = IMDBDataset(tokenized_datasets["train"])
    test_dataset = IMDBDataset(tokenized_datasets["test"])

    train_size = int(0.8 * len(train_dataset))  # 80% for training
    val_size = len(train_dataset) - train_size  # 20% for validation
    train_data, val_data = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # save model
    model = TransformerModel(args)
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # trainer
    trainer = FinetuneTrainer(
        model, train_dataloader, val_dataloader, test_dataloader, None, args
    )

    # early_stoping & pretrain & retraining
    early_stopping = EarlyStopping(args.checkpoint_path, patience=3)
    if args.using_pretrain:
        print(args.using_pretrain)
        args.backbone_weight_path = os.path.join(args.output_dir, "Pretrain.pt")
        try:
            early_stopping.best_score = retrainer(trainer, args)
        except FileNotFoundError:
            print(f"{args.backbone_weight_path} Not Found!")
    else:
        print("Not using pretrained model.")

    # model training
    for epoch in range(args.epochs):
        trainer.train(epoch)
        val_loss = trainer.valid(epoch)

        early_stopping([val_loss], trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # train log image save
    trainer.evaluate()

    # test
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores = trainer.test(0)
    print(scores)


if __name__ == "__main__":
    main()
