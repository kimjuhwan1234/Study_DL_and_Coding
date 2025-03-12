import os
import torch
from parser import args
from models import RNNModel
from datasets import CustomDataset
from trainers import FinetuneTrainer
from torch.utils.data import DataLoader
from utils import EarlyStopping, check_path, set_seed, save_args_txt, load_data, retrainer


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
    args.data_file = args.data_dir + "DB.pkl"
    train, val, test = load_data(args.data_file)

    # dataset
    train_dataset = CustomDataset(train)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)

    eval_dataset = CustomDataset(val)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)

    test_dataset = CustomDataset(test)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    # save model
    model = RNNModel(args)
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # trainer
    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, test_dataloader, None, args
    )

    # early_stoping & pretrain & retraining
    early_stopping = EarlyStopping(args.checkpoint_path, patience=10)
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
