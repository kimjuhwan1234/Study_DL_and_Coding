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
    submission_dataset = CustomDataset(test, True)
    submission_dataloader = DataLoader(submission_dataset, batch_size=args.batch_size)

    # save model
    model = RNNModel(args)
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # trainer
    trainer = FinetuneTrainer(model, None, None, None, submission_dataloader, args)

    # test
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    pred = trainer.submission(0)
    print(pred)


if __name__ == "__main__":
    main()
