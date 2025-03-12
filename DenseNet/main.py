import os
import torch
import torchvision
import torchvision.transforms as transforms
from parser import args
from models import ResNetModel
from trainers import FinetuneTrainer
from torch.utils.data import DataLoader, random_split
from utils import EarlyStopping, check_path, set_seed, save_args_txt, retrainer


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
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 전체 데이터셋 로드
    full_train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 훈련 데이터셋을 훈련(train)과 검증(val)으로 분리
    train_size = int(0.8 * len(full_train_dataset))  # 80% for training
    val_size = len(full_train_dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # DataLoader 설정
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # save model
    model = ResNetModel('DenseNet121_Weights.IMAGENET1K_V1')
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # trainer
    trainer = FinetuneTrainer(
        model, train_dataloader, val_dataloader, test_dataloader, None, args
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
