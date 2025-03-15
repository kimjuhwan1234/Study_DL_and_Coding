import argparse

parser = argparse.ArgumentParser()

# initial setting
parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--batch_size", type=int, default=64, help="number of batch_size")
parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
parser.add_argument("--data_dir", default="../Database/", type=str)
parser.add_argument("--output_dir", default="output/", type=str)

# log location
parser.add_argument("--model_name", default="RNN", type=str)
parser.add_argument("--data_name", default="SP_vol", type=str)

# model parameter
parser.add_argument("--input_size", type=int, default=2, help="hidden size of transformer model")
parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
parser.add_argument("--nonlinearity", default="tanh", type=str)
parser.add_argument("--bias", default=True, type=bool)
parser.add_argument("--batch_first", default=True, type=bool)

# trainer parameter
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
parser.add_argument("--log_freq", type=int, default=5, help="per epoch print res")

# main
parser.add_argument("--using_pretrain", default=False, action="store_true")

args = parser.parse_args()
