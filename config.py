import argparse
import os

HOME = os.environ['HOME']


def config_USC():
    parser = argparse.ArgumentParser(prog="CMFD")

    parser.add_argument("--dataset", type=str, default="usc")

    parser.add_argument("--size", type=str, default="320x320",
                        help="image shape (h x w)")

    parser.add_argument("--model", type=str, default="dlab", help="model name")
    # network config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "-b", type=int, default=20)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--resume", type=int, default=1,
                        help="resume from epoch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", type=str, default="",
                        help="model name suffix")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="pretrained model path")
    parser.add_argument("--test", action='store_true', help="test only mode")
    parser.add_argument("--thres", type=float, default=0.5,
                        help="threshold for detection")
    # path config
    parser.add_argument("--lmdb-dir", type=str,
                        default=HOME+"/dataset/CMFD/USCISI-CMFD")
    parser.add_argument("--train-key", type=str, default="valid.keys")
    parser.add_argument("--test-key", type=str, default="test.keys")
    parser.add_argument("--valid-key", type=str, default="valid.keys")
    parser.add_argument("--out-channel", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--gamma2", type=float, default=1e-5)

    parser.add_argument("--bw", action='store_true', help='whether to add boundary loss')

    args = parser.parse_args()
    print(args)
    return args
