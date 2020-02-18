import argparse
import os

HOME = os.environ["HOME"]


def config():
    parser = argparse.ArgumentParser(prog="CISDL")
    parser.add_argument("--dataset", type=str, default="coco")

    parser.add_argument(
        "--size", type=str, default="320x320", help="image shape (w x h)"
    )
    parser.add_argument("--model", type=str, default="base", help="model name")
    # network config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "-b", type=int, default=10)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument(
        "--resume", type=int, default=1, help="resume from epoch"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--suffix", type=str, default="", help="model name suffix"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="pretrained model path"
    )
    parser.add_argument("--test", action="store_true", help="test only mode")
    parser.add_argument(
        "--thres", type=float, default=0.5, help="threshold for detection"
    )
    # path config
    parser.add_argument(
        "--data-root", type=str, default=HOME + "/dataset/CMFD/DMAC-COCO/"
    )
    parser.add_argument("--out-channel", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--gamma2", type=float, default=0.1)
    parser.add_argument(
        "--bw", action="store_false", help="whether to add boundary loss"
    )
    parser.add_argument(
        "--wo-det",
        action="store_false",
        help="whether to remove detection loss",
    )
    parser.add_argument(
        "--plot", action="store_true", help="whether to plot during test"
    )

    parser.add_argument("--tune", action="store_true")

    parser.add_argument("--eval-bn", action="store_true")

    parser.add_argument("--mode", type=str, default=None)

    args = parser.parse_args()
    args.size = tuple(int(i) for i in args.size.split("x"))
    print(args)
    return args


def config_casia():
    parser = argparse.ArgumentParser(prog="CISDL")
    parser.add_argument("--dataset", type=str, default="casia")

    parser.add_argument(
        "--size", type=str, default="320x320", help="image shape (h x w)"
    )
    parser.add_argument("--num", type=int, default=10)

    parser.add_argument("--model", type=str, default="base", help="model name")
    # network config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "-b", type=int, default=10)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument(
        "--resume", type=int, default=1, help="resume from epoch"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--suffix", type=str, default="", help="model name suffix"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="pretrained model path"
    )
    parser.add_argument("--test", action="store_true", help="test only mode")
    parser.add_argument(
        "--thres", type=float, default=0.5, help="threshold for detection"
    )
    # path config
    parser.add_argument(
        "--data-root", type=str, default=HOME + "/dataset/CMFD/DMAC-COCO/"
    )
    parser.add_argument("--out-channel", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--gamma2", type=float, default=1e-5)
    parser.add_argument(
        "--bw", action="store_true", help="whether to add boundary loss"
    )
    parser.add_argument(
        "--wo-det",
        action="store_false",
        help="whether to remove detection loss",
    )
    parser.add_argument(
        "--plot", action="store_true", help="whether to plot during test"
    )

    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--tune", action="store_true")

    args = parser.parse_args()
    args.size = tuple(int(i) for i in args.size.split("x"))
    print(args)
    return args




def config_video():
    parser = argparse.ArgumentParser(prog="CISDL_video")
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument(
        "--size", type=str, default="320x320", help="image shape (w x h)"
    )
    parser.add_argument("--model", type=str, default="base", help="model name")
    # network config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "-b", type=int, default=10)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument(
        "--resume", type=int, default=1, help="resume from epoch"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--suffix", type=str, default="", help="model name suffix"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="pretrained model path"
    )
    parser.add_argument("--test", action="store_true", help="test only mode")
    parser.add_argument(
        "--thres", type=float, default=0.5, help="threshold for detection"
    )
    # path config
    parser.add_argument(
        "--root",
        type=str,
        default=HOME + "/dataset/video_forge/",
        help="root folder for dataset",
    )
    parser.add_argument("--out-channel", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--gamma2", type=float, default=0.1)
    parser.add_argument(
        "--bw", action="store_false", help="whether to add boundary loss"
    )
    parser.add_argument(
        "--plot", action="store_true", help="whether to plot during test"
    )
    parser.add_argument(
        "--wo-det",
        action="store_false",
        help="whether to remove detection loss",
    )
    # split
    parser.add_argument("--split", type=float, default=0.5)
    parser.add_argument("--eval-bn", action="store_true")

    parser.add_argument("--tune", action="store_true")

    args = parser.parse_args()
    args.size = tuple(int(i) for i in args.size.split("x"))
    print(args)
    return args


def config_video_full():
    parser = argparse.ArgumentParser(prog="CISDL_video")
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument(
        "--size", type=str, default="320x320", help="image shape (w x h)"
    )
    parser.add_argument("--model", type=str, default="base", help="model name")
    # network config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "-b", type=int, default=10)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument(
        "--resume", type=int, default=1, help="resume from epoch"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--suffix", type=str, default="", help="model name suffix"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="pretrained model path"
    )
    parser.add_argument(
        "--ckptM", type=str, default=None, help="pretrained model path"
    )
    parser.add_argument("--test", action="store_true", help="test only mode")
    parser.add_argument(
        "--thres", type=float, default=0.5, help="threshold for detection"
    )
    # path config
    parser.add_argument(
        "--root",
        type=str,
        default=HOME + "/dataset/video_forge/",
        help="root folder for dataset",
    )
    parser.add_argument("--out-channel", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--gamma2", type=float, default=0.1)
    parser.add_argument(
        "--bw", action="store_false", help="whether to add boundary loss"
    )
    parser.add_argument(
        "--plot", action="store_true", help="whether to plot during test"
    )
    parser.add_argument(
        "--wo-det",
        action="store_false",
        help="whether to remove detection loss",
    )
    # split
    parser.add_argument("--split", type=float, default=0.5)
    parser.add_argument("--eval-bn", action="store_true")
    parser.add_argument("--tune", action="store_true")

    args = parser.parse_args()
    args.size = tuple(int(i) for i in args.size.split("x"))
    print(args)
    return args


def config_video_temporal():
    parser = argparse.ArgumentParser(prog="CISDL_video")
    parser.add_argument("--dataset", type=str, default="youtube")
    parser.add_argument(
        "--size", type=str, default="320x320", help="image shape (w x h)"
    )
    parser.add_argument("--model", type=str, default="base", help="model name")
    # network config
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", "-b", type=int, default=3)
    parser.add_argument("--t-max", type=int, default=5)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument(
        "--resume", type=int, default=1, help="resume from epoch"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--suffix", type=str, default="", help="model name suffix"
    )
    parser.add_argument(
        "--ckpt", type=str, default=None, help="pretrained model path"
    )
    parser.add_argument("--test", action="store_true", help="test only mode")
    parser.add_argument(
        "--thres", type=float, default=0.5, help="threshold for detection"
    )
    # path config
    parser.add_argument(
        "--root",
        type=str,
        default=HOME + "/dataset/video_forge/",
        help="root folder for dataset",
    )
    parser.add_argument("--out-channel", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--gamma2", type=float, default=0.3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument(
        "--bw", action="store_false", help="whether to add boundary loss"
    )
    parser.add_argument(
        "--plot", action="store_true", help="whether to plot during test"
    )
    parser.add_argument(
        "--wo-det",
        action="store_false",
        help="whether to remove detection loss",
    )
    # split
    parser.add_argument("--split", type=float, default=0.5)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--eval-bn", action="store_true")


    parser.add_argument("--tune", action="store_true")

    args = parser.parse_args()
    args.size = tuple(int(i) for i in args.size.split("x"))
    print(args)
    return args
