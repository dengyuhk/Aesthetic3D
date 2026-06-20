import argparse
import logging
import time
import os


def log(savePath):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(savePath, 'log.txt'), 'w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def get_opt():
    parser = argparse.ArgumentParser()

    # data paths
    parser.add_argument('--data_path', type=str, default='./data/pre_rendered_3.pt', help="")
    parser.add_argument('--pairs_path', type=str, default='./data/pairs_weighted_filtered.pt', help="")

    # saving parameters
    parser.add_argument('--logs_root', type=str, default='./Logs', help='')
    parser.add_argument('--log_path', type=str, default='', help='log path')
    parser.add_argument('--save_rate', type=int, default=5, help='')

    # loss parameters
    parser.add_argument('--margin', type=float, default=0.5, help='')

    # learning parameter
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs to train for')
    parser.add_argument('--use_SGD', type=int, default=0, help='')
    parser.add_argument('--lr', type=float, default=0.0005, help='')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='')
    parser.add_argument('--decay_mode', type=int, default=2, help='')
    parser.add_argument('--decay_rate', type=float, default=0.5, help='')
    parser.add_argument('--decay_epoch', type=int, default=10, help='number of iteration per decay')

    # network parameters
    parser.add_argument('--architecture', type=str, default='ModelImg2', 
                        choices=['ModelImg1', 'ModelImg2', 'ModelImg3', 'ModelImgLarge1', 'ModelImgLarge2', 'ModelImgHybrid'])
    parser.add_argument('--input_dims', type=int, default=6)
    parser.add_argument('--use_transform', type=int, default=1)

    opt = parser.parse_args()
    if not os.path.exists(opt.logs_root):
        os.mkdir(opt.logs_root)
    if opt.log_path == "":
        opt.log_path = time.strftime('%m%d%H%M')
    opt.log_path = os.path.join(opt.logs_root, opt.log_path)
    assert not os.path.exists(opt.log_path)
    os.mkdir(opt.log_path)
    return opt
