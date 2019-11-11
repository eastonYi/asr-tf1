#!/usr/bin/env
# coding=utf-8
import time
from utils.arguments import args
from models.utils.tfData import save2tfrecord, readTFRecord
from utils.tools import get_bucket


def main():
    confirm = input("You are going to generate new tfdata, may covering the existing one.\n press ENTER to continue. ")
    if confirm == "":
        print('will generate tfdata in 5 secs!')
        time.sleep(5)
    # save2tfrecord(args.dataset_dev, args.dirs.dev.tfdata, size_file=10000000)
    save2tfrecord(args.dataset_train, args.dirs.train.tfdata, size_file=10000000)
    # save2tfrecord(args.dataset_untrain, args.dirs.untrain.tfdata, size_file=10000000)
    # print(args.data.dim_feature)
    # feat, label = readTFRecord(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
    # get_bucket(args.dirs.train.tfdata / 'feature_length.txt', args.num_batch_tokens, 170)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    main()
