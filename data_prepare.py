#!/usr/bin/env
# coding=utf-8
import time
from utils.arguments import args
from models.utils.tfData import TFData, save2tfrecord, split_save, readTFRecord, save2tfrecord_multilabel
from utils.tools import get_bucket


def main():
    # confirm = input("You are going to generate new tfdata, may covering the existing one.\n press ENTER to continue. ")
    # if confirm == "":
    #     print('will generate tfdata in 5 secs!')
    #     time.sleep(5)
    # save2tfrecord(args.dataset_dev, args.dirs.dev.tfdata, size_file=1000000)
    # save2tfrecord(args.dataset_train, args.dirs.train.tfdata, size_file=10000000)
    # save2tfrecord(args.dataset_untrain, args.dirs.untrain.tfdata, size_file=10000000)
    # save2tfrecord_multilabel(args.dataset_dev, args.dirs.dev.tfdata, size_file=10000000)
    split_save(args.dataset_dev, args.dirs.dev.tfdata, size_file=5000)
    # save2tfrecord_multilabel(args.dataset_train, args.dirs.train.tfdata, size_file=10000000)
    #save2tfrecord_multilabel(args.dataset_untrain, args.dirs.untrain.tfdata, size_file=10000000)
    # TFData(args.dataset_train, args.dirs.train.tfdata, args, size_file=10000, max_feat_len=3000).split_save()
    # TFData(args.dataset_dev, args.dirs.dev.tfdata, args, size_file=5000, max_feat_len=3000).split_save()
    # TFData(args.dataset_dev, args.dirs.dev.tfdata, args, size_file=5000, max_feat_len=3000).save()
    # TFData(args.dataset_test, args.dirs.test.tfdata, args, size_file=1000, max_feat_len=3000).split_save()
    # print(args.data.dim_feature)
    # feat, label = readTFRecord(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
    # get_bucket(args.dirs.train.tfdata / 'feature_length.txt', args.num_batch_tokens, 80)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    main()
