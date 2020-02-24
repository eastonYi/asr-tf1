#!/usr/bin/env
# coding=utf-8
from utils.arguments import args
from models.utils.tfData import TFDataSaver
from utils.tools import get_bucket


def main(overwrite=False):
    # import pdb; pdb.set_trace()
    # confirm = input("You are going to generate new tfdata, may covering the existing one.\n press ENTER to continue. ")
    # if confirm == "":
    #     print('will generate tfdata in 5 secs!')
    #     time.sleep(5)
    # save2tfrecord_multilabel(args.dataset_train, args.dirs.train.tfdata, size_file=10000000)
    #save2tfrecord_multilabel(args.dataset_untrain, args.dirs.untrain.tfdata, size_file=10000000)
    TFDataSaver(args.dataset_train, args.dirs.train.tfdata, args, overwrite, size_file=30000, max_feat_len=3000).split_save()
    TFDataSaver(args.dataset_dev, args.dirs.dev.tfdata, args, overwrite, size_file=10000, max_feat_len=3000).split_save()
    # print(args.data.dim_feature)
    # feat, label = readTFRecord(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
    get_bucket(args.dirs.train.tfdata / 'feature_length.txt', args.num_batch_tokens, 180)


def check():
    import tensorflow as tf
    from pathlib import Path
    from utils.dataset import ASR_scp_DataSet, ASRDataLoader
    from models.utils.tfData import TFDataReader

    dataset = ASR_scp_DataSet(
        f_scp=args.dirs.demo.scp,
        f_trans=args.dirs.demo.trans,
        args=args,
        _shuffle=False,
        transform=False)
    TFDataSaver(dataset, Path(args.dirs.demo.tfdata), args, size_file=1, max_feat_len=3000).split_save()

    # train
    dataReader = TFDataReader(
        args.dirs.demo.tfdata,
        args=args,
        _shuffle=True,
        transform=True)
    batch = dataReader.fentch_batch_bucket()

    # dev
    dataReader = TFDataReader(
        args.dirs.demo.tfdata,
        args=args,
        _shuffle=False,
        transform=True)
    dataLoader = ASRDataLoader(
        dataset,
        args,
        dataReader.feat,
        dataReader.label,
        batch_size=2,
        num_loops=1)

    # test
    dataset = ASR_scp_DataSet(
        f_scp=args.dirs.demo.scp,
        f_trans=args.dirs.demo.trans,
        args=args,
        _shuffle=False,
        transform=True)

    dataset_2 = ASR_scp_DataSet(
        f_scp=args.dirs.demo.scp,
        f_trans=args.dirs.demo.trans,
        args=args,
        _shuffle=False,
        transform=False)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        dataLoader.sess = sess

        import pdb; pdb.set_trace()

        batch = sess.run(batch)
        sample_dev = next(iter(dataLoader))
        sample = dataset[0]


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import sys

    overrite = sys.argv[-1]
    if overrite == '1':
        overrite = True
    else:
        overrite = False

    main(overrite=overrite)
#     check()
