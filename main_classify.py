#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import logging
import tensorflow as tf
import numpy as np

from utils.arguments import args
from models.utils.tools import get_session, size_variables
from models.utils.tfData import TFReader, readTFRecord
from utils.dataset import ASRDataLoader
from utils.summaryTools import Summary
from utils.performanceTools import dev


def train():
    print('reading data form ', args.dirs.train.tfdata)
    dataReader_train = TFReader(args.dirs.train.tfdata, args=args)
    batch_train = dataReader_train.fentch_batch_bucket()

    feat, label = readTFRecord(args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
    dataloader_dev = ASRDataLoader(args.dataset_dev, args, feat, label, batch_size=args.batch_size, num_loops=1)

    tensor_global_step = tf.train.get_or_create_global_step()

    model = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.structure,
        decoder=args.model.decoder.structure,
        batch=batch_train,
        training=True,
        args=args)

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.structure,
        decoder=args.model.decoder.structure,
        training=False,
        args=args)

    size_variables()
    start_time = datetime.now()

    saver = tf.train.Saver(max_to_keep=15)

    summary = Summary(str(args.dir_log))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        if args.dirs.checkpoint_init:
            checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
            saver.restore(sess, checkpoint)

        dataloader_dev.sess = sess

        batch_time = time()
        num_processed = 0
        progress = 0

        while progress < args.num_epochs:
            global_step, lr = sess.run([tensor_global_step, model.learning_rate])
            loss, shape_batch, _, _ = sess.run(model.list_run)

            num_processed += shape_batch[0]
            used_time = time()-batch_time
            batch_time = time()
            progress = num_processed/args.data.train_size

            if global_step % 10 == 0:
                logging.info('loss: {:.3f}\tbatch: {} lr:{:.6f} time:{:.2f}s {:.3f}% step: {}'.format(
                              loss, shape_batch, lr, used_time, progress*100.0, global_step))
                summary.summary_scalar('loss', loss, global_step)
                summary.summary_scalar('lr', lr, global_step)

            if global_step % args.save_step == args.save_step - 1:
                saver.save(get_session(sess), str(args.dir_checkpoint/'model'), global_step=global_step, write_meta_graph=True)

            if global_step % args.dev_step == args.dev_step - 1:
                dev()

    logging.info('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def dev(step, dataloader, model, sess):
    start_time = time()
    batch_time = time()
    processed = 0
    list_res = []
    for batch in dataloader:
        if not batch: continue
        dict_feed = {model.list_pl[0]: batch[0],
                     model.list_pl[1]: batch[2]}
        _y, shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)

        used_time = time()-batch_time
        batch_time = time()
        processed += shape_batch[0]
        progress = processed/len(dataloader)
        y = batch[2]
        list_res.append(y==_y)

    precision = np.mean(list_res)
    used_time = time() - start_time
    logging.warning('=====dev info, total used time {:.2f}s ==== precision: {:.4f}'.format(
        used_time, precision))

    return precision

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)
    # import pdb; pdb.set_trace()

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info('enter the TRAINING phrase')
        train()
