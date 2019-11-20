#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import sys
import logging
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import numpy as np

from models.utils.tools import get_session, size_variables
from utils.arguments import args
from utils.dataset import TextDataSet
from utils.summaryTools import Summary
from utils.textTools import array_idx2char, array2text
from utils.tools import get_batch_length
from models.generator.baseGenerator import Generator
from models.gan import Random_GAN


def train():
    dataset_text = TextDataSet(list_files=[args.dirs.text.data], args=args, _shuffle=True)
    tfdata_train = tf.data.Dataset.from_generator(
        dataset_text, (tf.int32), (tf.TensorShape([None])))
    iter_text = tfdata_train.cache().repeat().shuffle(1000).\
        padded_batch(args.text_batch_size, ([args.max_label_len])).prefetch(buffer_size=5).\
        make_one_shot_iterator().get_next()

    tensor_global_step0 = tf.Variable(0, dtype=tf.int32, trainable=False)
    tensor_global_step1 = tf.Variable(0, dtype=tf.int32, trainable=False)

    G = G_infer = Generator(tensor_global_step0, hidden=128, num_blocks=5, args=args)
    vars_G = G.trainable_variables

    D = args.Model_D(
        tensor_global_step1,
        training=True,
        name='discriminator',
        args=args)

    gan = Random_GAN([tensor_global_step0, tensor_global_step1], G, D,
                     batch=None, unbatch=None, name='text_gan', args=args)

    size_variables()

    start_time = datetime.now()
    saver_G = tf.train.Saver(vars_G, max_to_keep=1)
    saver = tf.train.Saver(max_to_keep=15)
    summary = Summary(str(args.dir_log))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        if args.dirs.checkpoint_G:
            saver_G.restore(sess, args.dirs.checkpoint_G)

        batch_time = time()
        global_step = 0
        while global_step < 99999999:

            global_step, lr_G, lr_D = sess.run([tensor_global_step1, gan.learning_rate_G, gan.learning_rate_D])

            # untrain
            text = sess.run(iter_text)
            text_lens = get_batch_length(text)
            shape_text = text.shape
            loss_D, loss_D_res, loss_D_text, loss_gp, _ = sess.run(gan.list_train_D,
                                                          feed_dict={gan.list_pl[0]:text,
                                                                     gan.list_pl[1]:text_lens})
            loss_G, _ = sess.run(gan.list_train_G)
            # loss_D_res = - loss_G
            # loss_G = loss_G_supervise = 0.0
            # loss_D = loss_D_text = loss_gp = 0.0
            # train
            # if global_step % 5 == 0:
            # for _ in range(2):
            #     loss_supervise, shape_batch, _, _ = sess.run(G.list_run)
            # loss_G_supervise = 0
            used_time = time()-batch_time
            batch_time = time()

            if global_step % 20 == 0:
                # print('loss_G: {:.2f} loss_G_supervise: {:.2f} loss_D_res: {:.2f} loss_D_text: {:.2f} step: {}'.format(
                #        loss_G, loss_G_supervise, loss_D_res, loss_D_text, global_step))
                print('loss res|real|gp: {:.2f}|{:.2f}|{:.2f}\tbatch: {}\tlr:{:.1e}|{:.1e} {:.2f}s step: {}'.format(
                       loss_D_res, loss_D_text, loss_gp, shape_text, lr_G, lr_D, used_time, global_step))
                # summary.summary_scalar('loss_G', loss_G, global_step)
                # summary.summary_scalar('loss_D', loss_D, global_step)
                # summary.summary_scalar('lr_G', lr_G, global_step)
                # summary.summary_scalar('lr_D', lr_D, global_step)

            if global_step % args.save_step == args.save_step - 1:
                saver.save(get_session(sess), str(args.dir_checkpoint/'model'), global_step=global_step, write_meta_graph=True)

            # if global_step % args.decode_step == args.decode_step - 1:
            if global_step % args.decode_step == 1:
                samples, len_samples = sess.run(G.run_list)
                list_res_txt = array_idx2char(samples, args.idx2token, seperator=' ')
                for text_sample in list_res_txt[:10]:
                    # text_sample = args.idx2token[sample]
                    print('sampled: ', text_sample)

    logging.info('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info('enter the TRAINING phrase')
        train()

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
