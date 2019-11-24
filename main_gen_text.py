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
from utils.tools import get_batch_length, int2vector
from utils.performanceTools import accuracy
from models.generator.baseGenerator import Generator
from models.gan import Conditional_GAN

def train():
    dataset_text = TextDataSet(list_files=[args.dirs.text.data], args=args, _shuffle=True)
    tfdata_train = tf.data.Dataset.from_generator(
        dataset_text, (tf.int32), (tf.TensorShape([None])))
    iter_text = tfdata_train.cache().repeat().shuffle(10000).\
        padded_batch(args.text_batch_size, ([args.max_label_len])).prefetch(buffer_size=5).\
        make_one_shot_iterator().get_next()

    dataset_text_supervise = TextDataSet(list_files=[args.dirs.text.supervise], args=args, _shuffle=True)
    tfdata_supervise = tf.data.Dataset.from_generator(
        dataset_text_supervise, (tf.int32), (tf.TensorShape([None])))
    iter_supervise = tfdata_supervise.cache().repeat().shuffle(100).\
        padded_batch(10, ([args.max_label_len])).prefetch(buffer_size=5).\
        make_one_shot_iterator().get_next()

    dataset_text_dev = TextDataSet(list_files=[args.dirs.text.dev], args=args, _shuffle=False)
    tfdata_dev = tf.data.Dataset.from_generator(
        dataset_text_dev, (tf.int32), (tf.TensorShape([None])))
    iter_text_dev = tfdata_dev.cache().repeat().\
        padded_batch(args.text_batch_size, ([args.max_label_len])).prefetch(buffer_size=5).\
        make_one_shot_iterator().get_next()

    tensor_global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
    tensor_global_step0 = tf.Variable(0, dtype=tf.int32, trainable=False)
    tensor_global_step1 = tf.Variable(0, dtype=tf.int32, trainable=False)

    G = Generator(tensor_global_step,
                  hidden=args.model.hidden_size,
                  num_blocks=args.model.num_blocks,
                  training=True,
                  args=args)
    G_infer = Generator(tensor_global_step,
                        hidden=args.model.hidden_size,
                        num_blocks=args.model.num_blocks,
                        training=False,
                        args=args)
    vars_G = G.trainable_variables

    D = args.Model_D(tensor_global_step1,
                     training=True,
                     name='discriminator',
                     args=args)

    gan = Conditional_GAN([tensor_global_step0, tensor_global_step1], G, D,
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

        text_supervise = sess.run(iter_supervise)
        text_len_supervise = get_batch_length(text_supervise)
        feature_text_supervise = int2vector(text_supervise)
        feature_text_supervise += np.random.randn(*feature_text_supervise.shape)/args.noise

        batch_time = time()
        global_step = 0
        while global_step < 99999999:

            # global_step, lr_G, lr_D = sess.run([tensor_global_step0, gan.learning_rate_G, gan.learning_rate_D])
            global_step, lr_G = sess.run([tensor_global_step0, G.learning_rate])

            # text_supervise = sess.run(iter_supervise)
            # text_len_supervise = get_batch_length(text_supervise)
            # feature_text_supervise = int2vector(text_supervise)
            # feature_text_supervise += np.random.randn(*feature_text_supervise.shape)/args.noise
            # feature_text = feature_text_supervise

            # supervise
            # for _ in range(1):
            #     loss_G_supervise, _ = sess.run(G.run_list,
            #                          feed_dict={G.list_pl[0]:feature_text_supervise,
            #                                     G.list_pl[1]:text_len_supervise,
            #                                     G.list_pl[2]:text_supervise,
            #                                     G.list_pl[3]:text_len_supervise})

            # generator input
            text_G = sess.run(iter_text)
            text_lens_G = get_batch_length(text_G)
            feature_text = int2vector(text_G)
            feature_text += np.random.randn(*feature_text.shape)/args.noise
            loss_G, loss_G_supervise, _ = sess.run(gan.list_train_G,
                 feed_dict={gan.list_G_pl[0]:feature_text,
                            gan.list_G_pl[1]:text_lens_G,
                            gan.list_G_pl[2]:feature_text_supervise,
                            gan.list_G_pl[3]:text_len_supervise,
                            gan.list_G_pl[4]:text_supervise,
                            gan.list_G_pl[5]:text_len_supervise})
            # loss_G = loss_G_supervise = 0

            # discriminator input
            for _ in range(3):
                text_G = sess.run(iter_text)
                text_lens_G = get_batch_length(text_G)
                feature_text = int2vector(text_G)
                feature_text += np.random.randn(*feature_text.shape)/args.noise

                text_D = sess.run(iter_text)
                text_lens_D = get_batch_length(text_D)
                shape_text = text_D.shape
                loss_D, loss_D_res, loss_D_text, loss_gp, _ = sess.run(gan.list_train_D,
                        feed_dict={gan.list_D_pl[0]:text_D,
                                   gan.list_D_pl[1]:text_lens_D,
                                   gan.list_G_pl[0]:feature_text,
                                   gan.list_G_pl[1]:text_lens_G})
            # loss_D_res = loss_D_text = loss_gp = 0
            # shape_text = [0,0,0]

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

            if global_step % 10 == 0:
                # print('loss_G: {:.2f} loss_G_supervise: {:.2f} loss_D_res: {:.2f} loss_D_text: {:.2f} step: {}'.format(
                #        loss_G, loss_G_supervise, loss_D_res, loss_D_text, global_step))
                print('loss_G_supervise: {:.2f} loss res|real|gp: {:.2f}|{:.2f}|{:.2f}\tbatch: {}\tlr:{:.1e} {:.2f}s step: {}'.format(
                       loss_G_supervise, loss_D_res, loss_D_text, loss_gp, shape_text, lr_G, used_time, global_step))
                # summary.summary_scalar('loss_G', loss_G, global_step)
                # summary.summary_scalar('loss_D', loss_D, global_step)
                # summary.summary_scalar('lr_G', lr_G, global_step)
                # summary.summary_scalar('lr_D', lr_D, global_step)

            if global_step % args.save_step == args.save_step - 1:
                saver.save(get_session(sess), str(args.dir_checkpoint/'model'), global_step=global_step, write_meta_graph=True)

            if global_step % args.dev_step == args.dev_step - 1:
                text_G_dev = sess.run(iter_text_dev)
                text_lens_G_dev = get_batch_length(text_G_dev)
                feature_text_dev = int2vector(text_G_dev)
                feature_text_dev += np.random.randn(*feature_text_dev.shape)/args.noise

                list_res = []; list_ref = []; list_length = []
                process = 0
                while process < len(dataset_text_dev):
                    _, samples, len_samples = sess.run(G_infer.run_list,
                                                        feed_dict={G_infer.list_pl[0]:feature_text_dev,
                                                                   G_infer.list_pl[1]:text_lens_G_dev})
                    list_res.append(samples)
                    list_ref.append(text_G_dev)
                    list_length.append(text_lens_G_dev)
                    process += len(text_G_dev)

                all_res = np.concatenate(list_res, 0)
                all_ref = np.concatenate(list_ref, 0)
                all_length = np.concatenate(list_length, 0)

                acc = accuracy(all_res, all_ref, all_length)
                print('dev accuracy: {:.2f}%'.format(acc*100.0))

            # if global_step % args.decode_step == args.decode_step - 1:
            if global_step % args.decode_step == args.decode_step - 1:
                logits, samples, len_samples = sess.run(G_infer.run_list,
                                                feed_dict={G_infer.list_pl[0]:feature_text_dev,
                                                           G_infer.list_pl[1]:text_lens_G_dev})

                list_res_txt = array_idx2char(samples, args.idx2token, seperator=' ')
                list_ref_txt = array_idx2char(text_G_dev, args.idx2token, seperator=' ')

                for res, ref in zip(list_res_txt[:5], list_ref_txt):
                    # text_sample = args.idx2token[sample]
                    print('sampled: ', res)
                    print('label  : ', ref)

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
