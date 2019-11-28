#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import logging
import tensorflow as tf
import numpy as np

from models.utils.tools import get_session, size_variables
from utils.arguments import args
from utils.dataset import TextDataSet
from utils.summaryTools import Summary
from utils.textTools import array_idx2char, array2text, batch_cer
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
        padded_batch(args.num_supervised, ([args.max_label_len])).prefetch(buffer_size=5).\
        make_one_shot_iterator().get_next()

    dataset_text_dev = TextDataSet(list_files=[args.dirs.text.dev], args=args, _shuffle=False)
    tfdata_dev = tf.data.Dataset.from_generator(
        dataset_text_dev, (tf.int32), (tf.TensorShape([None])))
    tfdata_dev = tfdata_dev.cache().\
        padded_batch(args.text_batch_size, ([args.max_label_len])).prefetch(buffer_size=5).\
        make_initializable_iterator()
    iter_text_dev = tfdata_dev.get_next()

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
            print('revocer G from', args.dirs.checkpoint_G)

        # np.random.seed(0)
        text_supervise = sess.run(iter_supervise)
        text_len_supervise = get_batch_length(text_supervise)
        feature_supervise, feature_len_supervise = int2vector(text_supervise, text_len_supervise, hidden_size=args.model.dim_input , uprate=args.uprate)
        feature_supervise += np.random.randn(*feature_supervise.shape)/args.noise
        batch_time = time()
        global_step = 0

        # for _ in range(100):
        #     np.random.seed(1)
        #     text_G = sess.run(iter_text)
        #     text_lens_G = get_batch_length(text_G)
        #     feature_text, text_lens_G = int2vector(text_G, text_lens_G, hidden_size=args.model.dim_input , uprate=args.uprate)
        #     feature_text += np.random.randn(*feature_text.shape)/args.noise
        #     loss_G, loss_G_supervise, _ = sess.run(gan.list_train_G,
        #          feed_dict={gan.list_G_pl[0]:feature_text,
        #                     gan.list_G_pl[1]:text_lens_G,
        #                     gan.list_G_pl[2]:feature_supervise,
        #                     gan.list_G_pl[3]:feature_len_supervise,
        #                     gan.list_G_pl[4]:text_supervise,
        #                     gan.list_G_pl[5]:text_len_supervise})
        #     saver.save(get_session(sess), str(args.dir_checkpoint/'model'), global_step=0, write_meta_graph=True)

        while global_step < 99999999:

            # global_step, lr_G, lr_D = sess.run([tensor_global_step0, gan.learning_rate_G, gan.learning_rate_D])
            global_step, lr_G = sess.run([tensor_global_step0, G.learning_rate])

            text_supervise = sess.run(iter_supervise)
            text_len_supervise = get_batch_length(text_supervise)
            feature_supervise, feature_len_supervise = int2vector(text_supervise, text_len_supervise, hidden_size=args.model.dim_input , uprate=args.uprate)
            feature_supervise += np.random.randn(*feature_supervise.shape)/args.noise

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
            feature_text, text_lens_G = int2vector(text_G, text_lens_G, hidden_size=args.model.dim_input , uprate=args.uprate)
            feature_text += np.random.randn(*feature_text.shape)/args.noise
            loss_G, loss_G_supervise, _ = sess.run(gan.list_train_G,
                 feed_dict={gan.list_G_pl[0]:feature_text,
                            gan.list_G_pl[1]:text_lens_G,
                            gan.list_G_pl[2]:feature_supervise,
                            gan.list_G_pl[3]:feature_len_supervise,
                            gan.list_G_pl[4]:text_supervise,
                            gan.list_G_pl[5]:text_len_supervise})
            # loss_G = loss_G_supervise = 0

            # discriminator input
            # for _ in range(3):
            #     # np.random.seed(2)
            #     text_G = sess.run(iter_text)
            #     text_lens_G = get_batch_length(text_G)
            #     feature_G, feature_lens_G = int2vector(text_G, text_lens_G, hidden_size=args.model.dim_input, uprate=args.uprate)
            #     feature_G += np.random.randn(*feature_G.shape)/args.noise
            #
            #     text_D = sess.run(iter_text)
            #     text_lens_D = get_batch_length(text_D)
            #     shape_text = text_D.shape
            #     loss_D, loss_D_res, loss_D_text, loss_gp, _ = sess.run(gan.list_train_D,
            #             feed_dict={gan.list_D_pl[0]:text_D,
            #                        gan.list_D_pl[1]:text_lens_D,
            #                        gan.list_G_pl[0]:feature_G,
            #                        gan.list_G_pl[1]:feature_lens_G})
            loss_D_res = loss_D_text = loss_gp = 0
            shape_text = [0,0,0]

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
                print('saved model as',  str(args.dir_checkpoint) + '/model-' + str(global_step))
            # if global_step % args.dev_step == args.dev_step - 1:
            if global_step % args.dev_step == 1:
                text_G_dev = dev(iter_text_dev, tfdata_dev, dataset_text_dev, sess, G_infer)

            # if global_step % args.decode_step == args.decode_step - 1:
            if global_step % args.decode_step == args.decode_step - 1:
                decode(text_G_dev, tfdata_dev, sess, G_infer)

    logging.info('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def dev(iter_text_dev, tfdata_dev, dataset_text_dev, sess, G_infer):

    list_res = []; list_ref = []; list_length = []; length_rate = []
    process = 0

    sess.run(tfdata_dev.initializer)
    while 1:
        try:
            np.random.seed(process)
            text_G_dev = sess.run(iter_text_dev)
            text_lens_G_dev = get_batch_length(text_G_dev)
            feature_dev, feature_lens_G_dev = int2vector(text_G_dev, text_lens_G_dev, hidden_size=args.model.dim_input, uprate=args.uprate)
            feature_dev += np.random.randn(*feature_dev.shape)/args.noise
            _, samples, len_samples = sess.run(G_infer.run_list,
                                                feed_dict={G_infer.list_pl[0]:feature_dev,
                                                           G_infer.list_pl[1]:feature_lens_G_dev})
            list_res.extend(list(samples))
            list_ref.extend(list(text_G_dev))
            list_length.extend(list(text_lens_G_dev))
            length_rate.extend(list(len_samples/text_lens_G_dev))
            process += len(text_G_dev)
            # if process // 5000 == 0:
            #     print('processed {} samples'.format(process))
        except tf.errors.OutOfRangeError:
            break

    print('dev len_res/len_ref : {:.2f}  process {} samples.'.format(np.mean(length_rate), process))
    if args.uprate == 1.0:
        acc = accuracy(list_res, list_ref, list_length)
        print('dev accuracy: {:.2f}%'.format(acc*100.0))
    else:
        cer_dist, cer_len = batch_cer(
            result=list_res,
            reference=list_ref)
        cer = cer_dist / cer_len
        print('dev cer: {:.2f}%'.format(cer*100.0))

    return text_G_dev


def decode(text_G_dev, iter_text_dev, sess, G_infer):
    text_lens_G_dev = get_batch_length(text_G_dev)
    feature_dev, feature_lens_G_dev = int2vector(text_G_dev, text_lens_G_dev, hidden_size=args.model.dim_input , uprate=args.uprate)
    feature_dev += np.random.randn(*feature_dev.shape)/args.noise

    logits, samples, len_samples = sess.run(G_infer.run_list,
                                    feed_dict={G_infer.list_pl[0]:feature_dev,
                                               G_infer.list_pl[1]:feature_lens_G_dev})
    list_res_txt = array_idx2char(samples, args.idx2token, seperator=' ')
    list_ref_txt = array_idx2char(text_G_dev, args.idx2token, seperator=' ')

    for res, ref in zip(list_res_txt[:5], list_ref_txt):
        # text_sample = args.idx2token[sample]
        print('sampled: ', res)
        print('label  : ', ref)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=None)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    if param.name:
        args.dir_exps = args.dir_exps / param.name
        args.dir_log = args.dir_exps / 'log'
        args.dir_checkpoint = args.dir_exps / 'checkpoint'
        if not args.dir_exps.is_dir(): args.dir_exps.mkdir()
        if not args.dir_log.is_dir(): args.dir_log.mkdir()
        if not args.dir_checkpoint.is_dir(): args.dir_checkpoint.mkdir()

    if param.gpu:
        print('CUDA_VISIBLE_DEVICES: ', args.gpus)
        args.gpus = param.gpu

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info('enter the TRAINING phrase')
        train()

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
