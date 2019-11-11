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
import editdistance as ed

from models.utils.tools import get_session, create_embedding, size_variables
from models.utils.tfData import TFReader, readTFRecord, TFData
from utils.arguments import args
from utils.dataset import ASRDataLoader, TextDataSet
from utils.summaryTools import Summary
from utils.performanceTools import dev, decode_test
from utils.textTools import array_idx2char, array2text
from utils.tools import get_batch_length


def train():
    print('reading data form ', args.dirs.train.tfdata)
    dataReader_train = TFReader(args.dirs.train.tfdata, args=args)
    batch_train = dataReader_train.fentch_batch_bucket()
    dataReader_untrain = TFReader(args.dirs.untrain.tfdata, args=args)
    # batch_untrain = dataReader_untrain.fentch_batch(args.batch_size)
    batch_untrain = dataReader_untrain.fentch_batch_bucket()
    args.dirs.untrain.tfdata = Path(args.dirs.untrain.tfdata)
    args.data.untrain_size = TFData.read_tfdata_info(args.dirs.untrain.tfdata)['size_dataset']

    dataset_text = TextDataSet(list_files=[args.dirs.text.data], args=args, _shuffle=True)
    tfdata_train = tf.data.Dataset.from_generator(
        dataset_text, (tf.int32), (tf.TensorShape([None])))
    iter_text = tfdata_train.cache().repeat().shuffle(1000).\
        padded_batch(args.text_batch_size, ([args.max_label_len])).prefetch(buffer_size=5).\
        make_one_shot_iterator().get_next()

    feat, label = readTFRecord(args.dirs.dev.tfdata, args,
                               _shuffle=False, transform=True)
    dataloader_dev = ASRDataLoader(args.dataset_dev, args, feat, label,
                                   batch_size=args.batch_size, num_loops=1)

    # tensor_global_step = tf.train.get_or_create_global_step()
    tensor_global_step0 = tf.Variable(0, dtype=tf.int32, trainable=False)
    tensor_global_step1 = tf.Variable(0, dtype=tf.int32, trainable=False)
    tensor_global_step2 = tf.Variable(0, dtype=tf.int32, trainable=False)

    G = args.Model(
        tensor_global_step2,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        batch=batch_train,
        # batch=batch_untrain,
        training=True,
        args=args)

    G_infer = args.Model(
        tensor_global_step2,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        training=False,
        args=args)
    vars_G = G.variables()

    D = args.Model_D(
        tensor_global_step0,
        training=True,
        name='discriminator',
        args=args)

    gan = args.GAN([tensor_global_step0, tensor_global_step1], G, D,
                   batch=batch_untrain, name='GAN', args=args)

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
        dataloader_dev.sess = sess
        if args.dirs.checkpoint_G:
            checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_G)
            saver_G.restore(sess, checkpoint)

            # cer, wer = dev(
            #     step=0,
            #     dataloader=dataloader_dev,
            #     model=G_infer,
            #     sess=sess,
            #     unit=args.data.unit,
            #     idx2token=args.idx2token,
            #     eos_idx=args.eos_idx,
            #     min_idx=0,
            #     max_idx=args.dim_output-1)
            # decode_test(
            #     step=0,
            #     sample=args.dataset_test[10],
            #     model=G_infer,
            #     sess=sess,
            #     unit=args.data.unit,
            #     idx2token=args.idx2token,
            #     eos_idx=None,
            #     min_idx=0,
            #     max_idx=None)

        batch_time = time()
        num_processed = 0
        progress = 0
        while progress < args.num_epochs:

            global_step, lr_G, lr_D = sess.run([tensor_global_step2, gan.learning_rate_G, gan.learning_rate_D])

            # untrain
            text = sess.run(iter_text)
            text_lens = get_batch_length(text)
            shape_feature = sess.run(gan.list_info[0])
            shape_text = text.shape
            loss_G, _ = sess.run(gan.list_train_G)
            loss_D, _ = sess.run(gan.list_train_D,
                                 feed_dict={gan.list_pl[0]:text,
                                            gan.list_pl[1]:text_lens})
            # loss_G = 0.0
            # loss_D = 0.0
            # train
            # if global_step % 5 == 0:
            for _ in range(1):
                loss_supervise, shape_batch, _, _ = sess.run(G.list_run)
            # loss_supervise = 0

            num_processed += shape_feature[0]
            used_time = time()-batch_time
            batch_time = time()
            progress = num_processed/args.data.untrain_size

            if global_step % 50 == 0:
                logging.info('loss_supervise: {:.3f}, loss: {:.3f}|{:.3f}\tbatch: {}|{} lr:{:.6f}|{:.6f} time:{:.2f}s {:.3f}% step: {}'.format(
                              loss_supervise, loss_G, loss_D, shape_feature, shape_text, lr_G, lr_D, used_time, progress*100.0, global_step))
                # summary.summary_scalar('loss_G', loss_G, global_step)
                # summary.summary_scalar('loss_D', loss_D, global_step)
                # summary.summary_scalar('lr_G', lr_G, global_step)
                # summary.summary_scalar('lr_D', lr_D, global_step)

            if global_step % args.save_step == args.save_step - 1:
                saver.save(get_session(sess), str(args.dir_checkpoint/'model'), global_step=global_step, write_meta_graph=True)

            if global_step % args.dev_step == args.dev_step - 1:
            # if global_step % args.dev_step == 0:
                cer, wer = dev(
                    step=global_step,
                    dataloader=dataloader_dev,
                    model=G_infer,
                    sess=sess,
                    unit=args.data.unit,
                    idx2token=args.idx2token,
                    eos_idx=args.eos_idx,
                    min_idx=0,
                    max_idx=args.dim_output-1)
                # summary.summary_scalar('dev_cer', cer, global_step)
                # summary.summary_scalar('dev_wer', wer, global_step)

            if global_step % args.decode_step == args.decode_step - 1:
                decode_test(
                    step=global_step,
                    sample=args.dataset_test[10],
                    model=G_infer,
                    sess=sess,
                    unit=args.data.unit,
                    idx2token=args.idx2token,
                    eos_idx=None,
                    min_idx=0,
                    max_idx=None)

    logging.info('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def infer():
    tensor_global_step = tf.train.get_or_create_global_step()

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        training=False,
        args=args)

    dataset_dev = args.dataset_test if args.dataset_test else args.dataset_dev

    saver = tf.train.Saver(max_to_keep=40)
    size_variables()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
        saver.restore(sess, checkpoint)

        total_cer_dist = 0
        total_cer_len = 0
        total_wer_dist = 0
        total_wer_len = 0
        with open(args.dir_model.name+'_decode.txt', 'w') as fw:
            for i, sample in enumerate(dataset_dev):
                if not sample:
                    continue
                dict_feed = {model_infer.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                             model_infer.list_pl[1]: np.array([len(sample['feature'])])}
                sample_id, shape_batch, _ = sess.run(model_infer.list_run, feed_dict=dict_feed)
                # decoded, sample_id, decoded_sparse = sess.run(model_infer.list_run, feed_dict=dict_feed)
                res_txt = array2text(sample_id[0], args.data.unit, args.idx2token, eos_idx=args.eos_idx, min_idx=0, max_idx=args.dim_output-1)
                # align_txt = array2text(alignment[0], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1)
                ref_txt = array2text(sample['label'], args.data.unit, args.idx2token, eos_idx=args.eos_idx, min_idx=0, max_idx=args.dim_output-1)

                list_res_char = list(res_txt)
                list_ref_char = list(ref_txt)
                list_res_word = res_txt.split()
                list_ref_word = ref_txt.split()
                cer_dist = ed.eval(list_res_char, list_ref_char)
                cer_len = len(list_ref_char)
                wer_dist = ed.eval(list_res_word, list_ref_word)
                wer_len = len(list_ref_word)
                total_cer_dist += cer_dist
                total_cer_len += cer_len
                total_wer_dist += wer_dist
                total_wer_len += wer_len
                if cer_len == 0:
                    cer_len = 1000
                    wer_len = 1000
                if wer_dist/wer_len > 0:
                    fw.write('id:\t{} \nres:\t{}\nref:\t{}\n\n'.format(sample['id'], res_txt, ref_txt))
                sys.stdout.write('\rcurrent cer: {:.3f}, wer: {:.3f};\tall cer {:.3f}, wer: {:.3f} {}/{} {:.2f}%'.format(
                    cer_dist/cer_len, wer_dist/wer_len, total_cer_dist/total_cer_len,
                    total_wer_dist/total_wer_len, i, len(dataset_dev), i/len(dataset_dev)*100))
                sys.stdout.flush()
        logging.info('dev CER {:.3f}:  WER: {:.3f}'.format(total_cer_dist/total_cer_len, total_wer_dist/total_wer_len))


def infer_lm():
    tensor_global_step = tf.train.get_or_create_global_step()
    dataset_dev = args.dataset_test if args.dataset_test else args.dataset_dev

    model_lm = args.Model_LM(
        tensor_global_step,
        training=False,
        args=args.args_lm)

    args.lm_obj = model_lm
    saver_lm = tf.train.Saver(model_lm.variables())

    args.top_scope = tf.get_variable_scope()   # top-level scope
    args.lm_scope = model_lm.decoder.scope

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        training=False,
        args=args)

    saver = tf.train.Saver(model_infer.variables())

    size_variables()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
        checkpoint_lm = tf.train.latest_checkpoint(args.dirs.lm_checkpoint)
        saver.restore(sess, checkpoint)
        saver_lm.restore(sess, checkpoint_lm)

        total_cer_dist = 0
        total_cer_len = 0
        total_wer_dist = 0
        total_wer_len = 0
        with open(args.dir_model.name+'_decode.txt', 'w') as fw:
        # with open('/mnt/lustre/xushuang/easton/projects/asr-tf/exp/aishell/lm_acc.txt', 'w') as fw:
            for sample in tqdm(dataset_dev):
                if not sample:
                    continue
                dict_feed = {model_infer.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                             model_infer.list_pl[1]: np.array([len(sample['feature'])])}
                sample_id, shape_batch, beam_decoded = sess.run(model_infer.list_run, feed_dict=dict_feed)
                # decoded, sample_id, decoded_sparse = sess.run(model_infer.list_run, feed_dict=dict_feed)
                res_txt = array2text(sample_id[0], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1)
                ref_txt = array2text(sample['label'], args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1)

                list_res_char = list(res_txt)
                list_ref_char = list(ref_txt)
                list_res_word = res_txt.split()
                list_ref_word = ref_txt.split()
                cer_dist = ed.eval(list_res_char, list_ref_char)
                cer_len = len(list_ref_char)
                wer_dist = ed.eval(list_res_word, list_ref_word)
                wer_len = len(list_ref_word)
                total_cer_dist += cer_dist
                total_cer_len += cer_len
                total_wer_dist += wer_dist
                total_wer_len += wer_len
                if cer_len == 0:
                    cer_len = 1000
                    wer_len = 1000
                if wer_dist/wer_len > 0:
                    print('ref  ' , ref_txt)
                    for i, decoded, score, rerank_score in zip(range(10), beam_decoded[0][0], beam_decoded[1][0], beam_decoded[2][0]):
                        candidate = array2text(decoded, args.data.unit, args.idx2token, min_idx=0, max_idx=args.dim_output-1)
                        print('res' ,i , candidate, score, rerank_score)
                        fw.write('res: {}; ref: {}\n'.format(candidate, ref_txt))
                    fw.write('id:\t{} \nres:\t{}\nref:\t{}\n\n'.format(sample['id'], res_txt, ref_txt))
                logging.info('current cer: {:.3f}, wer: {:.3f};\tall cer {:.3f}, wer: {:.3f}'.format(
                    cer_dist/cer_len, wer_dist/wer_len, total_cer_dist/total_cer_len, total_wer_dist/total_wer_len))
        logging.info('dev CER {:.3f}:  WER: {:.3f}'.format(total_cer_dist/total_cer_len, total_wer_dist/total_wer_len))


def save(gpu, name=0):
    from utils.IO import store_2d

    tensor_global_step = tf.train.get_or_create_global_step()

    embed = create_embedding(
        name='embedding_table',
        size_vocab=args.dim_output,
        size_embedding=args.model.decoder.size_embedding)

    model_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        embed_table_decoder=embed,
        training=False,
        args=args)

    dataset_dev = args.dataset_test if args.dataset_test else args.dataset_dev

    saver = tf.train.Saver(max_to_keep=15)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        checkpoint = tf.train.latest_checkpoint(args.dirs.checkpoint_init)
        saver.restore(sess, checkpoint)
        if not name:
            name = args.dir_model.name
        with open('outputs/distribution_'+name+'.bin', 'wb') as fw, \
            open('outputs/res_ref_'+name+'.txt', 'w') as fw2:
        # with open('dev_sample.txt', 'w') as fw:
            for i, sample in enumerate(tqdm(dataset_dev)):
            # sample = dataset_dev[0]
                if not sample: continue
                dict_feed = {model_infer.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                             model_infer.list_pl[1]: np.array([len(sample['feature'])])}
                decoded, _, distribution = sess.run(model_infer.list_run, feed_dict=dict_feed)
                store_2d(distribution[0], fw)
                # pickle.dump(distribution[0], fw)
                # [fw.write(' '.join(map(str, line))+'\n') for line in distribution[0]]
                result_txt = array_idx2char(decoded, args.idx2token, seperator=' ')
                ref_txt = array_idx2char(sample['label'], args.idx2token, seperator=' ')
                fw2.write('{}_res: {}\n{}_ref: {}\n'.format(i, result_txt[0], i, ref_txt))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    if param.mode == 'infer':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info('enter the INFERING phrase')
        infer()

    elif param.mode == 'infer_lm':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info('enter the INFERING phrase')
        infer_lm()

    elif param.mode == 'save':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info('enter the SAVING phrase')
        save(gpu=param.gpu, name=param.name)

    elif param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logging.info('enter the TRAINING phrase')
        train()

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
