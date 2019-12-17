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
from models.utils.tfData import TFReader, readTFRecord_multilabel, TFData
from utils.arguments import args
from utils.dataset import ASR_phone_char_ArkDataSet, TextDataSet
from utils.summaryTools import Summary
# from utils.performanceTools import dev, decode_test
from utils.textTools import array_idx2char, array2text, batch_wer, batch_cer
from utils.tools import get_batch_length


def train():
    print('reading data form ', args.dirs.train.tfdata)
    dataReader_train = TFReader(args.dirs.train.tfdata, args=args)
    batch_train = dataReader_train.fentch_batch_bucket()
    # dataReader_untrain = TFReader(args.dirs.untrain.tfdata, args=args)
    # batch_untrain = dataReader_untrain.fentch_batch(args.batch_size)
    # batch_untrain = dataReader_untrain.fentch_batch_bucket()
    # args.dirs.untrain.tfdata = Path(args.dirs.untrain.tfdata)
    # args.data.untrain_size = TFData.read_tfdata_info(args.dirs.untrain.tfdata)['size_dataset']

    dataset_text = TextDataSet(list_files=[args.dirs.text.data], args=args, _shuffle=True)
    tfdata_train = tf.data.Dataset.from_generator(
        dataset_text, (tf.int32), (tf.TensorShape([None])))
    iter_text = tfdata_train.cache().repeat().shuffle(1000).\
        padded_batch(args.text_batch_size, ([args.max_label_len])).prefetch(buffer_size=5).\
        make_one_shot_iterator().get_next()

    feat, label = readTFRecord_multilabel(
        args.dirs.dev.tfdata, args, _shuffle=False, transform=True)
    dataloader_dev = ASR_phone_char_ArkDataSet(
        args.dataset_dev, args, feat, label, batch_size=args.batch_size, num_loops=1)

    tensor_global_step = tf.train.get_or_create_global_step()
    tensor_global_step0 = tf.Variable(0, dtype=tf.int32, trainable=False)
    tensor_global_step1 = tf.Variable(0, dtype=tf.int32, trainable=False)

    G = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        batch=batch_train,
        training=True,
        args=args)

    G_infer = args.Model(
        tensor_global_step,
        encoder=args.model.encoder.type,
        decoder=args.model.decoder.type,
        training=False,
        args=args)
    vars_G = G.trainable_variables()
    vars_G_en = G.trainable_variables('Ectc_Docd/encoder')
    vars_G_ctc = G.trainable_variables('Ectc_Docd/ctc_decoder')
    # vars_G_ocd = G.trainable_variables('Ectc_Docd/ocd_decoder')

    D = args.Model_D(
        tensor_global_step1,
        training=True,
        name='discriminator',
        args=args)

    # gan = args.GAN([tensor_global_step0, tensor_global_step1], G, D,
    #                batch=batch_train, unbatch=batch_untrain, name='GAN', args=args)

    size_variables()

    start_time = datetime.now()
    saver_G = tf.train.Saver(vars_G, max_to_keep=10)
    saver_G_en = tf.train.Saver(vars_G_en + vars_G_ctc, max_to_keep=10)
    saver = tf.train.Saver(max_to_keep=15)
    summary = Summary(str(args.dir_log))

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    config.log_device_placement = False
    with tf.train.MonitoredTrainingSession(config=config) as sess:
        dataloader_dev.sess = sess
        if args.dirs.checkpoint_G:
            saver_G.restore(sess, args.dirs.checkpoint_G)
        if args.dirs.checkpoint_G_en:
            saver_G_en.restore(sess, args.dirs.checkpoint_G_en)

        # for i in range(500):
        #     ctc_loss, *_ = sess.run(G.list_run)
        #     if i % 100 == 0:
        #         print('i: {} ctc_loss: {:.2f}'.format(i, ctc_loss))
        # saver_G_en.save(get_session(sess), str(args.dir_checkpoint/'model_G_en'), global_step=0, write_meta_graph=True)
        # print('saved model in',  str(args.dir_checkpoint)+'/model_G_en-0')

        batch_time = time()
        num_processed = 0
        num_processed_unbatch = 0
        progress = 0
        while progress < args.num_epochs:

            global_step, lr = sess.run([tensor_global_step, G.learning_rate])
            # global_step, lr_G, lr_D = sess.run([tensor_global_step0, gan.learning_rate_G, gan.learning_rate_D])

            # supervised training
            loss_G, shape_batch, _, (ctc_loss, ce_loss, *_) = sess.run(G.list_run)

            # untrain
            # for _ in range(5):
            #     text = sess.run(iter_text)
            #     text_lens = get_batch_length(text)
            #     shape_text = text.shape
            #     loss_D, loss_D_res, loss_D_text, loss_gp, _ = sess.run(gan.list_train_D,
            #                                               feed_dict={gan.list_pl[0]:text,
            #                                                          gan.list_pl[1]:text_lens})
            #     loss_D=loss_D_res=loss_D_text=loss_gp=0
            # (loss_G, ctc_loss, ce_loss, _), (shape_batch, shape_unbatch) = \
            #     sess.run([gan.list_train_G, gan.list_feature_shape])

            num_processed += shape_batch[0]
            # num_processed_unbatch += shape_unbatch[0]
            used_time = time()-batch_time
            batch_time = time()
            progress = num_processed/args.data.train_size
            progress_unbatch = num_processed_unbatch/args.data.untrain_size

            if global_step % 40 == 0:
                print('ctc_loss: {:.2f}, ce_loss: {:.2f} batch: {} lr:{:.1e} {:.2f}s {:.3f}% step: {}'.format(
                     np.mean(ctc_loss), np.mean(ce_loss), shape_batch, lr, used_time, progress*100, global_step))
                # print('ctc|ce loss: {:.2f}|{:.2f}, loss res|real|gp: {:.2f}|{:.2f}|{:.2f}\tlr:{:.1e}|{:.1e} {:.2f}s {:.3f}% step: {}'.format(
                #        np.mean(ctc_loss), np.mean(ce_loss), loss_D_res, loss_D_text, loss_gp, lr_G, lr_D, used_time, progress*100, global_step))
                # summary.summary_scalar('loss_G', loss_G, global_step)
                # summary.summary_scalar('loss_D', loss_D, global_step)
                # summary.summary_scalar('lr_G', lr_G, global_step)
                # summary.summary_scalar('lr_D', lr_D, global_step)

            if global_step % args.save_step == args.save_step - 1:
                # saver.save(get_session(sess), str(args.dir_checkpoint/'model_ce'), global_step=global_step, write_meta_graph=True)
                # print('saved model in',  str(args.dir_checkpoint)+'/model_ce-'+str(global_step))
                saver_G.save(get_session(sess), str(args.dir_checkpoint/'model_G'), global_step=global_step, write_meta_graph=True)
                print('saved G in',  str(args.dir_checkpoint)+'/model_G-'+str(global_step))
                # saver_G_en.save(get_session(sess), str(args.dir_checkpoint/'model_G_en'), global_step=global_step, write_meta_graph=True)
                # print('saved model in',  str(args.dir_checkpoint)+'/model_G_en-'+str(global_step))

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
                summary.summary_scalar('dev_cer', cer, global_step)
                summary.summary_scalar('dev_wer', wer, global_step)

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


def dev(step, dataloader, model, sess, unit, idx2token, eos_idx=None, min_idx=0, max_idx=None):
    start_time = time()
    batch_time = time()
    processed = 0

    total_cer_ctc_dist = 0
    total_cer_dist = 0
    total_cer_len = 0

    total_wer_ctc_dist = 0
    total_wer_dist = 0
    total_wer_len = 0

    for batch in dataloader:
        if not batch: continue
        dict_feed = {model.list_pl[0]: batch[0],
                     model.list_pl[1]: batch[2]}
        (decoded_ctc, decoded), shape_batch, _ = sess.run(model.list_run, feed_dict=dict_feed)
        # import pdb; pdb.set_trace()

        batch_cer_ctc_dist, batch_cer_len = batch_cer(
            result=decoded_ctc,
            reference=batch[1],
            eos_idx=eos_idx,
            min_idx=min_idx,
            max_idx=max_idx)
        batch_cer_dist, batch_cer_len = batch_cer(
            result=decoded,
            reference=batch[1],
            eos_idx=eos_idx,
            min_idx=min_idx,
            max_idx=max_idx)
        _cer_ctc = batch_cer_ctc_dist/batch_cer_len
        _cer = batch_cer_dist/batch_cer_len
        total_cer_ctc_dist += batch_cer_ctc_dist
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len

        batch_wer_ctc_dist, batch_wer_len = batch_wer(
            result=decoded_ctc,
            reference=batch[1],
            idx2token=idx2token,
            unit=unit,
            eos_idx=eos_idx,
            min_idx=min_idx,
            max_idx=max_idx)
        batch_wer_dist, batch_wer_len = batch_wer(
            result=decoded,
            reference=batch[1],
            idx2token=idx2token,
            unit=unit,
            eos_idx=eos_idx,
            min_idx=min_idx,
            max_idx=max_idx)
        _wer_ctc = batch_wer_ctc_dist/batch_wer_len
        _wer = batch_wer_dist/batch_wer_len
        total_wer_ctc_dist += batch_wer_ctc_dist
        total_wer_dist += batch_wer_dist
        total_wer_len += batch_wer_len

        used_time = time()-batch_time
        batch_time = time()
        processed += shape_batch[0]
        progress = processed/len(dataloader)
        sys.stdout.write('\rbatch CTC cer: {:.3f}\twer: {:.3f} \tbatch cer: {:.3f}\twer: {:.3f} batch: {}\t time:{:.2f}s {:.3f}%'.format(
                      _cer_ctc, _wer_ctc, _cer, _wer, shape_batch, used_time, progress*100.0))
        sys.stdout.flush()

    used_time = time() - start_time
    cer_ctc = total_cer_ctc_dist/total_cer_len
    wer_ctc = total_wer_ctc_dist/total_wer_len
    cer = total_cer_dist/total_cer_len
    wer = total_wer_dist/total_wer_len
    logging.warning('\n=====dev info, total used time {:.2f}h==== \nCTC WER: {:.4f}\tWER: {:.4f}\ntotal_wer_len: {}'.format(
                 used_time/3600, wer_ctc, wer, total_wer_len))

    return cer, wer


def decode_test(step, sample, model, sess, unit, idx2token, eos_idx=None, min_idx=0, max_idx=None):
    # sample = dataset_dev[0]
    dict_feed = {model.list_pl[0]: np.expand_dims(sample['feature'], axis=0),
                 model.list_pl[1]: np.array([len(sample['feature'])])}
    (decoded_ctc, decoded), shape_sample, _ = sess.run(model.list_run, feed_dict=dict_feed)

    res_ctc_txt = array2text(decoded_ctc[0], unit, idx2token, eos_idx, min_idx, max_idx)
    res_txt = array2text(decoded[0], unit, idx2token, eos_idx, min_idx, max_idx)
    ref_txt = array2text(sample['label'], unit, idx2token, eos_idx, min_idx, max_idx)

    logging.warning('length: {}, \nres_ctc: \n{}\nres: \n{}\nref: \n{}'.format(
        shape_sample[1], res_ctc_txt, res_txt, ref_txt))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    if param.gpu:
        args.gpus = param.gpu
    print('CUDA_VISIBLE_DEVICES: ', args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if param.name:
        args.dir_exps = args.dir_exps / param.name
        args.dir_log = args.dir_exps / 'log'
        args.dir_checkpoint = args.dir_exps / 'checkpoint'
        if not args.dir_exps.is_dir(): args.dir_exps.mkdir()
        if not args.dir_log.is_dir(): args.dir_log.mkdir()
        if not args.dir_checkpoint.is_dir(): args.dir_checkpoint.mkdir()

    if param.mode == 'infer':
        logging.info('enter the INFERING phrase')
        infer()
    elif param.mode == 'train':
        logging.info('enter the TRAINING phrase')
        train()

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
