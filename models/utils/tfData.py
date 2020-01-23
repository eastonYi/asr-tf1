#!/usr/bin/env
# coding=utf-8
import tensorflow as tf
import numpy as np
import logging
import os
from tqdm import tqdm
from pathlib import Path
from random import shuffle
from multiprocessing import Process, Queue

from .tfAudioTools import splice, down_sample, add_delt
from utils.tools import mkdirs


def _bytes_feature(value):
    """Returns a bytes_list from a list of string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def process_raw_feature(seq_raw_features, args):
    # 1-D, 2-D
    if args.data.add_delta:
        seq_raw_features = add_delt(seq_raw_features)

    # Splice
    fea = splice(
        seq_raw_features,
        left_num=args.data.left_context,
        right_num=args.data.right_context)

    # downsample
    fea = down_sample(
        fea,
        rate=args.data.downsample,
        axis=0)
    fea.set_shape([None, args.data.dim_input])

    return fea


def fentch_filelist(dir_data, _shuffle=False):
    p = Path(dir_data)
    assert p.is_dir()

    list_filenames = [str(i) for i in p.glob('*.recode')]

    if _shuffle:
        shuffle(list_filenames)
    else:
        list_filenames.sort()

    return list_filenames


class TFDataReader:
    def __init__(self, dir_tfdata, args, _shuffle=False, num_epochs=None, transform=True):
        self.dir_data = dir_tfdata
        self._shuffle = _shuffle
        self.args = args
        self.sess = None
        self.read_multilabel = args.dirs.vocab_phone
        self.max_feature_len = self.args.data.max_feature_len
        self.list_batch_size = self.args.list_batch_size
        self.list_bucket_boundaries = self.args.list_bucket_boundaries
        self.readTFRecord(_shuffle=_shuffle, num_epochs=num_epochs, transform=transform)

    def __iter__(self):
        """It is only a demo! Using `fentch_batch_with_TFbuckets` in practice."""
        if not self.sess:
            raise NotImplementedError('please assign sess to the TFReader! ')

        for i in range(len(self.args.data.size_dev)):
            yield self.sess.run([self.feat, self.label])

    @staticmethod
    def read_tfdata_info(dir_save):
        data_info = {}
        with open(dir_save/'tfdata.info') as f:
            for line in f:
                if 'dim_feature' in line or \
                    'num_tokens' in line or \
                    'size_dataset' in line:
                    line = line.strip().split(' ')
                    data_info[line[0]] = int(line[1])

        return data_info

    def readTFRecord(self, _shuffle=False, num_epochs=None, transform=False):

        list_filenames = fentch_filelist(self.dir_data, _shuffle)

        filename_queue = tf.train.string_input_producer(
            list_filenames, num_epochs=num_epochs, shuffle=shuffle)

        reader_tfRecord = tf.TFRecordReader()
        _, serialized_example = reader_tfRecord.read(filename_queue)

        if self.read_multilabel:
            features = tf.parse_single_example(
                serialized_example,
                features={'feature': tf.FixedLenFeature([], tf.string),
                          'phone': tf.FixedLenFeature([], tf.string),
                          'label': tf.FixedLenFeature([], tf.string)})
        else:
            features = tf.parse_single_example(
                serialized_example,
                features={'feature': tf.FixedLenFeature([], tf.string),
                          'label': tf.FixedLenFeature([], tf.string)})

        feature = tf.reshape(tf.decode_raw(features['feature'], tf.float32),
                             [-1, self.args.data.dim_feature])[:self.max_feature_len, :]
        label = tf.decode_raw(features['label'], tf.int32)

        if transform:
            feature = process_raw_feature(feature, self.args)
        if self.args.data.add_eos:
            label = tf.concat([label, [self.args.eos_idx]], 0)

        self.feat = feature
        self.label = label
        if self.read_multilabel:
            phone = tf.decode_raw(features['phone'], tf.int32)
            self.phone = phone

    def fentch_batch(self, batch_size):
        list_inputs = [self.feat, self.label, tf.shape(self.feat)[0], tf.shape(self.label)[0]]
        list_outputs = tf.train.batch(
            tensors=list_inputs,
            batch_size=batch_size,
            num_threads=8,
            capacity=30,
            dynamic_pad=True,
            allow_smaller_final_batch=True
        )
        seq_len_feats = tf.reshape(list_outputs[2], [-1])
        seq_len_label = tf.reshape(list_outputs[3], [-1])

        return list_outputs[0], list_outputs[1], seq_len_feats, seq_len_label

    def fentch_multi_batch_bucket(self):
        list_inputs = [self.feat, self.phone, self.label,
                       tf.shape(self.feat)[0], tf.shape(self.phone)[0], tf.shape(self.label)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[3],
            tensors=list_inputs,
            batch_size=self.list_batch_size,
            bucket_boundaries=self.list_bucket_boundaries,
            num_threads=8,
            bucket_capacities=[i*2 for i in self.list_batch_size],
            capacity=30,  # size of the top queue
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[3], [-1])
        seq_len_phone = tf.reshape(list_outputs[4], [-1])
        seq_len_label = tf.reshape(list_outputs[5], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2], seq_len_feats, seq_len_phone, seq_len_label

    def fentch_multi_batch(self, batch_size):
        list_inputs = [self.feat, self.phone, self.label,
                       tf.shape(self.feat)[0], tf.shape(self.phone)[0], tf.shape(self.label)[0]]
        list_outputs = tf.train.batch(
            tensors=list_inputs,
            batch_size=batch_size,
            num_threads=8,
            capacity=30,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[3], [-1])
        seq_len_phone = tf.reshape(list_outputs[4], [-1])
        seq_len_label = tf.reshape(list_outputs[5], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2], seq_len_feats, seq_len_phone, seq_len_label

    def fentch_batch_bucket(self):
        list_inputs = [self.feat, self.label, tf.shape(self.feat)[0], tf.shape(self.label)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=list_inputs[2],
            tensors=list_inputs,
            batch_size=self.list_batch_size,
            bucket_boundaries=self.list_bucket_boundaries,
            num_threads=8,
            bucket_capacities=[i*2 for i in self.list_batch_size],
            capacity=30,  # size of the top queue
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[2], [-1])
        seq_len_label = tf.reshape(list_outputs[3], [-1])

        return list_outputs[0], list_outputs[1], seq_len_feats, seq_len_label

    def fentch_multi_label_batch_bucket(self):
        """
        the input tensor length is not equal,
        so will add the len as a input tensor
        list_inputs: [tensor1, tensor2]
        added_list_inputs: [tensor1, tensor2, len_tensor1, len_tensor2]
        """
        list_inputs = [self.feat, self.phone, self.label,
                       tf.shape(self.feat)[0], tf.shape(self.phone)[0], tf.shape(self.label)[0]]
        _, list_outputs = tf.contrib.training.bucket_by_sequence_length(
            input_length=tf.shape(self.feat)[0],
            tensors=list_inputs,
            batch_size=self.args.list_batch_size,
            bucket_boundaries=self.args.list_bucket_boundaries,
            num_threads=8,
            bucket_capacities=[i*3 for i in self.args.list_batch_size],
            capacity=2000,
            dynamic_pad=True,
            allow_smaller_final_batch=True)
        seq_len_feats = tf.reshape(list_outputs[3], [-1])
        seq_len_phone = tf.reshape(list_outputs[4], [-1])
        seq_len_label = tf.reshape(list_outputs[5], [-1])

        return list_outputs[0], list_outputs[1], list_outputs[2],\
                seq_len_feats, seq_len_phone, seq_len_label


class TFDataSaver:
    """
    test on TF2.0-alpha
    """
    def __init__(self, dataset, dir_save, args, size_file=5000000, max_feat_len=3000):
        self.dataset = dataset
        self.max_feat_len = max_feat_len
        self.dir_save = dir_save
        mkdirs(self.dir_save)
        self.args = args
        self.save_multilabel = args.dirs.vocab_phone
        self.add_eos = args.data.add_eos
        self.size_file = size_file
        self.dim_feature = dataset[0]['feature'].shape[-1]

    def save(self):
        num_token = 0
        idx_file = -1
        num_damaged_sample = 0

        assert self.dataset.transform == False
        with open(self.dir_save/'feature_length.txt', 'w') as fw:
            for i, sample in enumerate(tqdm(self.dataset)):
                if not sample:
                    num_damaged_sample += 1
                    continue
                dim_feature = sample['feature'].shape[-1]
                if (num_token // self.size_file) > idx_file:
                    idx_file = num_token // self.size_file
                    print('saving to file {}/{}.recode'.format(self.dir_save, idx_file))
                    writer = tf.io.TFRecordWriter(str(self.dir_save/'{}.recode'.format(idx_file)))

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={'feature': _bytes_feature(sample['feature'].tostring()),
                                 'label': _bytes_feature(sample['label'].tostring())}
                    )
                )
                writer.write(example.SerializeToString())
                num_token += len(sample['feature'])
                line = sample['uttid'] + ' ' + str(len(sample['feature']))
                fw.write(line + '\n')

        with open(self.dir_save/'tfdata.info', 'w') as fw:
            # print('data_file {}'.format(dataset.list_files), file=fw)
            print('dim_feature {}'.format(dim_feature), file=fw)
            print('num_tokens {}'.format(num_token), file=fw)
            print('size_dataset {}'.format(i-num_damaged_sample), file=fw)
            print('damaged samples: {}'.format(num_damaged_sample), file=fw)

        return

    def split_save(self):
        output = Queue()
        coord = tf.train.Coordinator()
        assert self.dataset.transform == False

        def gen_recoder(i):
            num_saved = 0
            num_damaged_sample = 0
            idx_start = i*self.size_file
            idx_end = min((i+1)*self.size_file, len(self.dataset))
            writer = tf.io.TFRecordWriter(str(self.dir_save/'{}.recode'.format(i)))
            print('saving dataset[{}: {}] to file {}/{}.recode'.format(idx_start, idx_end, self.dir_save, i))

            with open(self.dir_save/'feature_length.{}.txt'.format(i), 'w') as fw:
                if i == 0:
                    m = tqdm(range(idx_start, idx_end))
                else:
                    m = range(idx_start, idx_end)
                for j in m:
                    sample = self.dataset[j]
                    if not sample:
                        num_damaged_sample += 1
                        continue

                    if self.save_multilabel:
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={'feature': _bytes_feature(sample['feature'].tostring()),
                                         'phone': _bytes_feature(sample['phone'].tostring()),
                                         'label': _bytes_feature(sample['label'].tostring())}))
                    else:
                        example = tf.train.Example(
                            features=tf.train.Features(
                                feature={'feature': _bytes_feature(sample['feature'].tostring()),
                                         'label': _bytes_feature(sample['label'].tostring())}))

                    writer.write(example.SerializeToString())
                    line = sample['uttid'] + ' ' + str(len(sample['feature']))
                    fw.write(line + '\n')
                    num_saved += 1
                    # if num_saved % 2000 == 0:
                    #     print('saved {} samples in {}.recode'.format(num_saved, i))
            print('{}.recoder finished, {} saved, {} damaged. '.format(i, num_saved, num_damaged_sample))
            output.put((i, num_damaged_sample, num_saved))

        processes = []
        workers = len(self.dataset)//self.size_file + 1
        print('save {} samples to {} recoder files'.format(len(self.dataset), workers))
        for i in range(workers):
            p = Process(target=gen_recoder, args=(i, ))
            p.start()
            processes.append(p)
        print('generating ...')
        coord.join(processes)
        print('save recode files finished.')

        res = [output.get() for _ in processes]
        num_saved = sum([x[2] for x in res])
        num_damaged = sum([x[1] for x in res])
        # TODO: concat feature length file
        with open(str(self.dir_save/'tfdata.info'), 'w') as fw:
            fw.write('data_file {}\n'.format(self.dataset.file))
            fw.write('dim_feature {}\n'.format(self.dataset[0]['feature'].shape[-1]))
            fw.write('size_dataset {}\n'.format(num_saved))
            fw.write('damaged samples: {}\n'.format(num_damaged))

        os.system('cat {}/feature_length.*.txt > {}/feature_length.txt'.format(self.dir_save, self.dir_save))

        print('ALL FINISHED.')


if __name__ == '__main__':
    # from configs.arguments import args
    from tqdm import tqdm
    import sys

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')
    # test_bucket_boundaries(args=args)
    # test_tfdata_bucket(args=args, num_threads=args.num_parallel)
    # test_queue()
