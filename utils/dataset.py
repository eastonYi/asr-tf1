# coding=utf-8
import numpy as np
import logging
from collections import defaultdict
from random import shuffle
from pathlib import Path
from abc import ABCMeta, abstractmethod
from pypinyin import pinyin, Style
from .dataProcess import load_vocab

from .dataProcess import audio2vector, process_raw_feature, down_sample, splice
from .tools import align2stamp, align2bound, size_bucket_to_put

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class DataSet:
    __metaclass__ = ABCMeta
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx):
        """
        """

    @abstractmethod
    def __len__(self):
        """
        """
    def __call__(self, idx):
        return self.__getitem__(idx)


class ASRDataSet(DataSet):
    def __init__(self, file, args, _shuffle,transform):
        self.file = file
        self.args = args
        self.transform = transform
        self._shuffle = _shuffle
        self.token2idx,self.idx2token = args.token2idx, args.idx2token
        self.end_id = [args.token2idx['<eos>']] if args.data.add_eos else []

    @staticmethod
    def gen_utter_list(file):

        return list(open(file).readlines())

    def __len__(self):
        return len(self.list_utterances)


class ASR_scp_DataSet(ASRDataSet):
    def __init__(self, f_scp, f_trans, args, _shuffle, transform):
        """
        Args:
            f_scp: the scp file consists of paths to feature data
            f_trans: the scp file consists of id and trans
            f_id2label: the normalized transcripts
        """
        from .tools import ArkReader
        self.list_files = [f_scp]
        super().__init__(self.list_files, args, _shuffle, transform)
        self.reader = ArkReader(f_scp)
        self.dict_trans = self.load_trans(f_trans)
        self.list_uttids = list(self.dict_trans.keys())

    def __getitem__(self, idx):
        sample = {}

        try:
            sample['uttid'] = uttid = self.list_uttids[idx]

            trans = self.dict_trans[uttid]
            sample['label'] = np.array(
                [self.token2idx.get(token, self.token2idx['<unk>'])
                for token in trans],
                dtype=np.int32)
            assert len(trans) > 0

            sample['feature'] = self.reader.read_utt_data(uttid)
            if self.transform:
                sample['feature'] = process_raw_feature(sample['feature'], self.args)
        except KeyError:
            print('Not found {}!'.format(self.list_uttids[idx]))
            sample = None
        except AssertionError:
            # print('{} label is None!'.format(self.list_uttids[idx]))
            sample = None

        return sample

    def __len__(self):
        return len(self.list_uttids)

    def uttid2sample(self, uttid):
        id = self.list_uttids.index(uttid)
        return self[id]

    def load_trans(self, f_trans):
        dict_trans = {}
        with open(f_trans, encoding='utf8') as f:
            for line in f:
                line = line.strip().split()
                uttid = line[0]
                trans = line[1:]
                dict_trans[uttid] = trans

        return dict_trans


class ASR_phone_char_ArkDataSet(ASR_scp_DataSet):
    """
    for dataset with phone and char dataset
    needs:
        - phone_file
        - char_file
        - uttid2wav.txt
            uttid wav
        - vocab.txt (used for model output)
            phone
        -
    """
    def __init__(self, f_scp, f_phone, f_char, args, _shuffle, transform):
        self.phone2idx, self.idx2phone = args.phone2idx, args.idx2phone
        super().__init__(f_scp, f_char, args, _shuffle, transform)
        self.phone_char_rate = 2.0
        self.dict_phone_trans = self.load_trans(f_phone)
        self.dict_char_trans = self.load_trans(f_char)
        self.list_uttids = list(set(self.dict_phone_trans.keys()) & set(self.dict_char_trans.keys()))

        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, idx):
        sample = {}

        try:
            sample['uttid'] = uttid = self.list_uttids[idx]
            sample['feature'] = self.reader.read_utt_data(uttid)
            if self.transform:
                sample['feature'] = process_raw_feature(sample['feature'], self.args)

            phones = self.dict_phone_trans[uttid]
            chars = self.dict_char_trans[uttid]
            sample['phone'] = np.array(
                [self.phone2idx[token] for token in phones], dtype=np.int32)
            sample['label'] = np.array(
                [self.token2idx.get(token, self.token2idx['<unk>'])
                for token in chars],
                dtype=np.int32)
            if not self.fix_char(sample['phone'], sample['label']):
                sample = None
        except:
            print('Not found {}!'.format(self.reader.utt_ids[idx]))
            sample = None

        return sample

    def fix_char(self, phone_trans, char_trans):
        res = True
        if len(phone_trans) / len(char_trans) != self.phone_char_rate:
            print(phone_trans)
            print(char_trans)
            res = False
        return res


class LMDataSet(DataSet):
    """
    dataset for language model. Refer to the PTB dataset
    """
    def __init__(self, list_files, args, _shuffle):
        self.list_files = list_files
        self.args = args
        self._shuffle = _shuffle
        self.token2idx, self.idx2token = args.token2idx, args.idx2token
        self.start_id = 1
        if _shuffle:
            shuffle(self.list_files)
        self.size_dataset = self.get_size()

    def __getitem__(self, idx):
        pass

    def __call__(self):
        return self.__iter__()

    def __len__(self):
        return self.size_dataset

    def get_size(self):
        num_lines = 0
        for filename in self.list_files:
            num_lines += sum(1 for line in open(filename))

        return num_lines

    def __iter__(self):
        """
        (Pdb) i[0]
        [1,18,2,36,1,17,7,9,9,6,25,28,3,5,14,1,11,32,24,16,26,22,3,1,16,15,1,18,8,3,1,4]
        (Pdb) i[1]
        [18,2,36,1,17,7,9,9,6,25,28,3,5,14,1,11,32,24,16,26,22,3,1,16,15,1,18,8,3,1,4,1]
        """
        for filename in self.list_files:
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    if len(line) > self.args.model.D.max_label_len:
                        continue
                    text_ids = [self.token2idx[word] for word in line]
                    src_ids = text_ids[:-1]
                    tar_ids = text_ids[1:]
                    # sample = {'src': src_ids, 'tar': tar_ids}

                    yield src_ids, tar_ids


class TextDataSet(LMDataSet):

    def __iter__(self):
        """
        (Pdb) i
        [1,18,2,36,1,17,7,9,9,6,25,28,3,5,14,1,11,32,24,16,26,22,3,1,16,15,1,18,8,3,1,4,1]
        """
        for filename in self.list_files:
            with open(filename) as f:
                for line in f:
                    line = line.strip().split()
                    text_ids = [self.token2idx[word] for word in line]

                    yield text_ids[:self.args.max_label_len]


class SimpleDataLoader:
    def __init__(self, dataset, num_loops=1, batch_size=10):
        self.dataset = dataset
        self.num_loops = num_loops
        self.batch_size = batch_size
        self.list_seq_features = []
        self.list_seq_labels = []

    def __iter__(self):
        return self.next_batch(self.batch_size)

    def next_batch(self, size_batch):
        for _ in range(self.num_loops):
            for sample in self.dataset:
                seq_features, seq_labels = sample['feature'], sample['label']

                self.list_seq_features.append(seq_features)
                self.list_seq_labels.append(seq_labels)

                if len(self.list_seq_features) >= size_batch:
                    yield self.padding_list_seq_with_labels(self.list_seq_features, self.list_seq_labels)
                    self.list_seq_features = []
                    self.list_seq_labels = []

    @staticmethod
    def padding_list_seqs(list_seqs, dtype=np.float32, pad=0.):
        '''
        Pads each sequence to the same length of the longest sequence.
        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens.

        Args:
            list_seqs: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            pad: float, value to pad the list_seqs to the desired value.

        Returns:
            numpy.ndarray: Padded list_seqs shape = (number_of_list_seqs, maxlen)
            list: original sequence lengths
        DEmo:
            >> padding_list_seqs([[21, 11, 3], [31,1]])
            >> (array([[ 21.,  11.,   3.],
                [ 31.,   1.,   0.]], dtype=float32), [3, 2])
        '''
        len_x = [len(s) for s in list_seqs]

        size_batch = len(list_seqs)
        maxlen = max(len_x)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        shape_feature = tuple()
        for s in list_seqs:
            if len(s) > 0:
                shape_feature = np.asarray(s).shape[1:]
                break

        # a tensor filled with padding value
        x = (np.ones((size_batch, maxlen) + shape_feature) * pad).astype(dtype)
        for idx, s in enumerate(list_seqs):
            x[idx, :len(s)] = s

        return x, len_x

    @staticmethod
    def padding_list_seq_with_labels(list_seqs_features,
                                     list_seqs_labels,
                                     dtype=np.float32,
                                     value1=0.,
                                     value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)

        return [x, y, len_x, len_y]


class DataLoader(SimpleDataLoader):
    __metaclass__ = ABCMeta

    '''
    Train/test/dev dataset API for loading via threads and delivering batches.
    '''
    def __init__(self, dataset, args, num_loops=1, num_thread=4, size_queue=2000):
        super().__init__(dataset, num_loops)
        self.args = args
        self.num_thread = num_thread
        self.num_batch_tokens = args.num_batch_tokens
        self.bucket_boundaries = args.bucket_boundaries
        self.list_batch_size = args.list_infer_batch_size if args.list_infer_batch_size else args.list_batch_size

    @abstractmethod
    def __iter__(self):
        '''
        return a iterator of seq, which is used to fentch a batch(with or without bucket)
        yield (seq_features, seq_labels)
        '''

    def batch_with_tfReader_buckets(self):
        buckets = self.args.list_bucket_boundaries
        # max_length = buckets[-1]
        caches = defaultdict(lambda: [[], [], 0])
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_labels = self.sess.run([self.feat, self.label])

            # assert len(seq_features) == len(seq_labels)
            id_bucket, bucket = size_bucket_to_put(len(seq_features), buckets)
            if bucket is None:
                continue
            caches[bucket][0].append(seq_features)
            caches[bucket][1].append(seq_labels)

            caches[bucket][2] += 1
            if caches[bucket][2] >= self.list_batch_size[id_bucket]:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]

        # Clean remain samples.
        for bucket in buckets:
            if caches[bucket][2] > 0:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]
                # logging.info('empty the bucket {}'.format(bucket))

class ASRDataLoader(DataLoader):
    def __init__(self, dataset, args, feat, label, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, num_loops=num_loops, num_thread=num_thread, size_queue=size_queue)
        self.sess = None
        self.feat = feat
        self.label = label
        self.size_dataset = len(dataset)
        self.batch_size = batch_size

    def __iter__(self):
        # return self.batch_with_tfReader(self.batch_size)
        return self.batch_with_tfReader_buckets()

    def __len__(self):
        return self.size_dataset


class ASR_Multi_DataLoader(ASRDataLoader):
    def __init__(self, dataset, args, feat, phone, label, batch_size, num_loops, num_thread=4, size_queue=2000):
        super().__init__(dataset, args, feat, label, batch_size, num_loops, num_thread, size_queue)
        self.phone = phone

    def __iter__(self):
        buckets = self.args.list_bucket_boundaries
        # max_length = buckets[-1]
        caches = defaultdict(lambda: [[], [], [], 0])
        for _ in range(len(self)*self.num_loops):
            seq_features, seq_phones, seq_labels = self.sess.run([self.feat, self.phone, self.label])

            id_bucket, bucket = size_bucket_to_put(len(seq_features), buckets)
            if bucket is None:
                continue
            caches[bucket][0].append(seq_features)
            caches[bucket][1].append(seq_phones)
            caches[bucket][2].append(seq_labels)

            caches[bucket][3] += 1
            if caches[bucket][3] >= self.list_batch_size[id_bucket]:
                batch = (caches[bucket][0], caches[bucket][1], caches[bucket][2])
                yield self.padding_list_seq_with_multi_labels(*batch)
                caches[bucket] = [[], [], [], 0]

        # Clean remain samples.
        for bucket in buckets:
            if caches[bucket][3] > 0:
                batch = (caches[bucket][0], caches[bucket][1], caches[bucket][2])
                yield self.padding_list_seq_with_multi_labels(*batch)
                caches[bucket] = [[], [], [], 0]
                # logging.info('empty the bucket {}'.format(bucket))

    @staticmethod
    def padding_list_seq_with_multi_labels(list_seqs_features,
                                           list_seqs_phones,
                                           list_seqs_labels,
                                           dtype=np.float32,
                                           value1=0.,
                                           value2=0):
        x, len_x = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_features,
            dtype=dtype,
            pad=value1)
        y, len_y = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_phones,
            dtype=np.int32,
            pad=value2)
        y1, len_y1 = DataLoader.padding_list_seqs(
            list_seqs=list_seqs_labels,
            dtype=np.int32,
            pad=value2)

        return [x, y, y1, len_x, len_y, len_y1]
