# coding=utf-8
import numpy as np
import logging
from collections import defaultdict
from random import shuffle
from pathlib import Path
from abc import ABCMeta, abstractmethod

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
    def __init__(self,file,args,_shuffle,transform):
        self.file = file
        self.args = args
        self.transform = transform
        self._shuffle = _shuffle
        self.token2idx,self.idx2token = args.token2idx,args.idx2token
        self.end_id = self.gen_end_id(self.token2idx)

    def gen_end_id(self, token2idx):
        if '<eos>' in token2idx.keys():
            eos_id = [token2idx['<eos>']]
        else:
            eos_id = []

        return eos_id

    @staticmethod
    def gen_utter_list(file):

        return list(open(file).readlines())

    def __len__(self):
        return len(self.list_utterances)


class ASR_csv_DataSet(ASRDataSet):
    def __init__(self, list_files, args, _shuffle, transform):
        super().__init__(list_files, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(list_files)
        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]
        wav, seq_label = utterance.strip().split(',')
        fea = audio2vector(wav, self.args.data.dim_raw_input)
        if self.transform:
            fea = process_raw_feature(fea, self.args)

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['<unk>'])
            for word in seq_label.split(' ')] + self.end_id,
            dtype=np.int32)

        sample = {'uttid': wav, 'feature': fea, 'label': seq_label}

        return sample

    @staticmethod
    def gen_utter_list(list_files):
        list_utter = []
        for file in list_files:
            with open(file) as f:
                list_utter.extend(f.readlines())
        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

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
            sample['feature'] = self.reader.read_utt_data(uttid)
            if self.transform:
                sample['feature'] = process_raw_feature(sample['feature'], self.args)

            trans = self.dict_trans[uttid]
            sample['label'] = np.array(
                [self.token2idx.get(token, self.token2idx['<unk>'])
                for token in trans] + self.end_id,
                dtype=np.int32)
        except:
            print('Not found {}!'.format(self.reader.utt_ids[idx]))
            sample = None

        return sample

    def __len__(self):
        return len(self.list_uttids)

    def load_trans(self, f_trans):
        dict_trans = {}
        with open(f_trans, encoding='utf8') as f:
            for line in f:
                line = line.strip().split()
                uttid = line[0]
                trans = line[1:]
                dict_trans[uttid] = trans

        return dict_trans


class ASR_align_DataSet(ASRDataSet):
    """
    for dataset with alignment, i.e. TIMIT
    needs:
        vocab.txt remains the index of phones in phones.txt !!
        - align_file
            uttid phone_id phone_id ...
        - trans_file

        - uttid2wav.txt
            uttid wav
        - vocab.txt (used for model output)
            phone
        -
    """
    def __init__(self, trans_file, align_file, uttid2wav, feat_len_file, args, _shuffle, transform):
        super().__init__(align_file, args, _shuffle, transform)
        self.dict_wavs = self.load_uttid2wav(uttid2wav)
        self.list_uttids = list(self.dict_wavs.keys())
        self.dict_trans = self.load_trans(trans_file) if trans_file else None
        self.dict_aligns = self.load_aligns(align_file, feat_len_file) if align_file else None

        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, id):
        uttid = self.list_uttids[id]
        wav = self.dict_wavs[uttid]

        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        try:
            trans = self.dict_trans[uttid]
        except:
            trans = None
        try:
            align = self.dict_aligns[uttid]
            stamps = align2stamp(align)
        except:
            align = None
            stamps = None

        sample = {'uttid': uttid,
                  'feature': feat,
                  'trans': trans,
                  'align': align,
                  'stamps': stamps}

        return sample

    def get_attrs(self, attr, uttids, max_len=None):
        """
        length serves for the align attr to ensure the align's length same as feature
        """
        list_res = []
        list_len = []
        for uttid in uttids:
            if type(uttid) == bytes:
                uttid = uttid.decode('utf-8')
            if attr == 'wav':
                wav = self.dict_wavs[uttid]
                res = wav
            elif attr == 'feature':
                wav = self.dict_wavs[uttid]
                feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
                if self.transform:
                    feat = process_raw_feature(feat, self.args)
                res = feat
            elif attr == 'trans':
                trans = self.dict_trans[uttid]
                res = trans
            elif attr == 'align':
                align = self.dict_aligns[uttid]
                res = align
            elif attr == 'stamps':
                align = self.dict_aligns[uttid]
                stamps = align2stamp(align)
                res = stamps
            elif attr == 'bounds':
                align = self.dict_aligns[uttid]
                bounds = align2bound(align)
                res = bounds
            else:
                raise KeyError
            list_res.append(res)
            list_len.append(len(res))

        if attr in ('trans', 'align', 'stamps', 'bounds'):
            max_len = max(list_len) if not max_len else max_len
            list_padded = []
            for res in list_res:
                list_padded.append(np.concatenate([res, [0]*(max_len-len(res))])[: max_len])
            list_res = np.array(list_padded, np.int32)

        return list_res

    def load_uttid2wav(self, uttid2wav):
        dict_wavs = {}
        with open(uttid2wav) as f:
            for line in f:
                uttid, wav = line.strip().split()
                dict_wavs[uttid] = wav

        return dict_wavs

    def load_aligns(self, align_file, feat_len_file):
        dict_aligns = defaultdict(lambda: np.array([0]))
        dict_feat_len = {}

        with open(feat_len_file) as f:
            for line in f:
                uttid, len_feature = line.strip().split()
                dict_feat_len[uttid] = int(len_feature)
        align_rate = self.get_alignRate(align_file)

        with open(align_file) as f:
            for line in f:
                uttid, align = line.strip().split(maxsplit=1)
                len_feat = dict_feat_len[uttid]
                align = [int(i) for i in align.split()] + [1]
                # assert len(align) == len_feat + 1
                dict_aligns[uttid] = np.array(align[::align_rate][:len_feat])

        return dict_aligns


    def load_trans(self, trans_file):
        dict_trans = defaultdict(lambda: None)
        with open(trans_file) as f:
            for line in f:
                uttid, load_trans = line.strip().split(maxsplit=1)
                dict_trans[uttid] = np.array([self.token2idx[i] for i in load_trans.split()])

        return dict_trans

    def __len__(self):
        return len(self.list_uttids)

    def get_alignRate(self, align_file):
        with open(align_file) as f:
            uttid, align = f.readline().strip().split(maxsplit=1)

        wav = self.dict_wavs[uttid]
        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        align = align.split()

        return int(np.round(len(align)/len(feat)))


class ASR_align_ArkDataSet(ASRDataSet):
    """
    for dataset with alignment, i.e. TIMIT
    needs:
        vocab.txt remains the index of phones in phones.txt !!
        - align_file
            uttid phone_id phone_id ...
        - trans_file

        - uttid2wav.txt
            uttid wav
        - vocab.txt (used for model output)
            phone
        -
    """
    def __init__(self, scp_file, trans_file, align_file, feat_len_file, args, _shuffle, transform):
        super().__init__(align_file, args, _shuffle, transform)
        from .tools import ArkReader
        self.reader = ArkReader(scp_file)
        self.dict_trans = self.load_trans(trans_file) if trans_file else None
        self.list_uttids = list(self.dict_trans.keys())
        self.dict_aligns = self.load_aligns(align_file, feat_len_file) if align_file else None

        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, id):
        uttid = self.list_uttids[id]

        feat = self.reader.read_utt_data(id)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        try:
            trans = self.dict_trans[uttid]
        except:
            trans = None
        try:
            align = self.dict_aligns[uttid]
            stamps = align2stamp(align)
        except:
            align = None
            stamps = None

        sample = {'uttid': uttid,
                  'feature': feat,
                  'trans': trans,
                  'align': align,
                  'stamps': stamps}

        return sample

    def get_attrs(self, attr, uttids, max_len=None):
        """
        length serves for the align attr to ensure the align's length same as feature
        """
        list_res = []
        list_len = []
        for uttid in uttids:
            if type(uttid) == bytes:
                uttid = uttid.decode('utf-8')
            if attr == 'wav':
                wav = self.dict_wavs[uttid]
                res = wav
            elif attr == 'feature':
                wav = self.dict_wavs[uttid]
                feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
                if self.transform:
                    feat = process_raw_feature(feat, self.args)
                res = feat
            elif attr == 'trans':
                trans = self.dict_trans[uttid]
                res = trans
            elif attr == 'align':
                align = self.dict_aligns[uttid]
                res = align
            elif attr == 'stamps':
                align = self.dict_aligns[uttid]
                stamps = align2stamp(align)
                res = stamps
            elif attr == 'bounds':
                align = self.dict_aligns[uttid]
                bounds = align2bound(align)
                res = bounds
            else:
                raise KeyError
            list_res.append(res)
            list_len.append(len(res))

        if attr in ('trans', 'align', 'stamps', 'bounds'):
            max_len = max(list_len) if not max_len else max_len
            list_padded = []
            for res in list_res:
                list_padded.append(np.concatenate([res, [0]*(max_len-len(res))])[: max_len])
            list_res = np.array(list_padded, np.int32)

        return list_res

    def load_aligns(self, align_file, feat_len_file):
        dict_aligns = defaultdict(lambda: np.array([0]))
        dict_feat_len = {}

        with open(feat_len_file) as f:
            for line in f:
                uttid, len_feature = line.strip().split()
                dict_feat_len[uttid] = int(len_feature)
        align_rate = self.get_alignRate(align_file)

        with open(align_file) as f:
            for line in f:
                uttid, align = line.strip().split(maxsplit=1)
                len_feat = dict_feat_len[uttid]
                align = [int(i) for i in align.split()] + [1]
                # assert len(align) == len_feat + 1
                dict_aligns[uttid] = np.array(align[::align_rate][:len_feat])

        return dict_aligns

    def load_trans(self, trans_file):
        # dict_trans = defaultdict(lambda: None)
        dict_trans = {}
        with open(trans_file) as f:
            for line in f:
                uttid, load_trans = line.strip().split(maxsplit=1)
                dict_trans[uttid] = np.array([self.token2idx[i] for i in load_trans.split()])

        return dict_trans

    def __len__(self):
        return len(self.list_uttids)

    def get_alignRate(self, align_file):
        with open(align_file) as f:
            uttid, align = f.readline().strip().split(maxsplit=1)

        wav = self.dict_wavs[uttid]
        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        align = align.split()

        return int(np.round(len(align)/len(feat)))


class ASR_classify_DataSet(ASRDataSet):

    def __init__(self, dir_wavs, class_file, args, _shuffle, transform):
        super().__init__(class_file, args, _shuffle, transform)
        self.dict_wavs = self.load_wav(dir_wavs)
        self.dict_y, self.dict_class = self.load_y(class_file)
        self.list_uttids = list(self.dict_y.keys())

        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, id):
        uttid = self.list_uttids[id]
        wav = self.dict_wavs[uttid]
        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        y = self.dict_y[uttid]

        sample = {'uttid': uttid,
                  'feature': feat,
                  'class': y}

        return sample

    def load_y(self, class_file):
        dict_y = {}
        dict_class = {}
        with open(class_file) as f:
            for line in f:
                uttid, y = line.strip().split()
                if y not in dict_class.keys():
                    dict_class[y] = len(dict_class)
                dict_y[uttid] = dict_class[y]

        return dict_y, dict_class

    def get_y(self, uttids):
        list_y = []
        for uttid in uttids:
            if type(uttid) == bytes:
                uttid = uttid.decode('utf-8')
            y = self.dict_y[uttid]
            list_y.append(y)

        return np.array(list_y, np.int32)

    def load_wav(self, dir_wavs):
        dict_wavs = {}
        wav_path = Path(dir_wavs)
        for wav_file in wav_path.glob('*.wav'):
            uttid = str(wav_file.name)[:-4]
            dict_wavs[uttid] = str(wav_file)

        return dict_wavs

    def __len__(self):
        return len(self.list_uttids)


class ASR_classify_ArkDataSet(ASRDataSet):

    def __init__(self, scp_file, class_file, args, _shuffle):
        super().__init__(class_file, args, _shuffle, transform=False)
        from .tools import ArkReader
        self.reader = ArkReader(scp_file)
        self.dict_y, self.dict_class = self.load_y(class_file)
        self.list_uttids = list(self.dict_y.keys())
        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, id):
        uttid = self.list_uttids[id]
        feat = self.reader.read_utt_data(id)
        # feat = feat[::3, :]
        feat = down_sample(splice(feat, 2, 0), 3)
        y = self.dict_y[uttid]

        sample = {'uttid': uttid,
                  'feature': feat,
                  'label': y}

        return sample

    def load_y(self, class_file):
        dict_y = {}
        dict_class = {}
        with open(class_file) as f:
            for line in f:
                uttid, y = line.strip().split()
                if y not in dict_class.keys():
                    dict_class[y] = np.array(len(dict_class))
                dict_y[uttid] = dict_class[y]

        return dict_y, dict_class

    def __len__(self):
        return len(self.list_uttids)


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


class PTB_LMDataSet(LMDataSet):
    """
    dataset for language model. Refer to the PTB dataset
    """
    def __init__(self, list_files, args, _shuffle):
        super().__init__(list_files, args, _shuffle)
        self.start_id = args.token2idx['<sos>']
        self.end_id = args.token2idx['<eos>']

    def __iter__(self):
        for filename in self.list_files:
            with open(filename) as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) > self.args.list_bucket_boundaries[-1]:
                        continue
                    text_ids = [self.token2idx[word] for word in line]
                    src_ids = [self.start_id] + text_ids
                    tar_ids = text_ids + [self.end_id]

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
                    if len(line) > self.args.model.D.max_label_len:
                        continue
                    text_ids = [self.token2idx[word] for word in line]

                    yield text_ids


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
        self.num_batch_token = args.num_batch_token
        self.bucket_boundaries = args.bucket_boundaries

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
            if caches[bucket][2] >= self.args.list_batch_size[id_bucket]:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]

        # Clean remain samples.
        for bucket in buckets:
            if caches[bucket][2] > 0:
                batch = (caches[bucket][0], caches[bucket][1])
                yield self.padding_list_seq_with_labels(*batch)
                caches[bucket] = [[], [], 0]
                logging.info('empty the bucket {}'.format(bucket))


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
