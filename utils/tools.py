import numpy as np
import editdistance as ed
from tqdm import tqdm
from time import time
from struct import pack, unpack
import codecs
import logging
import os
from tempfile import mkstemp


def get_batch_length(batch):
    if batch.ndim == 3:
        return np.sum(np.max(np.abs(batch) > 0, -1), -1, dtype=np.int32)
    elif batch.ndim == 2:
        return np.sum(np.abs(batch) > 0, -1, dtype=np.int32)


def mkdirs(filename):
    if not filename.parent.is_dir():
        mkdirs(filename.parent)

    if '.' not in str(filename) and not filename.is_dir():
        filename.mkdir()


def get_dataset_ngram(text_file, n, k, savefile=None, split=5000):
    """
    Simply concatenate all sents into one will bring in noisy n-gram at end of each sent.
    Here we count ngrams for each sent and sum them up.
    """
    from utils.dataProcess import get_N_gram
    from nltk import FreqDist

    def iter_in_sent(sent):
        for word in sent.split():
            yield word

    print('analysing text ...')

    list_utterances = open(text_file).readlines()

    ngrams_global = FreqDist()
    for i in range(len(list_utterances)//split +1):
        ngrams = FreqDist()
        text = list_utterances[i*split: (i+1)*split]
        for utt in tqdm(text):
            _, seq_label, _ = utt.strip().split(',')
            ngram = get_N_gram(iter_in_sent(seq_label), n)
            ngrams += ngram

        ngrams_global += dict(ngrams.most_common(2*k))

    if savefile:
        with open(savefile, 'w') as fw:
            for ngram,num in ngrams_global.most_common(k):
                line = '{}:{}'.format(ngram,num)
                fw.write(line+'\n')

    return ngrams_global


def read_ngram(top_k, file, token2idx, type='list'):
    """
    """
    total_num = 0
    ngram_py = []
    with open(file) as f:
        for _, line in zip(range(top_k), f):
            ngram, num = line.strip().split(':')
            ngram = tuple(token2idx[i[1:-1]] for i in ngram[1:-1].split(', '))
            ngram_py.append((ngram, int(num)))
            total_num += int(num)

    if type == 'dict':
        dict_ngram_py = {}
        for ngram, num in ngram_py:
            dict_ngram_py[ngram] = num/total_num

        return dict_ngram_py

    elif type == 'list':
        list_ngram_py = []
        for ngram, num in ngram_py:
            list_ngram_py.append((ngram, num/total_num))

        return list_ngram_py, total_num


def align_shrink(align):
    _token = None
    list_tokens = []
    for token in align:
        if _token != token:
            list_tokens.append(token)
            _token = token

    return list_tokens


def batch_cer(preds, reference):
    """
    preds, reference: align type
    result and reference are lists of tokens
    eos_idx is the padding token or eos token
    """

    batch_dist = 0
    batch_len = 0
    batch_res_len = 0
    for res, ref in zip(preds, reference):
        res = align_shrink(res[res>0])
        ref = align_shrink(ref[ref>0])
        # print(len(res)/len(ref))
        batch_dist += ed.eval(res, ref)
        batch_len += len(ref)
        batch_res_len += len(res)

    return batch_dist, batch_len, batch_res_len


def pertubated_model_weights(w, p, sigma):
    weights_try = []
    for index, i in enumerate(p):
        jittered = sigma*i
        weights_try.append(w[index] + jittered)

    return weights_try


def ngram2kernel(ngram, args):
    kernel = np.zeros([args.data.ngram, args.dim_output, args.data.top_k], dtype=np.float32)
    list_py = []
    for i, (z, py) in enumerate(ngram):
        list_py.append(py)
        for j, token in enumerate(z):
            kernel[j][token][i] = 1.0
    py = np.array(list_py, dtype=np.float32)

    return kernel, py


def get_preds_ngram(preds, len_preds, n):
    """
    Simply concatenate all sents into one will bring in noisy n-gram at end of each sent.
    Here we count ngrams for each sent and sum them up.
    """
    from utils.dataProcess import get_N_gram

    def iter_preds(preds, len_preds):
        for len, utt in zip(len_preds, preds):
            for token in utt[:len]:
                yield token.numpy()
    ngrams = get_N_gram(iter_preds(preds, len_preds), n)

    return ngrams


def store_2d(array, fw):
    fw.write(pack('I', len(array)))
    for i, distrib in enumerate(array):
        for p in distrib:
            p = pack('f', p)
            fw.write(p)


class ArkReader(object):
    '''
    Class to read Kaldi ark format. Each time, it reads one line of the .scp
    file and reads in the corresponding features into a numpy matrix. It only
    supports binary-formatted .ark files. Text and compressed .ark files are not
    supported. The inspiration for this class came from pdnn toolkit (see
    licence at the top of this file) (https://github.com/yajiemiao/pdnn)
    '''

    def __init__(self, scp_path):
        '''
        ArkReader constructor

        Args:
            scp_path: path to the .scp file
        '''
        self.scp_position = 0
        fin = open(scp_path, "r", errors='ignore')
        self.dict_scp = {}
        line = fin.readline()
        while line != '' and line != None:
            uttid, path_pos = line.replace('\n', '').split(' ')
            path, pos = path_pos.split(':')
            self.dict_scp[uttid] = (path, pos)
            line = fin.readline()

        fin.close()

    def read_utt_data(self, uttid):
        '''
        read data from the archive

        Args:
            index: index of the utterance that will be read

        Returns:
            a numpy array containing the data from the utterance
        '''
        ark_read_buffer = open(self.dict_scp[uttid][0], 'rb')
        ark_read_buffer.seek(int(self.dict_scp[uttid][1]), 0)
        header = unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != b'B':
            print("Input .ark file is not binary")
            exit(1)
        if header == (b'B', b'C', b'M', b' '):
            # print('enter BCM')
            g_min_value, g_range, g_num_rows, g_num_cols = unpack('ffii', ark_read_buffer.read(16))
            utt_mat = np.zeros([g_num_rows, g_num_cols], dtype=np.float32)
            #uint16 percentile_0; uint16 percentile_25; uint16 percentile_75; uint16 percentile_100;
            per_col_header = []
            for i in range(g_num_cols):
                per_col_header.append(unpack('HHHH', ark_read_buffer.read(8)))
                #print per_col_header[i]

            tmp_mat = np.frombuffer(ark_read_buffer.read(g_num_rows * g_num_cols), dtype=np.uint8)

            pos = 0
            for i in range(g_num_cols):
                p0 = float(g_min_value + g_range * per_col_header[i][0] / 65535.0)
                p25 = float(g_min_value + g_range * per_col_header[i][1] / 65535.0)
                p75 = float(g_min_value + g_range * per_col_header[i][2] / 65535.0)
                p100 = float(g_min_value + g_range * per_col_header[i][3] / 65535.0)

                d1 = float((p25 - p0) / 64.0)
                d2 = float((p75 - p25) / 128.0)
                d3 = float((p100 - p75) / 63.0)
                for j in range(g_num_rows):
                    c = tmp_mat[pos]
                    if c <= 64:
                        utt_mat[j][i] = p0 + d1 * c
                    elif c <= 192:
                        utt_mat[j][i] = p25 + d2 * (c - 64)
                    else:
                        utt_mat[j][i] = p75 + d3 * (c - 192)
                    pos += 1
        elif header == (b'B', b'F', b'M', b' '):
            # print('enter BFM')
            m, rows = unpack('<bi', ark_read_buffer.read(5))
            n, cols = unpack('<bi', ark_read_buffer.read(5))
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=np.float32)
            utt_mat = np.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_mat


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item not in self:
            logging.warning('{} is not in the dict. None is returned as default.'.format(item))
            return None
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


class DataReader(object):
    """
    Read data and create batches for training and testing.
    """

    def __init__(self, config):
        self._config = config
        self._tmps = set()
        self.load_vocab()

    def __del__(self):
        for fname in self._tmps:
            if os.path.exists(fname):
                os.remove(fname)

    def load_vocab(self):
        """
        Load vocab from disk.
        The first four items in the vocab should be <PAD>, <UNK>, <S>, </S>
        """

        def load_vocab_(path, vocab_size):
            vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8')]
            vocab = vocab[:vocab_size]
            assert len(vocab) == vocab_size
            word2idx = {word: idx for idx, word in enumerate(vocab)}
            idx2word = {idx: word for idx, word in enumerate(vocab)}
            return word2idx, idx2word

        logging.debug('Load vocabularies %s and %s.' % (self._config.src_vocab, self._config.dst_vocab))
        self.src2idx, self.idx2src = load_vocab_(self._config.src_vocab, self._config.src_vocab_size)
        self.dst2idx, self.idx2dst = load_vocab_(self._config.dst_vocab, self._config.dst_vocab_size)

    @staticmethod
    def shuffle(list_of_files):
        tf_os, tpath = mkstemp()
        tf = open(tpath, 'w')

        fds = [open(ff) for ff in list_of_files]

        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            print("<CONCATE4SHUF>".join(lines), file=tf)

        [ff.close() for ff in fds]
        tf.close()

        os.system('shuf %s > %s' % (tpath, tpath + '.shuf'))

        fnames = ['/tmp/{}.{}.{}.shuf'.format(i, os.getpid(), time.time()) for i, ff in enumerate(list_of_files)]
        fds = [open(fn, 'w') for fn in fnames]

        for l in open(tpath + '.shuf'):
            s = l.strip().split('<CONCATE4SHUF>')
            for i, fd in enumerate(fds):
                print(s[i], file=fd)

        [ff.close() for ff in fds]

        os.remove(tpath)
        os.remove(tpath + '.shuf')

        return fnames

    def create_batch(self, sents, o):
        # Convert words to indices.
        assert o in ('src', 'dst')
        word2idx = self.src2idx if o == 'src' else self.dst2idx
        indices = []
        for sent in sents:
            x = [word2idx.get(word, 1) for word in (sent + [u"</S>"])]  # 1: OOV, </S>: End of Text
            indices.append(x)

        # Pad to the same length.
        maxlen = max([len(s) for s in indices])
        X = np.zeros([len(indices), maxlen], np.int32)
        for i, x in enumerate(indices):
            X[i, :len(x)] = x

        return X

    def indices_to_words(self, Y, o='dst'):
        assert o in ('src', 'dst')
        idx2word = self.idx2src if o == 'src' else self.idx2dst
        sents = []
        for y in Y: # for each sentence
            sent = []
            for i in y:  # For each word
                if i == 3:  # </S>
                    break
                w = idx2word[i]
                sent.append(w)
            sents.append(' '.join(sent))
        return sents


def expand_feed_dict(feed_dict):
    """If the key is a tuple of placeholders,
    split the input data then feed them into these placeholders.
    """
    new_feed_dict = {}
    for k, v in feed_dict.items():
        if type(k) is not tuple:
            new_feed_dict[k] = v
        else:
            # Split v along the first dimension.
            n = len(k)
            batch_size = v.shape[0]
            assert batch_size > 0
            span = batch_size // n
            remainder = batch_size % n
            base = 0
            for i, p in enumerate(k):
                if i < remainder:
                    end = base + span + 1
                else:
                    end = base + span
                new_feed_dict[p] = v[base: end]
                base = end
    return new_feed_dict


def padding_list_seqs(list_seqs, dtype=np.float32, pad=0.):
    len_x = [len(s) for s in list_seqs]

    size_batch = len(list_seqs)
    maxlen = max(len_x)

    shape_feature = tuple()
    for s in list_seqs:
        if len(s) > 0:
            shape_feature = np.asarray(s).shape[1:]
            break

    x = (np.ones((size_batch, maxlen) + shape_feature) * pad).astype(dtype)
    for idx, s in enumerate(list_seqs):
        x[idx, :len(s)] = s

    return x, len_x


def pad_to_split(batch, num_split):
    num_pad = num_split - len(batch) % num_split
    if num_pad != 0:
        if batch.ndim > 1:
            pad = np.tile(np.expand_dims(batch[0,:], 0), [num_pad]+[1]*(batch.ndim-1))
        elif batch.ndim == 1:
            pad = np.asarray([batch[0]] * num_pad, dtype=batch[0].dtype)
        batch = np.concatenate([batch, pad], 0)

    return batch


def size_bucket_to_put(l, buckets):
    for i, l1 in enumerate(buckets):
        if l < l1: return i, l1
    # logging.info("The sequence is too long: {}".format(l))
    return -1, 9999


class Sentence_iter(object):
    '''
    文件夹中文本文件遍历
    sentence_iter = MySentences('/some/directory')
    '''
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.strip().split()


def sparse_tuple_from(sequences):
    """
    Create a sparse representention of ``sequences``.

    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.int32)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape


def get_bucket(length_file, num_batch_tokens, idx_init=150):
    """
    enlarge idx_init can shrink the num of buckets
    """
    print('get the dataset info')
    list_len = []
    with open(length_file) as f:
        for line in f:
            length = int(line.strip().split()[1])
            list_len.append(length)

    hist, edges = np.histogram(list_len, bins=(max(list_len)-min(list_len)+1))
    list_num = []
    list_length = []
    for num, edge in zip(hist, edges):
        list_num.append(int(num))
        list_length.append(int(np.ceil(edge)))

    def next_idx(idx, energy):
        for i in range(idx, len(list_num), 1):
            if list_length[i]*sum(list_num[idx+1:i+1]) > energy:
                return i-1
        return

    M = num_batch_tokens
    b0 = int(M / list_length[idx_init])
    k = b0/sum(list_num[:idx_init+1])
    energy = M/k

    list_batchsize = [b0]
    list_boundary = [list_length[idx_init]]

    idx = idx_init
    while idx < len(list_num):
        idx = next_idx(idx, energy)
        if not idx:
            break
        if idx == idx_init:
            print('enlarge the idx_init!')
            break
        list_boundary.append(list_length[idx])
        list_batchsize.append(int(M / list_length[idx]))

    list_boundary.append(list_length[-1])
    list_batchsize.append(int(M/list_length[-1]))

    print('suggest boundaries: \n{}'.format(','.join(map(str, list_boundary))))
    print('corresponding batch size: \n{}'.format(','.join(map(str, list_batchsize))))


def align2stamp(align):
    if align is not None:
        list_stamps = []
        label_prev = align[0]
        for i, label in enumerate(align):
            if label_prev != label:
                list_stamps.append(i-1)
            label_prev = label
        list_stamps.append(i)
    else:
        list_stamps = None

    return np.array(list_stamps)


def align2bound(align):
    if align is not None:
        list_stamps = []
        label_prev = align[0]
        for label in align:
            list_stamps.append(1 if label_prev != label else 0)
            label_prev = label
    else:
        list_stamps = None

    return np.array(list_stamps)


def int2vector(seqs, seq_len, hidden_size=10, uprate=1.0):
    """
    m = np.array([[2,3,4],
                  [5,6,0]])
    int2vector(m, 5)
    """
    list_res = []
    list_length = []
    for seq, l in zip(seqs, seq_len):
        max_len = int(len(seq) * uprate)
        list_seq = []
        length = 0
        for m in seq[:l]:
            list_feat = []
            # org frame
            for i in np.arange(hidden_size-1, -1, -1):
                dec = np.power(2, i)
                p = m // dec
                m -= p * dec
                list_feat.append(p)
            list_seq.append(list_feat)
            length += 1

            # repeat frames
            for _ in range(int(np.random.random()*uprate)):
                list_seq.append(list_feat)
                length += 1

        # pad frames
        for _ in range(max_len - len(list_seq)):
            list_seq.append([0]*hidden_size)

        list_res.append(list_seq[:max_len])
        list_length.append(length)

    return np.array(list_res, np.float32), list_length
