import re
import codecs
import unicodedata
import numpy as np
import editdistance as ed
from tqdm import tqdm


# Constants
SPACE_TOKEN = '<space>'
SPACE_INDEX = 0
FIRST_INDEX = ord('a') - 1  # 0 is reserved to space


def unpadding(list_idx, token2idx):
    """
    for the 1d array
    Demo:
        a = np.array([2,2,3,4,5,1,0,0,0])
        unpadding(a, 1)
        # array([2, 2, 3, 4, 5])
    """
    eos_idx = token2idx['<eos>']
    min_idx = token2idx['<eos>']
    max_idx= token2idx['<blk>']

    # cut the sent at <eos>
    end_idx = np.where(list_idx==eos_idx)[0]
    end_idx = end_idx[0] if len(end_idx)>0 else None
    list_idx = list_idx[:end_idx]

    # remove specical tokens
    list_idx = list_idx[np.where(list_idx>min_idx)]
    list_idx = list_idx[np.where(list_idx<max_idx)]

    return list_idx


def batch_cer(result, reference, token2idx):
    """
    result and reference are lists of tokens
    eos_idx is the padding token or eos token
    """
    batch_dist = 0
    batch_len = 0
    for res, ref in zip(result, reference):
        res = unpadding(res, token2idx)
        ref = unpadding(ref, token2idx)
        batch_dist += ed.eval(res, ref)
        batch_len += len(ref)

    return batch_dist, batch_len


def batch_wer(result, reference, idx2token, token2idx, unit):
    """
    Args:
        result and reference are lists of tokens idx
        eos_idx is the padding token or eos token idx
        idx2token is a dict form idx to token
        seperator is what to join the tokens. If token is char, seperator is '';
            if token is word, seperator is ' '.
        eos_idx is the padding token idx or the eos token idx
    """
    batch_dist = 0
    batch_len = 0
    for res, ref in zip(result, reference):
        list_res_txt = array2text(res, unit, idx2token, token2idx).split()
        list_ref_txt = array2text(ref, unit, idx2token, token2idx).split()
        # print(' '.join(list_res_txt))
        # print(' '.join(list_ref_txt))
        batch_dist += ed.eval(list_res_txt, list_ref_txt)
        batch_len += len(list_ref_txt)

    return batch_dist, batch_len


def array2text(res, unit, idx2token, token2idx):
    """
    char: the english characters including blank. The Chinese characters belongs to the word
    for the 1d array
    """
    res = unpadding(res, token2idx)
    if unit == 'char':
        list_res_txt = array_idx2char(res, idx2token, seperator='')
    elif unit == 'word':
        list_res_txt = array_idx2char(res, idx2token, seperator=' ')
    elif unit == 'subword':
        list_res_txt = array_idx2char(res, idx2token, seperator=' ').replace('@@ ', '')
    else:
        raise NotImplementedError('not know unit!')

    return list_res_txt


def array_idx2char(array_idx, idx2token, seperator=''):
    # array_idx = np.asarray(array_idx, dtype=np.int32)
    if len(array_idx)==0 or np.isscalar(array_idx[0]):
        return seperator.join(idx2token[i] for i in array_idx)
    else:
        return [array_idx2char(i, idx2token, seperator=seperator) for i in array_idx]


def array_char2idx(list_idx, token2idx, seperator=''):
    """
    list of chars to the idx array and length
    """
    from utils.tools import padding_list_seqs
    sents = []
    if seperator:
        for sent in list_idx:
            sents.append([token2idx[token] for token in sent.split(seperator)])
    else:
        for sent in list_idx:
            sents.append([token2idx[token] for token in list(sent)])
    padded, len_seqs = padding_list_seqs(sents, dtype=np.int32)

    return padded, len_seqs


def text_to_char_array(original):
    """
    Given a Python string ``original``, map characters
    to integers and return a np array representing the processed string.
    """
    # Create list of sentence's words w/spaces replaced by ''
    result = original.replace(' ', '  ')
    result = result.split(' ')

    # Tokenize words into letters adding in SPACE_TOKEN where required
    result = np.hstack([SPACE_TOKEN if xt == '' else list(xt) for xt in result])

    # Return characters mapped into indicies
    return np.asarray([SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result])


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    a =[0,1]
    a[0] =text_to_char_array(" look at ___")
    a[1] =text_to_char_array("a son shane __")
    (array([[ 0,  0],
            [ 0,  1],
            [ 0,  2],
            [ 0,  3],
            [ 0,  4],
            [ 0,  5],
            [ 0,  6],
            [ 0,  7],
            [ 0,  8],
            [ 0,  9],
            [ 0, 10],
            [ 0, 11],
            [ 0, 12],
            [ 1,  0],
            [ 1,  1],
            [ 1,  2],
            [ 1,  3],
            [ 1,  4],
            [ 1,  5],
            [ 1,  6],
            [ 1,  7],
            [ 1,  8],
            [ 1,  9],
            [ 1, 10],
            [ 1, 11],
            [ 1, 12],
            [ 1, 13]]),
     array([ 0,  0, 12, 15, 15, 11,  0,  1, 20,  0, -1, -1, -1,  1,  0, 19, 15,
            14,  0, 19,  8,  1, 14,  5,  0, -1, -1], dtype=int32),
     array([ 2, 14]))
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape


def get_N_gram(iterator, n):
    """
    return :
        [(('ih', 'sil', 'k'), 1150),
         (('ih', 'n', 'sil'), 1067),
         ...],
         num of all the n-gram, i.e. num of tokens
    """
    from nltk import ngrams, FreqDist

    _n_grams = FreqDist(ngrams(iterator, n))

    return _n_grams


def get_dataset_ngram(text_file, n, k, savefile=None, split=5000):
    """
    Simply concatenate all sents into one will bring in noisy n-gram at end of each sent.
    Here we count ngrams for each sent and sum them up.
    """
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
        for sent in tqdm(text):
            ngram = get_N_gram(iter_in_sent(sent.strip()), n)
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


def ngram2kernel(ngram, n, k, dim_output):
    kernel = np.zeros([n, dim_output, k], dtype=np.float32)
    list_py = []
    for i, (z, py) in enumerate(ngram):
        list_py.append(py)
        for j, token in enumerate(z):
            kernel[j][token][i] = 1.0
    py = np.array(list_py, dtype=np.float32)

    return kernel, py
