import tensorflow as tf
import logging
import numpy as np


def get_tensor_len(tensor):
    if tensor.get_shape().ndims == 3:
        return tf.reduce_sum(tf.cast((tf.reduce_max(tf.abs(tensor), -1) > 0), tf.int32), -1)
    elif tensor.get_shape().ndims == 2:
        return tf.reduce_sum(tf.cast(tf.abs(tensor) > 0, tf.int32), -1)


def learning_rate_decay(config, global_step):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(config.train.warmup_steps)
    global_step = tf.to_float(global_step)
    return config.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def shift_right(input, pad=2):
    """Shift input tensor right to create decoder input. '2' denotes <S>"""
    return tf.concat((tf.ones_like(input[:, :1]) * pad, input[:, :-1]), 1)


def embedding(x, vocab_size, dense_size, name=None, reuse=None, kernel=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors."""
    with tf.variable_scope(
        name, default_name="embedding", values=[x], reuse=reuse):
        if kernel is not None:
            embedding_var = kernel
        else:
            embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        output = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            output *= multiplier
        return output


def sampleFrames(align):
    """
    align:
    return please ignore the value in sample where in align is 0
    """
    align = tf.cast(align, tf.float32)
    pad = tf.zeros([align.shape[0], 1], dtype=tf.float32)
    _align = tf.concat([pad, align[:, :-1]], 1)
    sample = tf.cast((_align + (align-_align)*tf.random.uniform(align.shape))*tf.cast(align > 0, tf.float32), tf.int32)

    return sample


def build_optimizer(args, lr=0.5, type='adam'):
    if type == 'adam':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.opti.peak,
            decay_steps=args.opti.decay_steps,
            decay_rate=0.5,
            staircase=False)
        optimizer = tf.keras.optimizers.Adam(
            lr_schedule,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9)
    elif type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            lr=lr,
            momentum=0.9,
            decay=0.98)

    return optimizer


def get_session(sess):
        session = sess
        while type(session).__name__ != 'Session':
            session = session._sess
        return session

def range_batch(shape, range_down=True, dtype=tf.int32):
    """
    sess.run(range_batch([2,5]))
    range_down=False:
        array([[0, 1, 2, 3, 4],
               [0, 1, 2, 3, 4]], dtype=int32)
    range_down=True:
        array([[0, 0, 0, 0, 0],
               [1, 1, 1, 1, 1]], dtype=int32)
    """
    if not range_down:
        result = tf.tile([tf.range(shape[1], dtype=dtype)],
                       [shape[0], 1])
    else:
        result = tf.tile(tf.reshape(tf.range(shape[0], dtype=dtype), (-1, 1)),
                   [1, shape[1]])
    return result


# TODO
def label_smoothing(z, cr=0.8):
    # Label smoothing
    table = tf.convert_to_tensor([[cr, 1.-cr]])

    return tf.nn.embedding_lookup(table, z)


def dense_sequence_to_sparse(seq, len_seq):
    '''convert sequence dense representations to sparse representations
    Args:
        seq: the dense seq as a [batch_size x max_length] tensor
        len_seq: the sequence lengths as a [batch_size] vector
    Returns:
        the sparse tensor representation of the seq

    the reverse op:
        tf.sparse_tensor_to_dense(sp_input, default_value=0, validate_indices=True, name=None)
        [[1, 0, 0, 0]
         [0, 0, 2, 0]
         [0, 0, 0, 0]]
        indices：[[0, 0], [1, 2]]
        values：[1, 2]
        dense_shape：[3, 4]
        the default value is `0`

        a_dense = tf.sparse_to_dense(
            sparse_indices=a_sparse.indices,
            output_shape=a_sparse.dense_shape,
            sparse_values=a_sparse.values,
            default_value=0)
    '''
    with tf.name_scope('dense_sequence_to_sparse'):
        #get all the non padding seq
        indices = tf.cast(get_indices(len_seq), tf.int64)
        #create the values
        values = tf.gather_nd(seq, indices)
        #the shape
        shape = tf.cast(tf.shape(seq), tf.int64)
        sparse = tf.SparseTensor(indices, values, shape)

    return sparse


def batch_pad(p, length, pad, direct='head'):
    """
    add the length
    Demo:
        p = tf.ones([4, 3], dtype=tf.float32)
        pad = tf.convert_to_tensor(0.0)
        sess.run(right_length_rows(p, 1, pad, direct='head'))
        array([[ 0.,  1.,  1.,  1.],
               [ 0.,  1.,  1.,  1.],
               [ 0.,  1.,  1.,  1.],
               [ 0.,  1.,  1.,  1.]], dtype=float32)
    """
    assert type(length) is int
    if direct == 'head':
        res = tf.concat([tf.fill(dims=[tf.shape(p)[0], length], value=pad), p],
                        axis=1)
    elif direct == 'tail':
        res = tf.concat([p, tf.fill(dims=[tf.shape(p)[0], length], value=pad)],
                        axis=1)
    else:
        raise NotImplementedError

    return res


def batch2D_pad_to(p, length, pad):
    """
    expend the 2d tensor to assigned length
    """
    length_p = tf.shape(p)[1]
    pad_length = tf.reduce_max([length_p, length])-length_p

    pad = tf.cast(tf.fill(dims=[tf.shape(p)[0], pad_length], value=pad), dtype=p.dtype)
    res = tf.concat([p, pad], axis=1)

    return res


def batch3D_pad_to(p, length, pad=0.0):
    """
    expend the 3d tensor to assigned length
    """
    length_p = tf.shape(p)[1]
    pad_length = tf.reduce_max([length_p, length])-length_p

    pad = tf.cast(tf.fill(dims=[tf.shape(p)[0], pad_length, tf.shape(p)[-1]], value=pad), dtype=p.dtype)
    res = tf.concat([p, pad], axis=1)[:, :length, :]

    return res


def pad_to(p, length, pad=0.0, axis=1):
    """
    expend the arbitrary shape tensor to assigned length along the assigned axis
    demo:
        p = tf.ones([10, 5])
        pad_to(p, 11)
        <tf.Tensor 'concat_2:0' shape=(10, 11) dtype=float32>
    """
    length_p = tf.shape(p)[axis]
    pad_length = tf.reduce_max([length_p, length])-length_p

    shape = p.get_shape()
    pad_shape = [*shape]
    pad_shape[axis] = pad_length
    pad_tensor = tf.ones(pad_shape, dtype=p.dtype) * pad
    res = tf.concat([p, pad_tensor], axis=axis)

    return res


def pad_to_same(list_tensors):
    """
    pad all the tensors to the same length , given the length info.
    """
    list_lens = []
    for tensor in list_tensors:
        list_lens.append(tf.shape(tensor)[1])
    len_max = tf.reduce_max(tf.stack(list_lens, 0))
    list_padded = []
    for tensor in list_tensors:
        list_padded.append(batch2D_pad_to(tensor, len_max, 0))

    return list_padded


def right_shift_rows(p, shift, pad):
    assert type(shift) is int

    return tf.concat([tf.fill(dims=[tf.shape(p)[0], 1], value=pad), p[:, :-shift]], axis=1)


def left_shift_rows(p, shift, pad):
    assert type(shift) is int

    return tf.concat([p[:, shift:], tf.fill(dims=[tf.shape(p)[0], 1], value=pad)], axis=1)


def sparse_shrink(sparse, pad=0):
    """
    sparsTensor to shrinked dense tensor:
    from:
     [[x 1 x x 3],
      [2 x x 5 4]]
    to:
     [[1 3 0],
      [2 5 4]]
    """
    dense = tf.sparse_tensor_to_dense(sparse, default_value=-1)
    mask = (dense>=0)
    len_seq = tf.reduce_sum(tf.to_int32(mask), -1)
    indices = get_indices(len_seq)
    values = sparse.values
    shape = [sparse.dense_shape[0], tf.to_int64(tf.reduce_max(len_seq))]
    sparse_shrinked = tf.SparseTensor(indices, values, shape)
    seq = tf.sparse_tensor_to_dense(sparse_shrinked, default_value=pad)

    return seq, len_seq, sparse_shrinked


def acoustic_shrink(distribution_acoustic, len_acoustic, dim_output):
    """
    filter out the distribution where blank_id dominants.
    the blank_id default to be dim_output-1.
    incompletely tested
    the len_no_blank will be set one if distribution_acoustic is all blank dominanted

    """
    blank_id = dim_output - 1
    no_blank = tf.to_int32(tf.not_equal(tf.argmax(distribution_acoustic, -1), blank_id))
    mask_acoustic = tf.sequence_mask(len_acoustic, maxlen=tf.shape(distribution_acoustic)[1], dtype=no_blank.dtype)
    no_blank = mask_acoustic*no_blank
    len_no_blank = tf.reduce_sum(no_blank, -1)

    batch_size = tf.shape(no_blank)[0]
    seq_len = tf.shape(no_blank)[1]

    # the repairing, otherwise the length would be 0
    no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        no_blank,
        tf.concat([tf.ones([batch_size, 1], dtype=tf.int32),
                   tf.zeros([batch_size, seq_len-1], dtype=tf.int32)], 1)
    )
    len_no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        len_no_blank,
        tf.ones_like(len_no_blank, dtype=tf.int32)
    )

    batch_size = tf.size(len_no_blank)
    max_len = tf.reduce_max(len_no_blank)
    acoustic_shrinked_init = tf.zeros([1, max_len, dim_output])

    def step(i, acoustic_shrinked):
        shrinked = tf.gather(distribution_acoustic[i], tf.reshape(tf.where(no_blank[i]>0), [-1]))
        shrinked_paded = pad_to(shrinked, max_len, axis=0)
        acoustic_shrinked = tf.concat([acoustic_shrinked,
                                       tf.expand_dims(shrinked_paded, 0)], 0)

        return i+1, acoustic_shrinked

    i, acoustic_shrinked = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=step,
        loop_vars=[0, acoustic_shrinked_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None, dim_output])]
    )
    # acoustic_shrinked = tf.gather_nd(distribution_acoustic, tf.where(no_blank>0))

    acoustic_shrinked = acoustic_shrinked[1:, :, :]

    return acoustic_shrinked, len_no_blank


def acoustic_hidden_shrink(distribution_acoustic, hidden, len_acoustic, blank_id, hidden_size, num_avg=1):
    """
    filter the hidden where blank_id dominants in distribution_acoustic.
    the blank_id default to be dim_output-1.
    incompletely tested
    the len_no_blank will be set one if distribution_acoustic is all blank dominanted
    shrink the hidden instead of distribution_acoustic
    """
    no_blank = tf.to_int32(tf.not_equal(tf.argmax(distribution_acoustic, -1), blank_id))
    mask_acoustic = tf.sequence_mask(len_acoustic, maxlen=tf.shape(distribution_acoustic)[1], dtype=no_blank.dtype)
    no_blank *= mask_acoustic
    len_no_blank = tf.reduce_sum(no_blank, -1)

    batch_size = tf.shape(no_blank)[0]
    seq_len = tf.shape(no_blank)[1]

    # the patch, the length of shrunk hidden is at least 1
    no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        no_blank,
        tf.concat([tf.ones([batch_size, 1], dtype=tf.int32),
                   tf.zeros([batch_size, seq_len-1], dtype=tf.int32)], 1)
    )
    len_no_blank = tf.where(
        tf.not_equal(len_no_blank, 0),
        len_no_blank,
        tf.ones_like(len_no_blank, dtype=tf.int32)
    )

    max_len = tf.reduce_max(len_no_blank)
    hidden_shrunk_init = tf.zeros([1, max_len, hidden_size])

    # average the hidden of n frames
    if num_avg == 3:
        hidden = (hidden + \
                tf.concat([hidden[:, 1:, :], hidden[:, -1:, :]], 1) + \
                tf.concat([hidden[:, :1, :], hidden[:, :-1, :]], 1)) / num_avg
    elif num_avg == 5:
        hidden = (hidden + \
                tf.concat([hidden[:, 1:, :], hidden[:, -1:, :]], 1) + \
                tf.concat([hidden[:, 2:, :], hidden[:, -2:, :]], 1) + \
                tf.concat([hidden[:, :2, :], hidden[:, :-2, :]], 1) + \
                tf.concat([hidden[:, :1, :], hidden[:, :-1, :]], 1)) / num_avg

    def step(i, hidden_shrunk):
        # loop over the batch
        shrunk = tf.gather(hidden[i], tf.reshape(tf.where(no_blank[i]>0), [-1]))
        shrunk_paded = pad_to(shrunk, max_len, axis=0)
        hidden_shrunk = tf.concat([hidden_shrunk,
                                   tf.expand_dims(shrunk_paded, 0)], 0)

        return i+1, hidden_shrunk

    i, hidden_shrunk = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=step,
        loop_vars=[0, hidden_shrunk_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None, hidden_size])]
    )
    hidden_shrunk = hidden_shrunk[1:, :, :]

    return hidden_shrunk, len_no_blank


def alignment_shrink(align, blank_id, pad_id=0):
    """
    //treat the alignment as a sparse tensor where the pad is blank.
    get the indices, values and new_shape
    finally, use the `tf.sparse_tensor_to_dense`

    loop along the batch dim
    """
    batch_size = tf.shape(align)[0]
    len_seq = tf.reduce_sum(tf.to_int32(tf.not_equal(align, blank_id)), -1)

    max_len = tf.reduce_max(len_seq)
    noblank_init = tf.zeros([1, max_len], dtype=align.dtype)

    def step(i, noblank):
        noblank_i = tf.reshape(tf.gather(align[i],
                                         tf.where(tf.not_equal(align[i], blank_id))), [-1])
        pad = tf.ones([max_len-tf.shape(noblank_i)[0]], dtype=align.dtype) * pad_id
        noblank_i = tf.concat([noblank_i, pad], -1)
        noblank = tf.concat([noblank, noblank_i[None, :]], 0)

        return i+1, noblank

    _, noblank = tf.while_loop(
        cond=lambda i, *_: tf.less(i, batch_size),
        body=step,
        loop_vars=[0, noblank_init],
        shape_invariants=[tf.TensorShape([]),
                          tf.TensorShape([None, None])]
    )

    return noblank[1:], len_seq


def get_indices(len_seq):
    '''get the indices corresponding to sequences (and not padding)
    Args:
        len_seq: the len_seqs as a N-D tensor
    Returns:
        A [sum(len_seq) x N-1] Tensor containing the indices'''

    with tf.name_scope('get_indices'):

        numdims = len(len_seq.shape)

        #get the maximal length
        max_length = tf.reduce_max(len_seq)

        sizes = tf.shape(len_seq)

        range_tensor = tf.range(max_length)
        for i in range(1, numdims):
            tile_dims = [1]*i + [sizes[i]]
            range_tensor = tf.tile(tf.expand_dims(range_tensor, i), tile_dims)

        indices = tf.where(tf.less(range_tensor,
                                   tf.expand_dims(len_seq, numdims)))

    return indices


def state2tensor(state):
    """
    for lstm cell

    demo:
        cell = make_multi_cell(
            num_cell_units=10,
            num_layers=2,
            is_train=True,
            keep_prob=0.9)
        x = cell.zero_state(3, tf.float32)
        state2tensor(x):
        <tf.Tensor 'stack_10:0' shape=(3, 4, 10) dtype=float32>
    """
    import itertools
    list_tensors = list(itertools.chain.from_iterable(state))
    tensor = tf.stack(list_tensors, 1)

    return tensor


def tensor2state(tensor):
    """
    tensor: [batch, 2*num_layer, hidden_size]

    demo:
        cell = make_multi_cell(
            num_cell_units=10,
            num_layers=2,
            is_train=True,
            keep_prob=0.9)
        cell.zero_state(3, tf.float32):
        (LSTMStateTuple(c=<zeros:0' shape=(3, 10)>,
                        h=<zeros_1:0' shape=(3, 10))>,
         LSTMStateTuple(c=<zeros:0' shape=(3, 10)>,
                        h=<zeros_1:0' shape=(3, 10))>)

        m = tf.placeholder(tf.float32, [None, 2*2, 10])
        tensor2state(m):
        (LSTMStateTuple(c=<shape=(?, 10)>,
                        h=<shape=(?, 10)>),
         LSTMStateTuple(c=<shape=(?, 10)>,
                        h=<shape=(?, 10)>))
    """
    from tensorflow.contrib.rnn import LSTMStateTuple
    cells = []
    list_tensors = tf.unstack(tensor, axis=1)
    for i in range(int(len(list_tensors)/2)):
        cells.append(LSTMStateTuple(list_tensors[2*i], list_tensors[2*i+1]))

    return tuple(cells)


def size_variables():
    total_size = 0
    all_weights = {v.name: v for v in tf.trainable_variables()}
    print('='*160)
    for v_name in sorted(list(all_weights)):
        v = all_weights[v_name]
        v_size = int(np.prod(np.array(v.shape.as_list())))
        print("Weight    %s\tshape    %s\tsize    %d" % (v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size))
        total_size += v_size
    print('='*160)
    print("Total trainable variables size: %d" % total_size)


def smoothing_cross_entropy(logits, labels, vocab_size, confidence):
    """Cross entropy with label smoothing to limit over-confidence."""
    with tf.name_scope("smoothing_cross_entropy"):
        # Low confidence is given to all non-true labels, uniformly.
        low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
        # Normalizing constant is the best cross-entropy value with soft targets.
        # We subtract it just for readability, makes no difference on learning.
        normalizing = -(confidence * tf.log(confidence) + tf.to_float(
            vocab_size - 1) * low_confidence * tf.log(low_confidence + 1e-20))
        # Soft targets.
        soft_targets = tf.one_hot(
            tf.cast(labels, tf.int32),
            depth=vocab_size,
            on_value=confidence,
            off_value=low_confidence)
        try:
            xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=soft_targets)
        except:
            xentropy = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=soft_targets)
    return xentropy - normalizing


def smoothing_distribution(distributions, vocab_size, confidence):
    share = 1-confidence
    num_targets = tf.reduce_sum(tf.to_int32(distributions>0.0), -1)
    num_zeros = vocab_size - num_targets
    reduce = tf.tile(tf.expand_dims(share/tf.to_float(num_targets), -1), [1, 1, vocab_size])
    add = tf.tile(tf.expand_dims(share/tf.to_float(num_zeros), -1), [1, 1, vocab_size])
    distribution_smoothed = (distributions-reduce)*tf.to_float(distributions>0) +\
            add*tf.to_float(distributions<1e-6)

    return distribution_smoothed

#============================================================================
#  building model
#============================================================================
def choose_device(op, device, default_device):
    if op.type.startswith('Variable'):
        device = default_device
    return device


def l2_penalty(iter_variables):
    l2_penalty = 0
    for v in iter_variables:
        if 'biase' not in v.name:
            l2_penalty += tf.nn.l2_loss(v)
    return l2_penalty

#============================================================================
#  learning rate
#============================================================================
def lr_decay_with_warmup(global_step, warmup_steps, hidden_units):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(warmup_steps)
    global_step = tf.to_float(global_step)
    return hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def warmup_exponential_decay(global_step, warmup_steps, peak, decay_rate, decay_steps):
    print('warmup_steps', warmup_steps, 'peak', peak, 'decay_rate', decay_rate, 'decay_steps', decay_steps)
    warmup_steps = tf.to_float(warmup_steps)
    global_step = tf.to_float(global_step)
    # return peak * global_step / warmup_steps
    return tf.where(global_step <= warmup_steps,
                    peak * global_step / warmup_steps,
                    peak * decay_rate ** ((global_step - warmup_steps) / decay_steps))


def stepped_down_decay(global_step, learning_rate, decay_rate, decay_steps):
    decay_rate = tf.to_float(decay_rate)
    decay_steps = tf.to_float(decay_steps)
    learning_rate = tf.to_float(learning_rate)
    global_step = tf.to_float(global_step)

    return learning_rate * decay_rate ** (global_step // decay_steps)


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """
    global_step: int64 (scalar) tensor representing global step.
    learning_rate_base: base learning rate.
    total_steps: total number of training steps.
    warmup_learning_rate: initial learning rate for warm up.
    warmup_steps: number of warmup steps.
    hold_base_rate_steps: Optional number of steps to hold base learning rate
      before decaying.
    """
    if learning_rate_base < warmup_learning_rate:
        raise ValueError('learning_rate_base must be larger '
                     'or equal to warmup_learning_rate.')
    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                     'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
                          3.1416 *
                          (tf.cast(global_step, tf.float32) - warmup_steps - hold_base_rate_steps)
                          / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = tf.where(global_step > warmup_steps + hold_base_rate_steps,
                             learning_rate, learning_rate_base)
    if warmup_steps > 0:
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(global_step, tf.float32) + warmup_learning_rate
        learning_rate = tf.where(global_step < warmup_steps, warmup_rate, learning_rate)

    return tf.where(global_step > total_steps, 0.0, learning_rate, name='learning_rate')


def exponential_decay(global_step, lr_init, decay_rate, decay_steps, lr_final=None):
    lr = tf.train.exponential_decay(lr_init, global_step, decay_steps, decay_rate, staircase=True)
    if lr_final:
        lr = tf.cond(tf.less(lr, lr_final),
                lambda: tf.constant(lr_final),
                lambda: lr)
    return lr


def create_embedding(size_vocab, size_embedding, name='embedding'):
    if type(size_embedding) == int:
        with tf.device("/cpu:0"):
            embed_table = tf.get_variable(name, [size_vocab, size_embedding])
    else:
        embed_table = None

    return embed_table


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_v, depth)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        return output, attention_weights


if __name__ == '__main__':
    global_step = tf.train.get_or_create_global_step()
    op_add = tf.assign_add(global_step, 1)
    # lr = cosine_decay_with_warmup(global_step, 0.1, 100000, 0.01, 10, 20)
    # lr = lr_decay_with_warmup(
    #     global_step,
    #     warmup_steps=10000,
    #     hidden_units=256)
    lr = stepped_down_decay(global_step,
                            learning_rate=0.002,
                            decay_rate=0.94,
                            decay_steps=3000)
    # lr = warmup_exponential_decay(global_step,
    #                               warmup_steps=10000,
    #                               peak=0.001,
    #                               decay_rate=0.5,
    #                               decay_steps=1000)

    with tf.train.MonitoredTrainingSession() as sess:
        list_x = []; list_y = []
        for i in range(50000):
            x, y = sess.run([global_step, lr])
            list_x.append(x)
            list_y.append(y)
            sess.run(op_add)

    import matplotlib.pyplot as plt
    plt.plot(list_x, list_y)
    plt.show()
