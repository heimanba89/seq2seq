import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
#from tensorflow.contrib.rnn.python.ops import rnn_cell

def multi_bidirectional_rnn(cells, inputs, sequence_length=None, initial_state_fw=None, initial_state_bw=None,
                            dtype=None, parallel_iterations=None, swap_memory=False, time_major=False, scope=None,
                            time_pooling=None, pooling_avg=None, residual_connections=False, **kwargs):
    if not time_major:
        time_dim = 1
        batch_dim = 0
    else:
        time_dim = 0
        batch_dim = 1

    name = scope or "MultiBiRNN"

    output_states_fw = []
    output_states_bw = []
    for i, (cell_fw, cell_bw) in enumerate(cells):
        # Forward direction
        with tf.variable_scope('{}_FW_{}'.format(name, i)) as fw_scope:
            inputs_fw, output_state_fw = rnn.dynamic_rnn(
                cell=cell_fw, inputs=inputs, sequence_length=sequence_length, initial_state=initial_state_fw,
                dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=fw_scope
            )

        # Backward direction
        inputs_reversed = tf.reverse_sequence(
            input=inputs, seq_lengths=sequence_length, seq_dim=time_dim, batch_dim=batch_dim
        )

        with tf.variable_scope('{}_BW_{}'.format(name, i)) as bw_scope:
            inputs_bw, output_state_bw = rnn.dynamic_rnn(
                cell=cell_bw, inputs=inputs_reversed, sequence_length=sequence_length, initial_state=initial_state_bw,
                dtype=dtype, parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major,
                scope=bw_scope
            )

        inputs_bw_reversed = tf.reverse_sequence(
            input=inputs_bw, seq_lengths=sequence_length,
            seq_dim=time_dim, batch_dim=batch_dim
        )
        new_inputs = tf.concat(2, [inputs_fw, inputs_bw_reversed])

        if residual_connections and i < len(cells) - 1:
            # the output's dimension is twice that of the initial input (because of bidir)
            if i == 0:
                inputs = tf.tile(inputs, (1, 1, 2))  # FIXME: temporary solution
            inputs = new_inputs + inputs
        else:
            inputs = new_inputs

        if time_pooling and i < len(cells) - 1:
            inputs, sequence_length = apply_time_pooling(inputs, sequence_length, time_pooling[i], pooling_avg)

        output_states_fw.append(output_state_fw)
        output_states_bw.append(output_state_bw)

    return inputs, tf.concat(1, output_states_fw), tf.concat(1, output_states_bw)


def multi_rnn(cells, inputs, sequence_length=None, initial_state=None, dtype=None, parallel_iterations=None,
              swap_memory=False, time_major=False, scope=None, time_pooling=None, pooling_avg=None,
              residual_connections=False, **kwargs):
    name = scope or "MultiRNN"

    assert time_pooling is None or len(time_pooling) == len(cells) - 1

    output_states = []
    for i, cell in enumerate(cells):
        with tf.variable_scope('{}_{}'.format(name, i)) as scope:
            new_inputs, output_state = rnn.dynamic_rnn(
                cell=cell, inputs=inputs, sequence_length=sequence_length, initial_state=initial_state, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory, time_major=time_major, scope=scope
            )

        if residual_connections and i < len(cells) - 1:
            inputs = new_inputs + inputs
        else:
            inputs = new_inputs

        if time_pooling and i < len(cells) - 1:
            inputs, sequence_length = apply_time_pooling(inputs, sequence_length, time_pooling[i], pooling_avg)

        output_states.append(output_state)

    return inputs, tf.concat(1, output_states)


def apply_time_pooling(inputs, sequence_length, stride, pooling_avg=False):
    shape = [tf.shape(inputs)[0], tf.shape(inputs)[1], inputs.get_shape()[2].value]

    if pooling_avg:
        inputs_ = [inputs[:, i::stride, :] for i in range(stride)]

        max_len = tf.shape(inputs_[0])[1]
        for k in range(1, stride):
            len_ = tf.shape(inputs_[k])[1]
            paddings = tf.pack([[0, 0], [0, max_len - len_], [0, 0]])
            inputs_[k] = tf.pad(inputs_[k], paddings=paddings)

        inputs = tf.reduce_sum(inputs_, 0) / len(inputs_)
    else:
        inputs = inputs[:, ::stride, :]

    inputs = tf.reshape(inputs, tf.pack([shape[0], tf.shape(inputs)[1], shape[2]]))
    sequence_length = (sequence_length + stride - 1) // stride  # rounding up

    return inputs, sequence_length


def unsafe_decorator(fun):
    """
    Wrapper that automatically handles the `reuse' parameter.
    This is rather unsafe, as it can lead to reusing variables
    by mistake, without knowing about it.
    """

    def fun_(*args, **kwargs):
        try:
            return fun(*args, **kwargs)
        except ValueError as e:
            if 'reuse' in str(e):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    return fun(*args, **kwargs)
            else:
                raise e

    return fun_


get_variable_unsafe = unsafe_decorator(tf.get_variable)
GRUCell_unsafe = unsafe_decorator(rnn_cell.GRUCell)
BasicLSTMCell_unsafe = unsafe_decorator(rnn_cell.BasicLSTMCell)
MultiRNNCell_unsafe = unsafe_decorator(rnn_cell.MultiRNNCell)
linear_unsafe = unsafe_decorator(rnn_cell._linear)
multi_rnn_unsafe = unsafe_decorator(multi_rnn)
multi_bidirectional_rnn_unsafe = unsafe_decorator(multi_bidirectional_rnn)


class MultiRNNCell(rnn_cell.RNNCell):
    """
    Same as rnn_cell.MultiRNNCell, except it accepts an additional `residual_connections` parameter
    """

    def __init__(self, cells, state_is_tuple=False, residual_connections=False):
        self._cells = cells
        self._state_is_tuple = state_is_tuple
        self._residual_connections = residual_connections

    @property
    def state_size(self):
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum([cell.state_size for cell in self._cells])

    @property
    def output_size(self):
        return self._cells[-1].output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            for i, cell in enumerate(self._cells):
                with tf.variable_scope("Cell%d" % i):
                    if self._state_is_tuple:
                        cur_state = state[i]
                    else:
                        cur_state = tf.slice(state, [0, cur_state_pos], [-1, cell.state_size])
                        cur_state_pos += cell.state_size
                    new_inp, new_state = cell(cur_inp, cur_state)
                    if self._residual_connections and i < len(self._cells) - 1:
                        cur_inp = cur_inp + new_inp
                    else:
                        cur_inp = new_inp
                    new_states.append(new_state)
        new_states = (tuple(new_states) if self._state_is_tuple else tf.concat(1, new_states))
        return cur_inp, new_states
