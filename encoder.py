import tensorflow as tf


def cell(size, type, dropout=None, proj=None, layers=1):
    cells = []
    cell = None
    for _ in range(layers):
        if type == "LSTM":
            cell= tf.contrib.rnn.BasicLSTMCell(size)
        elif type == "GRU":
            cell= tf.contrib.rnn.GRUCell(size)
        if proj:
            cell = tf.contrib.rnn.OutputProjectionWrapper(cell, proj)
        if dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, dtype=tf.float32, input_keep_prob=dropout, output_keep_prob=1.0, state_keep_prob=1.0)
        cells.append(cell)
    if layers==1:
        return cell
    else:
        return tf.contrib.rnn.MultiRNNCell(cells)

def morph_encoder(chars, chars_length, size, cell_type="LSTM", dropout=None):
    '''Here we take a batch of words and compute their morphological embeddings, i.e. a hidden representation
    of a RNN over their characters'''
    with tf.variable_scope("MorphologicEncoder"):
        with tf.variable_scope("fw"):
            char_rnn_cell_fw = cell(size, cell_type, dropout=dropout)
        with tf.variable_scope("bw"):
            char_rnn_cell_bw = cell(size, cell_type, dropout=dropout)
        _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=char_rnn_cell_fw,
                                                               cell_bw=char_rnn_cell_bw,
                                                               inputs=chars,
                                                               sequence_length=chars_length,
                                                               dtype=tf.float32)

    return  tf.concat((fw_state.h, bw_state.h), axis=1)



class ContextualEncoder(object):


    def __init__(self, sentences, config):
        self.sentences = sentences

        batch_size = tf.shape(sentences)[0]
        with tf.name_scope("ContextEncoder"):
            with tf.name_scope("CharacterEmbeddingsLookup"):
                embedding = tf.get_variable("embedding", [config.char_vocab_size, config.char_embed_size],
                                            dtype=tf.float32)
                self.embedding = embedding
                chars = tf.nn.embedding_lookup(embedding,
                                               self.sentences)  # batch_size x sentence_max_len x word_max_len x char_embed_size
                chars = tf.reshape(chars,
                                   [-1, config.word_max_len,
                                    config.char_embed_size])

            with tf.name_scope("MorphologicalEncoding"):
                with tf.variable_scope("MorphologicEncoder"):
                    with tf.variable_scope("fw"):
                        char_rnn_cell_fw = cell(config.morph_rnn_size, "LSTM", dropout=config.morph_encoder_keep_prob)
                    with tf.variable_scope("bw"):
                        char_rnn_cell_bw = cell(config.morph_rnn_size, "LSTM", dropout=config.morph_encoder_keep_prob)
                    _, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(cell_fw=char_rnn_cell_fw,
                                                                              cell_bw=char_rnn_cell_bw,
                                                                              inputs=chars,
                                                                              dtype=tf.float32)

                output = tf.concat((fw_state.h, bw_state.h), axis=1)

                self.segments_encodings = tf.reshape(output,
                                                     [-1, config.sentence_max_len,
                                                      2 * config.morph_rnn_size])

            with tf.name_scope("ContextEncoding"):
                with tf.name_scope("ContextBRNN"):
                    with tf.variable_scope("ContextEncoder"):
                        with tf.variable_scope("fw"):
                            sentences_rnn_cell_fw = cell(config.sentence_rnn_size, "LSTM",
                                                             dropout=config.w2f_keep_prob)
                        with tf.variable_scope("bw"):
                            sentences_rnn_cell_bw = cell(config.sentence_rnn_size, "LSTM",
                                                             dropout=config.w2f_keep_prob)

                        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=sentences_rnn_cell_fw,
                                                                     cell_bw=sentences_rnn_cell_bw,
                                                                     inputs=self.segments_encodings,
                                                                     dtype=tf.float32)

                        # batch_size x sentence_max_len x sentence_rnn_size

                        self.left_context = outputs[0]

                        self.right_context = outputs[1]

                        self.contextual_encodings = tf.concat((self.left_context, self.right_context), axis=2)

                with tf.variable_scope("MaskingTarget"):

                    left_context = tf.concat((tf.zeros([batch_size, 1, config.sentence_rnn_size]),
                                              outputs[0][:, :-1, :]), axis=1)
                    right_context = tf.concat((outputs[1][:, 1:, :],
                                               tf.zeros([batch_size, 1, config.sentence_rnn_size])),
                                              axis=1)

                    self.masked_contextual_encodings = tf.concat((left_context, right_context),
                                                      axis=2)  # batch_size x sentence_max_len x 2*sentence_rnn_size
