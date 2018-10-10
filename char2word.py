import tensorflow as tf
from encoder import ContextualEncoder
import numpy as np
import nltk
import codecs
from glob import glob
import os
import math
from collections import Counter
from argparse import ArgumentParser



eps = 1e-12

_PAD = 0
_GO = 1
_EOW = 2
_UNK = 3
chars = ['_PAD', '_GO', '_EOW', '_UNK', ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')',
         '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
         'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
         'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
         'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
         'z', '{', '|', '}', '~']
char_dict = {}
for char in chars:
    char_dict[char] = len(char_dict)

class Model():
    def __init__(self, elements_chars, labels=None, config=None, is_test=False):


        self.global_step = tf.contrib.framework.get_or_create_global_step()

        self.elements_chars = elements_chars  # 1 x sentence_max_len x word_max_len

        with tf.variable_scope("Encodings"):
            self.encoder = ContextualEncoder(elements_chars, config)
            self.words_encodings = self.encoder.segments_encodings
            print(self.encoder.masked_contextual_encodings)
            h1 = tf.layers.dense(inputs=self.encoder.masked_contextual_encodings,
                            units=config.hidden_size,
                            activation=tf.nn.relu,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=config.lambda_l2))

            self.context_encoding = tf.layers.dense(inputs=h1,
                                                units=config.embedding_size,
                                                activation=None,
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=config.lambda_l2))

            if not is_test:

                print(self.context_encoding)
                embed = tf.reshape(self.context_encoding, [-1, config.embedding_size])
                labels = tf.reshape(labels, [-1, 1])

                # Construct the variables for the NCE loss
                nce_weights = tf.Variable(tf.truncated_normal([config.vocabulary_size, config.embedding_size], stddev=1.0 / math.sqrt(config.embedding_size)))
                nce_biases = tf.Variable(tf.zeros([config.vocabulary_size]))

                loss = tf.reduce_mean(
                    tf.nn.nce_loss(weights=nce_weights,
                                   biases=nce_biases,
                                   labels=labels,
                                   inputs=embed,
                                   num_sampled=config.num_sampled,
                                   num_classes=config.vocabulary_size))


                self.loss = loss

                with tf.name_scope("AdamGradientDescentOptimization"):
                    optimizer = tf.train.AdamOptimizer(config.learning_rate)
                    self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


class Config():
    def __init__(self, is_test=False):
        self.is_test = is_test

        self._PAD = 0
        self._GO = 1
        self._EOW = 2
        self._UNK = 3
        self.char_vocab_size = len(char_dict)
        self.char_embed_size = 50
        self.morph_rnn_size = 500
        self.sentence_rnn_size = 600
        self.learning_rate = 0.001
        self.word_max_len = 15
        self.sentence_max_len = 64
        self.batch_size = 80
        self.epochs = 50
        self.steps = 1000
        self.character_decoder_size = 1200
        self.vocabulary_size = None
        self.embedding_size = 600
        self.num_sampled = 10

        self.morph_encoder_keep_prob = 1 if not is_test else 1
        self.w2f_keep_prob = 1 if not is_test else 1
        self.character_decoder_keep_prob = 1 if not is_test else 1

        self.lambda_l2 = 5e-4


def to_chars(words, word_max_size):
    char_words = np.ndarray(shape=[len(words), word_max_size], dtype=np.int32)
    for i in range(len(words)):
        if words[i]=="<PAD>":
            char_words[i][:] = _PAD
            continue
        char_words[i][0]=_GO
        for j in range(1,word_max_size):
            if j < len(words[i])+1:
                if words[i][j-1] in char_dict:
                    char_words[i][j] = char_dict[words[i][j-1]]
                else:
                    char_words[i][j] = _UNK
            elif j == len(words[i])+1:
                char_words[i][j] = _EOW
            else:
                char_words[i][j] = _PAD
        if char_words[i][word_max_size-1] != _PAD:
            char_words[i][word_max_size-1] =_EOW
    return char_words


def generator(data_path, word_max_len, sentence_max_len=None, batch_size=None, vocabulary=None):

    with open(data_path) as f:
        sentences = []
        labels = []


        for line in f:
            line = line.replace("\n", "")
            splits = line.split(" ")
            label =[]
            for w in splits:
                w = w.lower()
                if w in vocabulary:
                    label.append(vocabulary[w])
                else:
                    label.append(vocabulary["<UNK>"])
            npsentence = to_chars(splits, word_max_len)

            npsentence = npsentence[:sentence_max_len]
            label = label[:sentence_max_len]
            remaining = sentence_max_len - len(npsentence)
            if remaining > 0:
                npsentence = np.concatenate((npsentence, np.zeros([remaining, word_max_len], dtype=np.int32)), axis=0)
                label = label + [0 for _ in range(remaining)]
            sentences.append(npsentence)
            labels.append(label)
            if len(sentences)==batch_size:
                yield np.stack(sentences, axis=0), np.stack(labels, axis=0)
                sentences=[]
                labels=[]


def load_vocabulary_map(vocabulary_path):

    dict = {"<PAD>": 0, "<UNK>": 1}
    with open(vocabulary_path) as file:
        for line in file:
            word = line.split()[0]
            dict[word] = len(dict)

    inverse = {v:k for k,v in dict.items()}

    return dict, inverse


def train(log_root, vocab_path, data_wildcard):

    vocabulary, inverse = load_vocabulary_map(vocab_path)

    config = Config()
    config.vocabulary_size = len(vocabulary)
    words_ph = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, config.sentence_max_len, config.word_max_len])
    labels_pl = tf.placeholder(dtype=tf.int32, shape=[config.batch_size, config.sentence_max_len])
    agent = Model(elements_chars = words_ph, labels=labels_pl, config=config)


    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=log_root,
            save_checkpoint_secs=600,
            save_summaries_steps=0) as sess:

        files = glob(data_wildcard)
        for l in range(config.epochs):
            for i, data_path in enumerate(files):
                sentences = generator(data_path=data_path,
                                word_max_len=config.word_max_len,
                                sentence_max_len=config.sentence_max_len,
                                batch_size=config.batch_size,
                                vocabulary=vocabulary)
                for j, (batch,labels) in enumerate(sentences):
                    feed = {words_ph: batch, labels_pl: labels}
                    if j%100==0:
                        _, loss = sess.run((agent.train_op, agent.loss), feed)
                        print("Epoch %d File %d \t Batch %d \t Loss %f" % (l, i, j, loss))

                    else:
                        sess.run(agent.train_op, feed)


def create_vocabulary(vocabulary_path, data_wildcard, vocabulary_size):

    counter = Counter()
    f1 = open(vocabulary_path,  "w")
    files = glob(data_wildcard)
    for i, data_path in enumerate(files):
        with open(data_path) as f:
            for i, line in enumerate(f):
                sentences = nltk.sent_tokenize(line)
                for sentence in sentences:
                    tokens = nltk.word_tokenize(sentence)
                    if len(tokens) > 64: continue
                    counter.update(Counter(tokens))
    if vocabulary_size is None:
        vocabulary_size = len(counter)
    for w,c in counter.most_common(vocabulary_size):
        f1.write(w)
        f1.write("\n")
    f1.close()


class SentenceEncoder(object):

    def __init__(self, log_root):

        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.config = Config(is_test=True)
        self.config.batch_size = 1
        self.words_ph = tf.placeholder(dtype=tf.int32,
                                  shape=[self.config.batch_size, self.config.sentence_max_len, self.config.word_max_len])
        self.agent = Model(elements_chars=self.words_ph, config=self.config, is_test=True)  #is_test=True not to use dropout
        restorer = tf.train.Saver()
        self.sess = tf.Session()
        restorer.restore(self.sess, tf.train.latest_checkpoint(log_root))

    def encode_sentence(self, sentence, target_pos):

        npsentence = to_chars(sentence, self.config.word_max_len)
        remaining = self.config.sentence_max_len - len(npsentence)
        if remaining > 0:
            npsentence = np.concatenate((npsentence, np.zeros([remaining, self.config.word_max_len], dtype=np.int32)), axis=0)

        npsentence = np.reshape(npsentence, [1, self.config.sentence_max_len, self.config.word_max_len])
        return self.sess.run(self.agent.context_encoding, {self.words_ph: npsentence})[0, target_pos, :]

    def encode_word(self, word):

        sentence = [word]
        npsentence = to_chars(sentence, self.config.word_max_len)
        remaining = self.config.sentence_max_len - len(npsentence)
        if remaining > 0:
            npsentence = np.concatenate((npsentence, np.zeros([remaining, self.config.word_max_len], dtype=np.int32)), axis=0)

        npsentence = np.reshape(npsentence, [1, self.config.sentence_max_len, self.config.word_max_len])
        return self.sess.run(self.agent.words_encodings, {self.words_ph: npsentence})[0, 0, :]


def test(log_root):

    wsdobj = SentenceEncoder(log_root)
    print(wsdobj.encode_sentence(sentence=["the", "cat","is", "on", "the", "table"],
                        target_pos=3))
    print(wsdobj.encode_word(word="cat"))


if __name__ =='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("-l", "--logdir", dest="logdir", default="log",
                        help="Log directory")
    parser.add_argument("-d", "--datadir", dest="datadir", default="data",
                        help="Data directory.")
    parser.add_argument("-t", "--test", dest='test', type=bool, default=False,
                        help="Test on a dummy sentence")
    parser.add_argument("-t", "--test", dest='test', type=bool, default=False,
                        help="Test on a dummy sentence")

    args = parser.parse_args()

    folder = args.datadir
    data_wildcard = os.path.join(folder,"data*")
    vocabulary_path = os.path.join(folder, "vocabulary.txt")
    if not os.path.isfile(vocabulary_path):
        create_vocabulary(vocabulary_path,data_wildcard,None)
    log_root= args.logdir
    if not args.test:
        train(log_root, vocabulary_path, data_wildcard)
    else:
        test(log_root)

