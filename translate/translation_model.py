import tensorflow as tf
import os
import pickle
import time
import sys
import math
import shutil
from translate import utils
from translate.seq2seq_model import Seq2SeqModel
import pdb


class BaseTranslationModel(object):
    def __init__(self, name, checkpoint_dir, keep_best=1):
        self.name = name
        self.keep_best = keep_best
        self.checkpoint_dir = checkpoint_dir
        self.saver = tf.train.Saver(max_to_keep=3, keep_checkpoint_every_n_hours=5)

    def manage_best_checkpoints(self, step, score):
        score_filename = os.path.join(self.checkpoint_dir, 'scores.txt')
        # try loading previous scores
        try:
            with open(score_filename) as f:
                # list of pairs (score, step)
                scores = [(float(line.split()[0]), int(line.split()[1])) for line in f]
        except IOError:
            scores = []

        if any(step_ >= step for _, step_ in scores):
            utils.warn('inconsistent scores.txt file')

        best_scores = sorted(scores, reverse=True)[:self.keep_best]

        if any(score_ < score for score_, _ in best_scores) or not best_scores:
            shutil.copy(os.path.join(self.checkpoint_dir, 'translate-{}'.format(step)),
                        os.path.join(self.checkpoint_dir, 'best-{}'.format(step)))

            if all(score_ < score for score_, _ in best_scores):
                path = os.path.abspath(os.path.join(self.checkpoint_dir, 'best'))
                try:  # remove old links
                    os.remove(path)
                except OSError:
                    pass
                # make symbolic links to best model
                os.symlink('{}-{}'.format(path, step), path)

            best_scores = sorted(best_scores + [(score, step)], reverse=True)

            for _, step_ in best_scores[self.keep_best:]:
                # remove checkpoints that are not in the top anymore
                try:
                    os.remove(os.path.join(self.checkpoint_dir, 'best-{}'.format(step_)))
                except OSError:
                    pass

        # save bleu scores
        scores.append((score, step))

        with open(score_filename, 'w') as f:
            for score_, step_ in scores:
                f.write('{} {}\n'.format(score_, step_))

    def initialize(self, sess, checkpoints=None, reset=False, reset_learning_rate=False):
        sess.run(tf.initialize_all_variables())
        if checkpoints:  # load partial checkpoints
            for checkpoint in checkpoints:  # checkpoint files to load
                load_checkpoint(sess, None, checkpoint,
                                blacklist=('learning_rate', 'global_step', 'dropout_keep_prob'))
        elif not reset:
            blacklist = ('learning_rate', 'dropout_keep_prob') if reset_learning_rate else ()
            load_checkpoint(sess, self.checkpoint_dir, blacklist=blacklist)

    def save(self, sess):
        save_checkpoint(sess, self.saver, self.checkpoint_dir, self.global_step)


class TranslationModel(BaseTranslationModel):
    def __init__(self, name, encoders, decoder, checkpoint_dir, learning_rate,
                 learning_rate_decay_factor, batch_size, keep_best=1,
                 load_embeddings=None, optimizer='sgd', **kwargs):
        self.batch_size = batch_size
        self.src_ext = [encoder.get('ext') or encoder.name for encoder in encoders]
        self.trg_ext = decoder.get('ext') or decoder.name
        self.extensions = self.src_ext + [self.trg_ext]

        encoders_and_decoder = encoders + [decoder]
        self.binary_input = [encoder_or_decoder.binary for encoder_or_decoder in encoders_and_decoder]
        self.character_level = [encoder_or_decoder.character_level for encoder_or_decoder in encoders_and_decoder]

        self.learning_rate = tf.Variable(learning_rate, trainable=False, name='learning_rate', dtype=tf.float32)

        if optimizer == 'sgd':
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        else:
            self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate)

        with tf.device('/cpu:0'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.filenames = utils.get_filenames(extensions=self.extensions, **kwargs)
        # TODO: check that filenames exist
        utils.debug('reading vocabularies')
        self._read_vocab()

        for encoder_or_decoder, vocab in zip(encoders + [decoder], self.vocabs):
            #pdb.set_trace()
            if vocab is None:
                encoder_or_decoder.vocab_size = 0
                continue
            if encoder_or_decoder.vocab_size <= 0:
                encoder_or_decoder.vocab_size = len(vocab.reverse)

        # this adds an `embedding' attribute to each encoder and decoder
        utils.read_embeddings(self.filenames.embeddings, encoders + [decoder], load_embeddings, self.vocabs)

        # main model
        utils.debug('creating model {}'.format(name))
        self.model = Seq2SeqModel(encoders, decoder, self.learning_rate, self.global_step, optimizer=optimizer,
                                  **kwargs)

        super(TranslationModel, self).__init__(name, checkpoint_dir, keep_best)

        self.batch_iterator = None
        self.dev_batches = None

    def read_data(self, max_train_size, max_dev_size):
        utils.debug('reading training data')
        train_set = utils.read_dataset_list(self.filenames.train, self.extensions, self.vocabs, max_size=max_train_size,
                                       binary_input=self.binary_input, character_level=self.character_level)
        #utils.debug('train_set {}'.format(train_set))
        self.batch_iterator = utils.read_ahead_batch_iterator_list(train_set, self.batch_size, read_ahead=10)

        utils.debug('reading development data')
        dev_sets = [
            utils.read_dataset_list(dev, self.extensions, self.vocabs, max_size=max_dev_size,
                               binary_input=self.binary_input, character_level=self.character_level)
            for dev in self.filenames.dev
        ]
        # subset of the dev set whose perplexity is periodically evaluated
        self.dev_batches = [utils.get_batches_list(dev_set, batch_size=self.batch_size, batches=-1) for dev_set in dev_sets]

    def _read_vocab(self):
        # don't try reading vocabulary for encoders that take pre-computed features
        self.vocabs = [
            utils.initialize_vocabulary(vocab_path) if not binary else None
            for ext, vocab_path, binary in zip(self.extensions, self.filenames.vocab, self.binary_input)
        ]
        self.src_vocab = self.vocabs[:-1]
        self.trg_vocab = self.vocabs[-1]
        self.ngrams = self.filenames.lm_path and utils.read_ngrams(self.filenames.lm_path, self.trg_vocab.vocab)

    def train(self, *args, **kwargs):
        raise NotImplementedError('use MultiTaskModel')

    def train_step(self, sess):
        return self.model.step(sess, next(self.batch_iterator)).loss

    def eval_step(self, sess):
        # compute perplexity on dev set
        for dev_batches in self.dev_batches:
            eval_loss = sum(
                self.model.step(sess, batch, forward_only=True).loss * len(batch)
                for batch in dev_batches
            )
            eval_loss /= sum(map(len, dev_batches))

            perplexity = math.exp(eval_loss) if eval_loss < 300 else float('inf')
            utils.log("  eval: perplexity {:.2f}".format(perplexity))

    def _decode_sentence(self, sess, src_sentences, beam_size=1, remove_unk=False):
        # TODO: merge this with read_dataset
        token_ids = [
            utils.sentence_to_token_ids(sentence, vocab.vocab, character_level=char_level)
            if vocab is not None else sentence  # when `sentence` is not a sentence but a vector...
            for vocab, sentence, char_level in zip(self.vocabs, src_sentences, self.character_level)
        ]

        if beam_size <= 1 and not isinstance(sess, list):
            trg_token_ids, _ = self.model.greedy_decoding(sess, token_ids)
        else:
            hypotheses, scores = self.model.beam_search_decoding(sess, token_ids, beam_size, ngrams=self.ngrams)
            trg_token_ids = hypotheses[0]  # first hypothesis is the highest scoring one

        # remove EOS symbols from output
        if utils.EOS_ID in trg_token_ids:
            trg_token_ids = trg_token_ids[:trg_token_ids.index(utils.EOS_ID)]

        trg_tokens = [self.trg_vocab.reverse[i] if i < len(self.trg_vocab.reverse) else utils._UNK
                      for i in trg_token_ids]

        if remove_unk:
            trg_tokens = [token for token in trg_tokens if token != utils._UNK]

        if self.character_level[-1]:
            return ''.join(trg_tokens)
        else:
            return ' '.join(trg_tokens).replace('@@ ', '')  # merge subword units

    def align(self, sess, output=None, wav_files=None, **kwargs):
        if len(self.src_ext) != 1:
            raise NotImplementedError

        if len(self.filenames.test) != len(self.extensions):
            raise Exception('wrong number of input files')

        for line_id, lines in enumerate(utils.read_lines(self.filenames.test, self.extensions, self.binary_input)):
            token_ids = [
                utils.sentence_to_token_ids(sentence, vocab.vocab, character_level=char_level)
                if vocab is not None else sentence
                for vocab, sentence, char_level in zip(self.vocabs, lines, self.character_level)
            ]

            _, weights = self.model.step(sess, data=[token_ids], forward_only=True, align=True)
            trg_tokens = [self.trg_vocab.reverse[i] if i < len(self.trg_vocab.reverse) else utils._UNK
                          for i in token_ids[-1]]

            weights = weights.squeeze()[:len(trg_tokens), ::-1].T

            max_len = weights.shape[0]

            if self.binary_input[0]:
                src_tokens = None
            else:
                src_tokens = lines[0].split()[:max_len]

            if wav_files is not None:
                wav_file = wav_files[line_id]
            else:
                wav_file = None

            output_file = '{}.{}.svg'.format(output, line_id + 1) if output is not None else None
            utils.heatmap(src_tokens, trg_tokens, weights.T, wav_file=wav_file, output_file=output_file)

    def decode(self, sess, beam_size, output=None, remove_unk=False, **kwargs):
        utils.log('starting decoding')

        # empty `test` means that we read from standard input, which is not possible with multiple encoders
        assert len(self.src_ext) == 1 or self.filenames.test
        # we can't read binary data from standard input
        assert self.filenames.test or self.src_ext[0] not in self.binary_input
        # check that there is the right number of files for decoding
        assert not self.filenames.test or len(self.filenames.test) == len(self.src_ext)

        output_file = None
        try:
            output_file = sys.stdout if output is None else open(output, 'w')

            for lines in utils.read_lines(self.filenames.test, self.src_ext, self.binary_input):
                trg_sentence = self._decode_sentence(sess, lines, beam_size, remove_unk)
                output_file.write(trg_sentence + '\n')
                output_file.flush()
        finally:
            if output_file is not None:
                output_file.close()

    def evaluate(self, sess, beam_size, score_function, on_dev=True, output=None, remove_unk=False,
                 auxiliary_score_function=None, script_dir='scripts', **kwargs):
        """
        :param score_function: name of the scoring function used to score and rank models
          (typically 'bleu_score')
        :param on_dev: if True, evaluate the dev corpus, otherwise evaluate the test corpus
        :param output: save the hypotheses to this file
        :param remove_unk: remove the UNK symbols from the output
        :param auxiliary_score_function: optional scoring function used to display a more
          detailed summary.
        :param script_dir: parameter of scoring functions
        :return: scores of each corpus to evaluate
        :Chunlei updated for stream mode, 1/9/2017
        """
        utils.log('starting decoding')
        assert on_dev or len(self.filenames.test) == len(self.extensions)

        filenames = self.filenames.dev if on_dev else [self.filenames.test]

        # convert `output` into a list, for zip
        if isinstance(output, str):
            output = [output]
        elif output is None:
            output = [None] * len(filenames)

        scores = []

        for filenames_, output_ in zip(filenames, output):  # evaluation on multiple corpora
            lines = list(utils.read_lines_list(filenames_, self.extensions, binary_input=False))
            
            #utils.log('lines {}'.format(lines))
            big_batches_ = []
            for i in range(len(lines)):
                big_batches_.append([])
                tmp_list = lines[i][0]
                #debug('i: {}'.format(i))
                big_batches_[i].append(utils.read_binary_features_list(tmp_list.strip()))
                big_batches_[i].append(lines[i][1])
            
            hypotheses = []
            references = []

            try:
                output_file = open(output_, 'w') if output_ is not None else None

                for *src_sentences, trg_sentence in big_batches_:
                    hypotheses.append(self._decode_sentence(sess, src_sentences, beam_size, remove_unk))
                    references.append(trg_sentence.strip().replace('@@ ', ''))
                    if output_file is not None:
                        output_file.write(hypotheses[-1] + '\n')
                        output_file.flush()

            finally:
                if output_file is not None:
                    output_file.close()

            # main scoring function (used to choose which checkpoints to keep)
            # default is utils.bleu_score
            score, score_summary = getattr(utils, score_function)(hypotheses, references, script_dir)

            # optionally use an auxiliary function to get different scoring information
            if auxiliary_score_function is not None and auxiliary_score_function != score_function:
                try:
                    _, score_summary = getattr(utils, auxiliary_score_function)(hypotheses, references, script_dir)
                except:
                    pass

            # print the scoring information
            score_info = []
            if self.name is not None:
                score_info.append(self.name)
            score_info.append('score={}'.format(score))
            if score_summary:
                score_info.append(score_summary)

            utils.log(' '.join(map(str, score_info)))
            scores.append(score)

        return scores


def load_checkpoint(sess, checkpoint_dir, filename=None, blacklist=()):
    """ `checkpoint_dir` should be unique to this model
    if `filename` is None, we load last checkpoint, otherwise
      we ignore `checkpoint_dir` and load the given checkpoint file.
    """
    if filename is None:
        # load last checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt is not None:
            filename = ckpt.model_checkpoint_path
    else:
        checkpoint_dir = os.path.dirname(filename)

    var_file = os.path.join(checkpoint_dir, 'vars.pkl')

    if os.path.exists(var_file):
        with open(var_file, 'rb') as f:
            var_names = pickle.load(f)
            variables = [var for var in tf.all_variables() if var.name in var_names]
    else:
        variables = tf.all_variables()

    # remove variables from blacklist
    variables = [var for var in variables if not any(prefix in var.name for prefix in blacklist)]

    if filename is not None:
        utils.log('reading model parameters from {}'.format(filename))
        tf.train.Saver(variables).restore(sess, filename)

        utils.debug('retrieved parameters ({})'.format(len(variables)))
        for var in variables:
            utils.debug('  {} {}'.format(var.name, var.get_shape()))


def save_checkpoint(sess, saver, checkpoint_dir, step=None, name=None):
    """ `checkpoint_dir` should be unique to this model """
    var_file = os.path.join(checkpoint_dir, 'vars.pkl')
    name = name or 'translate'

    if not os.path.exists(checkpoint_dir):
        utils.log("creating directory {}".format(checkpoint_dir))
        os.makedirs(checkpoint_dir)

    with open(var_file, 'wb') as f:
        var_names = [var.name for var in tf.all_variables()]
        pickle.dump(var_names, f)

    utils.log('saving model to {}'.format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, name)
    saver.save(sess, checkpoint_path, step, write_meta_graph=False)
    utils.log('finished saving model')
