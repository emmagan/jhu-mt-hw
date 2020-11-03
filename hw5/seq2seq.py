#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way.
"""


from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib
#if you are running on the gradx/ugradx/ another cluster,
#you will need the following line
#if you run on a local machine, you can comment it out
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid,
# it can be very easy to confict with other people's jobs.
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("is available: ",torch.cuda.is_available())

# lol I am in the rare situation where my device is available but is too old to use with pycharm

device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"),
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################

class LSTMCell(nn.Module):
    """LSTM to be used for the encoder and decoder"""
    def __init__(self, embedding_size, hidden_size, context_size):
        super(LSTMCell, self).__init__()
        self.forget_gate_input = nn.Linear(in_features=embedding_size, out_features=context_size)
        self.forget_gate_hidden = nn.Linear(in_features=hidden_size, out_features=context_size)

        self.input_gate_input = nn.Linear(in_features=embedding_size, out_features=context_size)
        self.input_gate_hidden = nn.Linear(in_features=hidden_size, out_features=context_size)

        self.input_input = nn.Linear(in_features=embedding_size, out_features=context_size)
        self.input_hidden = nn.Linear(in_features=hidden_size, out_features=context_size)

        self.output_gate_input = nn.Linear(in_features=embedding_size, out_features=hidden_size)
        self.output_gate_hidden = nn.Linear(in_features=hidden_size, out_features=hidden_size)

    def forward(self, x, previous_hidden, previous_context):
        x = x.view(1, -1)
        # previous_hidden = previous_hidden.view(1, -1)
        previous_context = previous_context.view(1, -1)

        # print(x.shape)
        # print(previous_hidden.shape)
        # print(previous_context.dtype)
        forget_gate = torch.sigmoid(torch.add(self.forget_gate_input(x), self.forget_gate_hidden(previous_hidden)))
        # print(forget_gate.dtype)
        context_after_forgetting = previous_context * forget_gate
        # print(context_after_forgetting.dtype)

        input_gate = torch.sigmoid(torch.add(self.input_gate_input(x), self.input_gate_hidden(previous_hidden)))
        input_ = torch.tanh(torch.add(self.input_input(x), self.input_hidden(previous_hidden)))
        context_input = torch.mul(input_gate, input_)

        context = torch.add(context_input, context_after_forgetting)

        output_gate = torch.sigmoid(torch.add(self.output_gate_input(x), self.output_gate_hidden(previous_hidden)))

        # print(context.shape)
        # print(output_gate.shape)

        hidden = torch.mul(output_gate, torch.tanh(context))

        return hidden, context


class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, embedding_size, hidden_size, embeddings):
        super(EncoderRNN, self).__init__()
        """ TODO:
        Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM.
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        self.embedding = embeddings
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embedding_size, hidden_size)

    def forward(self, X, hidden, context):
        """runs the forward pass of the encoder
        returns the output and the hidden state

        NOTE: For right now, using a unidirectional LSTM bc I am unsure how to do bidirectional taking 1 word at a time
        """
        embedded = self.embedding(X.to(torch.long)).view(-1, 1, self.input_size)

        output, (hidden, context) = self.lstm(embedded, (hidden, context))
        return output, hidden, context

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device) # TODO: update with trainable parameters


class AttnDecoderRNN(nn.Module):
    """the class for the decoder
    """
    def __init__(self, hidden_size, output_size, embedding_size, embedding, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.embedding = embedding
        self.dropout = nn.Dropout(self.dropout_p)

        #self.softmax = nn.Softmax()
        #self.logSoftmax = nn.LogSoftmax()

        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.attn = nn.Linear(hidden_size + hidden_size, 1) # we can add more layers to the attention
        self.attn_combine = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, _input, hidden, context, encoder_outputs):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights

        Dropout (self.dropout) should be applied to the word embeddings.
        """
        embedded = self.embedding(_input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        encoder_outputs = encoder_outputs.reshape(-1, 15, 256)

        broadcast = torch.ones_like(encoder_outputs) * hidden.reshape(-1)


        attn_weights = F.softmax(
            self.attn(torch.cat((encoder_outputs, broadcast), 2)), dim=1)

        print(attn_weights.shape)
        print(encoder_outputs.shape)

        attn_applied = torch.bmm(attn_weights.reshape(-1, 1, 15), encoder_outputs)

        print(attn_applied.shape)

        attn_output = torch.cat((embedded, attn_applied), 2)
        attn_output = self.attn_combine(attn_output)

        output = F.relu(attn_output)

        # print(output.shape)

        output, (hidden, context) = self.lstm(output, (hidden, context))

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, context, attn_weights


    def get_initial_hidden_state(self):
        # initial hidden states are tanh(W_sh_1) in paper, for now just use zeros
        return torch.zeros(1, 1, self.hidden_size, device=device)
'''
        _in = _input.view(-1)
        context = context.view(-1, self.hidden_size)

        broadcast = torch.cat([context] * encoder_outputs.size()[0])

        # print(broadcast.shape)
        # print(encoder_outputs.shape)

        outputs_with_context = torch.cat((encoder_outputs, broadcast), 1)
        attn = self.attention(outputs_with_context)
        weights = self.softmax(attn).view(encoder_outputs.shape[0], -1)

        encoder_attention_context = torch.mm(encoder_outputs.T, weights).view(-1)

        # print(encoder_attention_context.shape)

        # combine this with the input embedding
        embed = self.embedding(_in).view(-1)

        embed_with_attention_context = torch.cat((embed, encoder_attention_context), 0)

        # print(embed_with_attention_context.shape)
        # print(hidden.shape)
        # print(context.shape)

        (output, new_hidden) = self.lstm.forward(embed_with_attention_context, hidden, context)

        # print(output.shape)
        # print(hidden.shape)

        return (self.logSoftmax(self.out(output)), output, new_hidden, weights)
'''


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, batch_size, max_length=MAX_LENGTH):
    encoder_hidden = encoder.get_initial_hidden_state()
    encoder_context= encoder.get_initial_hidden_state()

    optimizer.zero_grad()

    input_length = MAX_LENGTH
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros((batch_size, max_length, encoder.hidden_size), device=device)

    print(encoder_outputs[:, 0].shape)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_context = encoder(
            input_tensor[:, ei], encoder_hidden, encoder_context)


        encoder_outputs[:, ei] = encoder_output.reshape(batch_size, encoder.hidden_size)

    decoder_input = torch.tensor([[SOS_index]], device=device)

    decoder_hidden = encoder_hidden
    decoder_context = encoder_context

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_context, decoder_attention = decoder(
            decoder_input, decoder_hidden, decoder_context, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    optimizer.step()

    return loss.item() / target_length


######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state()
        encoder_context = encoder.get_initial_hidden_state()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden, encoder_context = encoder(input_tensor[ei], encoder_hidden, encoder_context)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_index]], device=device)  # SOS

        decoder_hidden = encoder_hidden
        decoder_context = encoder_context

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_context, decoder_attention = decoder(
                decoder_input, decoder_hidden, decoder_context, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data.squeeze()
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions, filename):
    """visualize the attention mechanism. And save it to a file.
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """
    # TODO vizualize attention
    figure = plt.figure()
    ax = figure.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='gray')
    figure.colorbar(cax)

    ax.set_xticklabels([''] + input_sentence.split(' ') + [EOS_token], rotation=90)
    ax.set_yticklabels([''] + output_words)

    plt.savefig(filename)


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab, filename):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions, filename)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    ap.add_argument('--embedding_size', default=64, type=int)
    ap.add_argument('--batch_size', default=10, type=int)

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    input_embeddings = nn.Embedding(src_vocab.n_words, embedding_dim=args.embedding_size)
    output_embeddings = nn.Embedding(tgt_vocab.n_words, embedding_dim=args.embedding_size)
    encoder = EncoderRNN(args.embedding_size, args.hidden_size, input_embeddings).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, args.embedding_size, output_embeddings, dropout_p=0.1).to(device)

    torch.autograd.set_detect_anomaly(True)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    print(EOS_index)

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    while iter_num < args.n_iters:
        iter_num += 1
        # TODO implement batching?
        batch_pairs = random.sample(train_pairs, args.batch_size)
        training_pairs = (torch.zeros((args.batch_size, MAX_LENGTH)), torch.zeros((args.batch_size, MAX_LENGTH)))
        for i in range(args.batch_size):
            training_pair = tensors_from_pair(src_vocab, tgt_vocab, batch_pairs[i])
            training_pairs[0][i] += torch.cat((training_pair[0].view(-1), torch.zeros((MAX_LENGTH)).to(torch.long)), 0)[:MAX_LENGTH]
            training_pairs[1][i] += torch.cat((training_pair[1].view(-1), torch.zeros((MAX_LENGTH)).to(torch.long)), 0)[:MAX_LENGTH]

        input_tensor = training_pairs[0]
        target_tensor = training_pairs[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion, args.batch_size)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab, '1.png')
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab, '2.png')
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab, '3.png')
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab, '4.png')


if __name__ == '__main__':
    main()
