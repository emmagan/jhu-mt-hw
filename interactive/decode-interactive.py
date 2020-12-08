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
from collections import namedtuple


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')


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
    # reverse input sentence but not target sentence, based on paper
    sentence = " ".join(pair[0].split()[::-1])
    input_tensor = tensor_from_sentence(src_vocab, sentence)
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor

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
        self.embedding_size = embedding_size
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
        self.attn = nn.Linear(hidden_size + hidden_size, hidden_size) # we can add more layers to the attention
        self.attn2 = nn.Linear(hidden_size, 1)
        self.attn_combine = nn.Linear(embedding_size + hidden_size, hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, _input, hidden, context, encoder_outputs):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights

        Dropout (self.dropout) should be applied to the word embeddings.
        """

        embedded = self.embedding(_input.to(torch.long)).view(-1, 1, self.embedding_size)
        embedded = self.dropout(embedded)

        encoder_outputs = encoder_outputs.reshape(-1, 15, 256)

        broadcast = torch.ones_like(encoder_outputs) * hidden.reshape(-1)


        attn_weights = F.softmax(
            self.attn2(F.relu(self.attn(torch.cat((encoder_outputs, broadcast), 2)))), dim=1)


        attn_applied = torch.bmm(attn_weights.reshape(-1, 1, 15), encoder_outputs)


        attn_output = torch.cat((embedded, attn_applied), 2)
        attn_output = self.attn_combine(attn_output)

        output = F.relu(attn_output)

        output, (hidden, context) = self.lstm(output, (hidden, context))

        output = F.log_softmax(self.out(output.reshape(-1, 256)), dim=1)


        return output, hidden, context, attn_weights


    def get_initial_hidden_state(self):
        # initial hidden states are tanh(W_sh_1) in paper, for now just use zeros
        return torch.zeros(1, 1, self.hidden_size, device=device)


def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, optimizer, criterion, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab, optimizer, criterion)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, optimizer, criterion, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    MAX_HYPS = 10

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()


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

    hypothesis = namedtuple("hypothesis", "logprob, hidden, context, word, words")
    initial_hypothesis = hypothesis(0.0, decoder_hidden, decoder_context, torch.tensor(SOS_index), [SOS_token])

    translations = [[] for _ in range(MAX_HYPS)]
    map(lambda x: x.append(initial_hypothesis), translations)

    stacks = [{} for _ in range(max_length)] + [{}]
    stacks[0][SOS_token] = initial_hypothesis

    optimizer.zero_grad()

    length = 0

    for i, stack in enumerate(stacks[:-1]):
        length = i
        for h in sorted(stack.values(), key=lambda h: -h.logprob)[:MAX_HYPS]:
            decoder_output, decoder_hidden, decoder_context, decoder_attention = decoder(h.word, h.hidden,
                                                                                         h.context, encoder_outputs)
            decoder_attentions[i] = decoder_attention.data.squeeze()
            topv, topi = decoder_output.data.topk(
                MAX_HYPS)  # since we are selecting MAX_HYPS at most, we only need to pick the 10 best from each possible translation
            # print(topi.shape)
            for ind in range(topi.shape[1]):
                print(f"{ind}. {tgt_vocab.index2word[topi.flatten()[ind].item()]}: {topv.flatten()[ind]}")

            print(f"{MAX_HYPS}: enter another word...")


            print(f"current sentence: {h.words}")

            best = int(input("enter the index of the best translation: "))

            loss = 0

            if best == MAX_HYPS:

                word = input("enter the desired word: ")

                while word not in tgt_vocab.word2index:
                    word = input(f"\"{word}\" is not in the vocab. Please enter another word: ")


                word_index = tgt_vocab.word2index[word]

                loss += criterion(decoder_output.reshape(1, -1), torch.tensor(best).reshape(-1))

                new_hypothesis = hypothesis(h.logprob + decoder_output.data.flatten()[word_index], decoder_hidden, decoder_context,
                                            torch.tensor(word_index),
                                            h.words + [word])
                if new_hypothesis.word not in stack or stack[
                    new_hypothesis.word].logprob < new_hypothesis.logprob:  # second case is recombination
                    stacks[i + 1][word] = new_hypothesis

                if word_index == EOS_index:
                    # decoded_words.append('<EOS>')
                    winner = max(stack.values(), key=lambda h: h.logprob)

                    loss.backward()
                    optimizer.step()

                    return winner.words, decoder_attentions[:i + 1]
            else:
                new_hypothesis = hypothesis(h.logprob + topv.flatten()[best], decoder_hidden, decoder_context,
                                            topi.flatten()[best],
                                            h.words + [tgt_vocab.index2word[topi.flatten()[best].item()]])

                loss += criterion(decoder_output.reshape(1, -1), torch.tensor(best).reshape(-1))


                if new_hypothesis.word.item() not in stack or stack[new_hypothesis.word.item()].logprob < new_hypothesis.logprob:  # second case is recombination
                    stacks[i + 1][tgt_vocab.index2word[topi.flatten()[best].item()]] = new_hypothesis

                if topi.flatten()[best].item() == EOS_index:
                    # decoded_words.append('<EOS>')
                    winner = max(stack.values(), key=lambda h: h.logprob)

                    loss.backward()
                    optimizer.step()

                    return winner.words, decoder_attentions[:i + 1]

    winner = max(stacks[-1].values(), key=lambda h: h.logprob)

    loss.backward()
    optimizer.step()

    return winner.words, decoder_attentions[:length + 1]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=['state.pt'], nargs=1)
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--embedding_size', default=64, type=int)
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--initial_learning_rate', default=0.01, type=int,
                    help='initial learning rate')
    ap.add_argument('--num-sents', default=1)
    args = ap.parse_args()

    # process the training, dev, test files

    print(args.checkpoint)

    # Create vocab from training data, or load if checkpointed
    # also set iteration
    state = torch.load(args.checkpoint[0])
    iter_num = state['iter_num']
    src_vocab = state['src_vocab']
    tgt_vocab = state['tgt_vocab']




    input_embeddings = nn.Embedding(src_vocab.n_words, embedding_dim=args.embedding_size)
    output_embeddings = nn.Embedding(tgt_vocab.n_words, embedding_dim=args.embedding_size)
    encoder = EncoderRNN(args.embedding_size, args.hidden_size, input_embeddings).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, args.embedding_size, output_embeddings, dropout_p=0.1).to(device)

    torch.autograd.set_detect_anomaly(True)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    encoder.load_state_dict(state['enc_state'])
    decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    dev_pairs = split_lines(args.dev_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters()) + list(input_embeddings.parameters()) + list(output_embeddings.parameters()) # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    # optimizer.load_state_dict(state['opt_state'])

    print(EOS_index)

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    # translate from the dev set
    translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, optimizer, criterion, n=args.num_sents)

    state = {'iter_num': iter_num,
             'enc_state': encoder.state_dict(),
             'dec_state': decoder.state_dict(),
             'opt_state': optimizer.state_dict(),
             'src_vocab': src_vocab,
             'tgt_vocab': tgt_vocab,
             }

    torch.save(state, 'state_FINAL.pt')

if __name__ == "__main__":
    main()