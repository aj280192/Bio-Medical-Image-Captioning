import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor


class SATModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(SATModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = DecoderRNN(args, tokenizer)

        # initializing the parameters of decoder.
        for p in self.encoder_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets, mode='train'):
        att_feats, _ = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets, mode='forward')
        elif mode == 'sample':
            output = self.encoder_decoder(att_feats, targets, mode='sample')
        else:
            raise ValueError
        return output


class DecoderRNN(nn.Module):
    """Attributes:
    - embedding_dim - specified size of embeddings;
    - hidden_dim - the size of RNN layer (number of hidden states)
    - vocab_size - size of vocabulary
    - p - dropout probability
    """

    def __init__(self, args, tokenizer):
        super(DecoderRNN, self).__init__()

        self.num_features = args.d_vf
        self.embedding_dim = args.emb_dim
        self.hidden_dim = args.d_ff
        self.vocab_size = len(tokenizer.idx2token) + 1
        # scale the inputs to softmax
        self.sample_temp = 0.5

        # embedding layer that turns words into a vector of a specified size
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        # LSTM will have a single layer of size 512 (512 hidden units)
        # it will input concatinated context vector (produced by attention)
        # and corresponding hidden state of Decoder
        self.lstm = nn.LSTMCell(self.embedding_dim + self.num_features, self.hidden_dim)
        # produce the final output
        self.fc = nn.Linear(self.hidden_dim, self.vocab_size)

        # add attention layer
        self.attention = BahdanauAttention(self.num_features, self.hidden_dim)
        # dropout layer
        self.drop = nn.Dropout(p=args.dropout)

        # add initialization fully-connected layers
        # initialize hidden state and cell memory using average feature vector
        # Source: https://arxiv.org/pdf/1502.03044.pdf
        self.init_h = nn.Linear(self.num_features, self.hidden_dim)
        self.init_c = nn.Linear(self.num_features, self.hidden_dim)

    def init_hidden(self, features):

        """Initializes hidden state and cell memory using average feature vector.
        Arguments:
        ----------
        - features - features returned from Encoder

        Retruns:
        ----------
        - h0 - initial hidden state (short-term memory)
        - c0 - initial cell state (long-term memory)
        """
        mean_annotations = torch.mean(features, dim=1)
        h0 = self.init_h(mean_annotations)
        c0 = self.init_c(mean_annotations)
        return h0, c0

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def _forward(self, features, captions, sample_prob=0.0):
        """Arguments
        ----------
        - captions - image captions
        - features - features returned from Encoder
        - sample_prob - use it for scheduled sampling

        Returns
        ----------
        - outputs - output logits from t steps
        """
        # create embeddings for captions of size (batch, sqe_len, embed_dim)
        embed = self.embeddings(captions)
        h, c = self.init_hidden(features)
        seq_len = captions.size(1)
        batch_size = features.size(0)

        # these tensors will store the outputs from lstm cell and attention weights
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(features.device)

        # scheduled sampling for training
        # we do not use it at the first timestep (<start> word)
        # but later we check if the probability is bigger than random
        for t in range(seq_len):
            sample_prob = 0.0 if t == 0 else 0.5
            use_sampling = np.random.random() < sample_prob
            if use_sampling == False:
                word_embed = embed[:, t, :]
            context = self.attention(features, h)
            # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
            input_concat = torch.cat([word_embed, context], 1)
            h, c = self.lstm(input_concat, (h, c))
            h = self.drop(h)
            output = self.fc(h)
            if use_sampling == True:
                # use sampling temperature to amplify the values before applying softmax
                scaled_output = output / self.sample_temp
                scoring = F.log_softmax(scaled_output, dim=1)
                top_idx = scoring.topk(1)[1]
                word_embed = self.embeddings(top_idx).squeeze(1)
            outputs[:, t, :] = output

        return outputs

    def _sample(self, features, captions):

        """Greedy search to sample top candidate from distribution.
        Arguments
        ----------
        - features - features returned from Encoder
        Returns:
        ----------
        - sentence - list of tokens
        """

        seq_len = captions.size(1)
        batch_size = features.size(0)

        outputs = torch.zeros(batch_size, seq_len).to(features.device)

        input_word = captions[:, 0]
        h, c = self.init_hidden(features)

        for t in range(seq_len):  # for the length of the batch example
            emb_word = self.embeddings(input_word)
            context = self.attention(features, h)
            input_concat = torch.cat([emb_word, context], dim=1)
            h, c = self.lstm(input_concat, (h, c))
            h = self.drop(h)
            output = self.fc(h)
            scoring = F.log_softmax(output, dim=1)
            top_idx = scoring.topk(1)[1].squeeze(1)
            input_word = top_idx
            outputs[:, t] = top_idx

        return outputs


class BahdanauAttention(nn.Module):
    """ Class performs Additive Bahdanau Attention.
    Source: https://arxiv.org/pdf/1409.0473.pdf

    """

    def __init__(self, num_features, hidden_dim, output_dim=1):
        super(BahdanauAttention, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # fully-connected layer to learn first weight matrix Wa
        self.W_a = nn.Linear(self.num_features, self.hidden_dim)
        # fully-connected layer to learn the second weight matrix Ua
        self.U_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        # fully-connected layer to produce score (output), learning weight matrix va
        self.v_a = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, features, decoder_hidden):
        """
        Arguments:
        ----------
        - features - features returned from Encoder
        - decoder_hidden - hidden state output from Decoder

        Returns:
        ---------
        - context - context vector with a size of (1,2048)
        - atten_weight - probabilities, express the feature relevance
        """
        # add additional dimension to a hidden (need for summation later)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        atten_1 = self.W_a(features)
        atten_2 = self.U_a(decoder_hidden)
        # apply tangent to combined result from 2 fc layers
        atten_tan = torch.tanh(atten_1 + atten_2)
        # one score corresponds to one Encoder's output
        atten_score = self.v_a(atten_tan)
        atten_weight = F.softmax(atten_score, dim=1)
        # first, we will multiply each vector by its softmax score
        # next, we will sum up this vectors, producing the attention context vector
        context = torch.sum(atten_weight * features,
                            dim=1)
        return context
