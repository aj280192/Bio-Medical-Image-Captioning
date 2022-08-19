import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modules.visual_extractor import VisualExtractor, ImageEncoder


class SATModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(SATModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        self.visual_extractor = ImageEncoder(args)
        self.encoder_decoder = DecoderWithAttention(args, tokenizer)

        # initializing the parameters of decoder.
        for p in self.encoder_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, images, targets = None, targets_len = None, mode='train'):
        # att_feats, _ = self.visual_extractor(images)

        att_feats = self.visual_extractor(images)

        if mode == 'train':
            output = self.encoder_decoder(att_feats, targets, targets_len, mode='forward')
        elif mode == 'sample':
            output = self.encoder_decoder(att_feats, targets_len= targets_len, mode='sample')
        else:
            raise ValueError
        return output


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(BahdanauAttention, self).__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoder's output
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_attn = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation (uses Multiplicative attention).
        :param encoder_out: encoded images, a tensor of dim (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dim (batch_size, decoder_dim)
        """
        attn1 = self.encoder_attn(encoder_out)  # (batch_size, num_pixels, attention_dim)
        attn2 = self.decoder_attn(decoder_hidden)  # (batch_size, attention_dim)
        attn = self.full_attn(F.relu(attn1 + attn2.unsqueeze(1)))  # (batch_size, num_pixels, 1)

        # apply softmax to calculate weights for weighted encoding based on attention
        alpha = F.softmax(attn, dim=1)  # (batch_size, num_pixels, 1)
        attn_weighted_encoding = (encoder_out * alpha).sum(dim=1)  # (batch_size, encoder_dim)
        alpha = alpha.squeeze(2)  # (batch_size, num_pixels)
        return attn_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, args, tokenizer):
        """
        :param args: contains all the required parameters
        :param tokenizer: the model tokenizer
        """
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = args.d_vf
        self.attention_dim = args.d_model
        self.embed_dim = args.emb_dim
        self.decoder_dim = args.d_ff
        self.vocab_size = len(tokenizer.idx2token)
        self.dropout = args.dropout

        # beam search parameter
        self.beam_size = args.beam_size

        self.attention = BahdanauAttention(self.encoder_dim, self.decoder_dim, self.attention_dim)  # attention network
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=self.dropout)

        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate

        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary

        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initialize some layers with the uniform distribution for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded_images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return hidden_state, cell_state
        """
        mean_encoder_out = encoder_out.mean(dim=1)  # (batch_size, encoder_dim)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)  # (batch_size, decoder_dim)
        return h, c

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def _forward(self, encoder_out, encoded_captions, caption_lens):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        batch_size = encoder_out.size(0)

        # flatten image
        encoder_out = encoder_out.view(batch_size, -1, self.encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # sort the input data by the decreasing caption length
        caption_lens, sort_idx = caption_lens.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_idx]
        encoded_captions = encoded_captions[sort_idx]

        # embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # initialize lstm state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are caption lengths - 1
        decode_lens = (caption_lens - 1).tolist()

        # create tensors to hold word prediction scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lens), self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lens), num_pixels).to(encoder_out.device)

        # At each time-step, decode by attention-weighing the encoder's output based on the
        # decoder's previous hidden state output then generate a new word in the decoder with
        # the previous word and the attention weighted encoding
        for t in range(max(decode_lens)):
            # get the batch size for each time step t
            batch_size_t = sum([l > t for l in decode_lens])

            # get the attention weighted encodings (batch_size_t, encoder_dim)
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = F.sigmoid(self.f_beta(h[:batch_size_t]))  # sigmoid gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # get the decoder hidden state and cell state based on the embeddings of timestep t word
            # and the attention weighted encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )  # (batch_size_t, decoder_dim)

            # get the next word prediction
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)

            # save the prediction and alpha for every time step
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lens, alphas

    def _sample(self, features, targets_len):

       if self.beam_size == 0:
           return self._greedy_search(features, targets_len)
       else:
           return self._beam_search(features, targets_len)

    def _greedy_search(self, features, targets_len):
        """
        Forward propagation.
        :param features: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param targets_len: caption lengths, a tensor of dimension (batch_size, 1)
        :return: predictions
        """
        batch_size = features.size(0)

        # flatten image
        encoder_out = features.view(batch_size, -1, self.encoder_dim)  # (batch_size, num_pixels, encoder_dim)

        # initial input tokens
        input_words = torch.tensor([[2]] * batch_size, dtype=torch.long).to(features.device) # batch_size , 1

        # initialize lstm state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # create tensors to hold word predictions
        predictions = torch.zeros(batch_size, max(targets_len)).to(encoder_out.device)

        # At each time-step, decode by attention-weighing the encoder's output based on the
        # decoder's previous hidden state output then generate a new word in the decoder with
        # the previous word and the attention weighted encoding
        for t in range(max(targets_len)):
            # embedding
            embeddings = self.embedding(input_words).squeeze(1)  # (batch_size, embed_dim)

            # get the attention weighted encodings (batch_size, encoder_dim)
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            print(attention_weighted_encoding.shape)

            gate = F.sigmoid(self.f_beta(h))  # sigmoid gating scalar, (batch_size, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            # get the decoder hidden state and cell state based on the embeddings of timestep t word
            # and the attention weighted encoding
            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1),
                (h, c)
            )  # (batch_size_t, decoder_dim)

            # get the next word prediction
            preds = self.fc(self.dropout(h))  # (batch_size, vocab_size)
            scoring = F.log_softmax(preds, dim=1)
            top_idx = scoring.topk(1)[1].squeeze(1)
            input_word = top_idx

            # save the prediction and alpha for every time step
            predictions[:, t] = top_idx


        return predictions.cpu().numpy()

    def _beam_search(self, features, targets_len):

        output = []
        # targets_len = targets_len.tolist()
        for feature, max_len in zip(features, targets_len.squeeze(1)):
            output.append(self._generate_captions(feature.unsqueeze(0), max_len))

        return output

    def _generate_captions(self, feature, max_len):
        """
            Reads an image and captions it with beam search as well as plot attention maps.
        """
        k = self.beam_size

        # encode the image
        encoder_dim = feature.size(3)

        # flatten encoded image representation
        encoder_out = feature.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # we'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

        # tensor to store top k previous words at each step; now they're just <start>
        top_k_prev_words = torch.tensor([[2]] * k, dtype=torch.long).to(feature.device)  # (k, 1)

        # tensor to store top k sequences; now they're just <start>
        top_k_seqs = top_k_prev_words  # (k, 1)

        # tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(feature.device)  # (k, 1)

        # lists to store completed sequences along with their scores
        complete_seqs = []
        complete_seqs_scores = []

        # start decoding
        step = 1
        h, c = self.init_hidden_state(encoder_out)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            embeddings = self.embedding(top_k_prev_words).squeeze(1)  # (s, embed_dim)

            attention_weighted_encoding, _ = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels, 1)

            gate = F.sigmoid(self.f_beta(h))  # gating scalar, (s, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding

            h, c = self.decode_step(
                torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c)
            )  # (s, decoder_dim)

            scores = self.fc(h)  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=1)

            # add the scores to prev scores
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # all the k points will have the same score for the first step (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // self.vocab_size  # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)

            # add new words to sequences
            top_k_seqs = torch.cat([top_k_seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step + 1)

            # which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds)
                               if next_word != 3]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(top_k_seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # proceed with incomplete sequences
            if k == 0:
                break

            top_k_seqs = top_k_seqs[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            top_k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # # break if things have been going on too long
            if step == max_len:
                if not (len(complete_seqs) > 0):
                    return top_k_seqs[top_k_scores.argmax()].tolist()
                break
            step += 1

        # select sequence with max score
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        return seq

# class DecoderRNN(nn.Module):
#     """Attributes:
#     - embedding_dim - specified size of embeddings;
#     - hidden_dim - the size of RNN layer (number of hidden states)
#     - vocab_size - size of vocabulary
#     - p - dropout probability
#     """
#
#     def __init__(self, args, tokenizer):
#         super(DecoderRNN, self).__init__()
#
#         self.num_features = args.d_vf
#         self.embedding_dim = args.emb_dim
#         self.hidden_dim = args.d_ff
#         self.vocab_size = len(tokenizer.idx2token)
#         # scale the inputs to softmax
#         self.sample_temp = 0.5
#
#         # beam search parameter
#         self.beam_size = args.beam_size
#
#         # embedding layer that turns words into a vector of a specified size
#         self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
#
#         # LSTM will have a single layer of size 512 (512 hidden units)
#         # it will input concatinated context vector (produced by attention)
#         # and corresponding hidden state of Decoder
#         self.lstm = nn.LSTMCell(self.embedding_dim + self.num_features, self.hidden_dim)
#         # produce the final output
#         self.fc = nn.Linear(self.hidden_dim, self.vocab_size)
#
#         # add attention layer
#         self.attention = BahdanauAttention(self.num_features, self.hidden_dim)
#         # dropout layer
#         self.drop = nn.Dropout(p=args.dropout)
#
#         # add initialization fully-connected layers
#         # initialize hidden state and cell memory using average feature vector
#         # Source: https://arxiv.org/pdf/1502.03044.pdf
#         self.init_h = nn.Linear(self.num_features, self.hidden_dim)
#         self.init_c = nn.Linear(self.num_features, self.hidden_dim)
#
#     def init_hidden(self, features):
#
#         """Initializes hidden state and cell memory using average feature vector.
#         Arguments:
#         ----------
#         - features - features returned from Encoder
#
#         Retruns:
#         ----------
#         - h0 - initial hidden state (short-term memory)
#         - c0 - initial cell state (long-term memory)
#         """
#         mean_annotations = torch.mean(features, dim=1)
#         h0 = self.init_h(mean_annotations)
#         c0 = self.init_c(mean_annotations)
#         return h0, c0
#
#     def forward(self, *args, **kwargs):
#         mode = kwargs.get('mode', 'forward')
#         if 'mode' in kwargs:
#             del kwargs['mode']
#         return getattr(self, '_' + mode)(*args, **kwargs)
#
#     def _forward(self, features, captions, sample_prob=0.0):
#         """Arguments
#         ----------
#         - captions - image captions
#         - features - features returned from Encoder
#         - sample_prob - use it for scheduled sampling
#
#         Returns
#         ----------
#         - outputs - output logits from t steps
#         """
#         # create embeddings for captions of size (batch, sqe_len, embed_dim)
#         embed = self.embeddings(captions)
#         h, c = self.init_hidden(features)
#         seq_len = captions.size(1)
#         batch_size = features.size(0)
#
#         # these tensors will store the outputs from lstm cell and attention weights
#         outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(features.device)
#
#         # scheduled sampling for training
#         # we do not use it at the first timestep (<start> word)
#         # but later we check if the probability is bigger than random
#         for t in range(seq_len):
#             sample_prob = 0.0 if t == 0 else 0.5
#             use_sampling = np.random.random() < sample_prob
#             if use_sampling == False:
#                 word_embed = embed[:, t, :]
#             context = self.attention(features, h)
#             # input_concat shape at time step t = (batch, embedding_dim + hidden_dim)
#             input_concat = torch.cat([word_embed, context], 1)
#             h, c = self.lstm(input_concat, (h, c))
#             h = self.drop(h)
#             output = self.fc(h)
#             if use_sampling == True:
#                 # use sampling temperature to amplify the values before applying softmax
#                 scaled_output = output / self.sample_temp
#                 scoring = F.log_softmax(scaled_output, dim=1)
#                 top_idx = scoring.topk(1)[1]
#                 word_embed = self.embeddings(top_idx).squeeze(1)
#             outputs[:, t, :] = output
#
#         return outputs
#
#     def _sample(self, features, captions):
#
#         """Greedy search to sample top candidate from distribution.
#         Arguments
#         ----------
#         - features - features returned from Encoder
#         Returns:
#         ----------
#         - sentence - list of tokens
#         """
#
#         if self.beam_size != 1:
#             return self._beam_search(features)
#         else:
#             return self._greedy_search(features, captions)
#
#     def _greedy_search(self, features, captions):
#
#         seq_len = captions.size(1)
#         batch_size = features.size(0)
#
#         outputs = torch.zeros(batch_size, seq_len).to(features.device)
#
#         input_word = captions[:, 0]
#         h, c = self.init_hidden(features)
#
#         for t in range(seq_len):  # for the length of the batch example
#             emb_word = self.embeddings(input_word)
#             context = self.attention(features, h)
#             input_concat = torch.cat([emb_word, context], dim=1)
#             h, c = self.lstm(input_concat, (h, c))
#             h = self.drop(h)
#             output = self.fc(h)
#             scoring = F.log_softmax(output, dim=1)
#             top_idx = scoring.topk(1)[1].squeeze(1)
#             input_word = top_idx
#             outputs[:, t] = top_idx
#
#         return outputs.cpu().numpy()
#
#     def _beam_search(self, features):
#         output = []
#         for feature in features:
#             print('calling beam!!')
#             output.append(self.generate_image_caption(feature))
#
#         return output
#
#     def generate_image_caption(self, img):
#         """
#         Reads an image and captions it with beam search as well as plot attention maps.
#         """
#         k = self.beam_size
#
#         # encode the image
#         encoder_dim = img.size(1)
#
#         # flatten encoded image representation
#         encoder_out = img.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
#         num_pixels = encoder_out.size(1)
#
#         # we'll treat the problem as having a batch size of k
#         encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)
#
#         # tensor to store top k previous words at each step; now they're just <start>
#         top_k_prev_words = torch.tensor([[2]] * k, dtype=torch.long).to(img.device)  # (k, 1)
#
#         # tensor to store top k sequences; now they're just <start>
#         top_k_seqs = top_k_prev_words  # (k, 1)
#
#         # tensor to store top k sequences' scores; now they're just 0
#         top_k_scores = torch.zeros(k, 1).to(img.device)  # (k, 1)
#
#         # lists to store completed sequences along with their scores
#         complete_seqs = []
#         complete_seqs_scores = []
#
#         # start decoding
#         step = 1
#         h, c = self.init_hidden(encoder_out)
#         count = 0
#
#         # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
#         while True:
#             embeddings = self.embeddings(top_k_prev_words).squeeze(1)  # (s, embed_dim)
#
#             attention_weighted_encoding = self.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels, 1)
#
#             h, c = self.lstm(
#                 torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c)
#             )  # (s, decoder_dim)
#
#             scores = self.fc(h)  # (s, vocab_size)
#             scores = F.log_softmax(scores, dim=1)
#
#             # add the scores to prev scores
#             scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
#
#             # all the k points will have the same score for the first step (since same k previous words, h, c)
#             if step == 1:
#                 top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
#             else:
#                 # unroll and find top scores, and their unrolled indices
#                 top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
#
#             # convert unrolled indices to actual indices of scores
#             prev_word_inds = top_k_words // self.vocab_size  # (s)
#             next_word_inds = top_k_words % self.vocab_size  # (s)
#
#             # add new words to sequences
#             top_k_seqs = torch.cat([top_k_seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step + 1)
#
#             # which sequences are incomplete (didn't reach <end>)?
#             incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds)
#                                if next_word != 3]
#             complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
#
#             # set aside complete sequences
#             if len(complete_inds) > 0:
#                 complete_seqs.extend(top_k_seqs[complete_inds].tolist())
#                 complete_seqs_scores.extend(top_k_scores[complete_inds])
#             k -= len(complete_inds)  # reduce beam length accordingly
#
#             # proceed with incomplete sequences
#             if k == 0:
#                 break
#
#             top_k_seqs = top_k_seqs[incomplete_inds]
#             h = h[prev_word_inds[incomplete_inds]]
#             c = c[prev_word_inds[incomplete_inds]]
#             encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
#             top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
#             top_k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
#
#             step += 1
#
#         # select sequence with max score
#         i = complete_seqs_scores.index(max(complete_seqs_scores))
#         seq = complete_seqs[i]
#
#         return seq
#
#
# class BahdanauAttentionprev(nn.Module):
#     """ Class performs Additive Bahdanau Attention.
#     Source: https://arxiv.org/pdf/1409.0473.pdf
#
#     """
#
#     def __init__(self, num_features, hidden_dim, output_dim=1):
#         super(BahdanauAttention, self).__init__()
#         self.num_features = num_features
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         # fully-connected layer to learn first weight matrix Wa
#         self.W_a = nn.Linear(self.num_features, self.hidden_dim)
#         # fully-connected layer to learn the second weight matrix Ua
#         self.U_a = nn.Linear(self.hidden_dim, self.hidden_dim)
#         # fully-connected layer to produce score (output), learning weight matrix va
#         self.v_a = nn.Linear(self.hidden_dim, self.output_dim)
#
#     def forward(self, features, decoder_hidden):
#         """
#         Arguments:
#         ----------
#         - features - features returned from Encoder
#         - decoder_hidden - hidden state output from Decoder
#
#         Returns:
#         ---------
#         - context - context vector with a size of (1,2048)
#         - atten_weight - probabilities, express the feature relevance
#         """
#         # add additional dimension to a hidden (need for summation later)
#         decoder_hidden = decoder_hidden.unsqueeze(1)
#         atten_1 = self.W_a(features)
#         atten_2 = self.U_a(decoder_hidden)
#         # apply tangent to combined result from 2 fc layers
#         atten_tan = torch.tanh(atten_1 + atten_2)
#         # one score corresponds to one Encoder's output
#         atten_score = self.v_a(atten_tan)
#         atten_weight = F.softmax(atten_score, dim=1)
#         # first, we will multiply each vector by its softmax score
#         # next, we will sum up this vectors, producing the attention context vector
#         context = torch.sum(atten_weight * features,
#                             dim=1)
#         return context