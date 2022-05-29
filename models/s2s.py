from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import numpy as np


class S2SModel(nn.Module):
    def __init__(self, args, tokenizer_in, tokenizer_out):
        super(S2SModel, self).__init__()
        self.args = args
        self.tokenizer_in = tokenizer_in
        self.tokenizer_out = tokenizer_out

        self.vocab_size_in = len(tokenizer_in.idx2token)
        self.vocab_size_out = len(tokenizer_out.idx2token)

        self.seqtransformer = Seq2SeqTransformer(self.args, self.vocab_size_in, self.vocab_size_out)

        for p in self.seqtransformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.tril(torch.ones((sz, sz), device=device)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        device = src.device

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

        src_padding_mask = (src == 1)
        tgt_padding_mask = (tgt == 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, report_ids, impression_ids, mode='train'):
        if mode == 'train':

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(report_ids,
                                                                                      impression_ids[:, :-1])
            output = self.seqtransformer(report_ids, impression_ids[:, :-1], src_mask, tgt_mask, src_padding_mask,
                                         tgt_padding_mask, src_padding_mask, mode='forward')

        elif mode == 'sample':
            num_tokens = report_ids.shape[1]
            src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(report_ids.device)
            max_len = impression_ids.size(1)
            output = self.seqtransformer(report_ids, src_mask, max_len, 2, mode='sample')

        else:
            raise ValueError
        return output


# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self, args, vocab_in_size, vocab_out_size):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=args.emb_dim,
                                       nhead=args.num_heads,
                                       num_encoder_layers=args.num_layers,
                                       num_decoder_layers=args.num_layers,
                                       dim_feedforward=args.d_ff,
                                       dropout=args.dropout,
                                       batch_first=True)

        self.generator = nn.Linear(args.emb_dim, vocab_out_size)
        self.src_tok_emb = TokenEmbedding(vocab_in_size, args.emb_dim)
        self.tgt_tok_emb = TokenEmbedding(vocab_out_size, args.emb_dim)
        self.positional_encoding = PositionalEncoding(
            args.emb_dim, dropout=args.dropout)

        # beam size
        self.beam_size = args.beam_size
        self.vocab_out_size = vocab_out_size

    def subsequent_mask(self, sz, device):
        mask = (torch.tril(torch.ones((sz, sz), device=device)) == 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def _forward(self,
                 src: Tensor,
                 trg: Tensor,
                 src_mask: Tensor,
                 tgt_mask: Tensor,
                 src_padding_mask: Tensor,
                 tgt_padding_mask: Tensor,
                 memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
            self.tgt_tok_emb(tgt)), memory,
            tgt_mask)

    # def _sample(self, src, src_mask, max_len, start_symbol):
    #     output = torch.zeros((src.size(0), max_len))
    #     for i in range(src.size(0)):
    #         memory = self.encode(src[i].unsqueeze(0), src_mask)
    #         ys = torch.zeros(1, 1).fill_(start_symbol).type(torch.long).to(src.device)
    #         for j in range(max_len - 1):
    #             memory = memory.to(src.device)
    #             tgt_mask = (self.subsequent_mask(ys.size(1), ys.device)).type(torch.bool)
    #             out = self.decode(ys, memory, tgt_mask)
    #             prob = self.generator(out[:, -1])
    #             prob = torch.log_softmax(prob, dim=-1)
    #             _, next_word = torch.max(prob, dim=1)
    #             next_word = next_word.item()
    #             ys = torch.cat(
    #                 [ys, torch.zeros(1, 1).type_as(src[i].data).fill_(next_word)], dim=1
    #             )
    #         output[i] = ys.squeeze(0)
    #     return output

    def _sample(self, src, src_mask, max_len, start_symbol):

        if self.beam_size == 0:
            return self._greedy_search(src, src_mask, max_len, start_symbol)
        else:
            return self._beam_search(src, src_mask, max_len, start_symbol)

    def _greedy_search(self, src, src_mask, max_len, start_symbol):
        batch_size = src.size(0)

        memory = self.encode(src, src_mask)
        ys = torch.zeros(batch_size, 1).fill_(start_symbol).type(torch.long).to(src.device)

        for j in range(max_len - 1):
            memory = memory.to(src.device)
            tgt_mask = (self.subsequent_mask(ys.size(1), ys.device)).type(torch.bool)
            out = self.decode(ys, memory, tgt_mask)
            prob = self.generator(out[:, -1])
            prob = torch.log_softmax(prob, dim=-1)
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat(
                    [ys, next_word.unsqueeze(1)], dim=1
                )
        return ys.cpu().numpy()

    def _beam_search(self, src, src_mask, max_len, start_symbol):

        memory = self.encode(src, src_mask)
        seq = []
        for m in memory:
            seq.append(self.generate_caption(m, max_len, start_symbol))
        return seq

    def generate_caption(self, memory, max_len, start_symbol):

        k = self.beam_size
        seq_len, em_dim = memory.shape

        # we'll treat the problem as having a batch size of k
        memory = memory.unsqueeze(0)
        memory = memory.expand(k, seq_len, em_dim)

        # tensor to store top k previous seq at each step; now they're just <start>
        top_k_seqs = torch.tensor([[start_symbol]] * k, dtype=torch.long).to(memory.device)  # (k, 1)

        # tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(memory.device)  # (k, 1)

        # lists to store completed sequences along with their alphas and scores
        complete_seqs = []
        complete_seqs_scores = []

        # start decoding
        step = 1

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            tgt_mask = (self.subsequent_mask(top_k_seqs.size(1), top_k_seqs.device)).type(torch.bool)
            out = self.decode(top_k_seqs, memory, tgt_mask)
            scores = self.generator(out[:, -1])
            scores = torch.log_softmax(scores, dim=-1)

            # add the scores to prev scores
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # all the k points will have the same score for the first step (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // self.vocab_out_size  # (s)
            next_word_inds = top_k_words % self.vocab_out_size  # (s)

            # add new words to sequences, alphas
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
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            memory = memory[incomplete_inds]

            # break if things have been going on too long
            if step > max_len:
                if not (len(complete_seqs) > 0):
                    return top_k_seqs[top_k_scores.argmax()].tolist()
                break
            step += 1

        # select sequence with max score
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        return seq