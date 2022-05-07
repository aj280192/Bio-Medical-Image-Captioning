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

        self.vocab_size_in = len(tokenizer_in.idx2token) + 1
        self.vocab_size_out = len(tokenizer_out.idx2token) + 1

        self.seqtransformer = Seq2SeqTransformer(self.args, self.vocab_size_in, self.vocab_size_out)

        for p in self.seqtransformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]
        device = src.device

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

        src_padding_mask = (src == 0).transpose(0, 1)
        tgt_padding_mask = (tgt == 0).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


    def forward(self, report_ids, impression_ids, mode='train'):
        if mode == 'train':

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(report_ids, impression_ids[:-1, :])
            output = self.seqtransformer(report_ids, impression_ids[:-1, :], src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask, mode='forward')

        elif mode == 'sample':
            src_mask = (report_ids != 0).unsqueeze(-2)
            print(report_ids.shape, src_mask.shape)
            max_len = impression_ids.size(1)
            output = self.seqtransformer(report_ids.T, src_mask, max_len, 0, mode='sample')

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
    def __init__(self,args, vocab_in_size, vocab_out_size):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=args.emb_dim,
                                       nhead=args.num_heads,
                                       num_encoder_layers=args.num_layers,
                                       num_decoder_layers=args.num_layers,
                                       dim_feedforward=args.d_ff,
                                       dropout=args.dropout)

        self.generator = nn.Linear(args.emb_dim, vocab_out_size)
        self.src_tok_emb = TokenEmbedding(vocab_in_size, args.emb_dim)
        self.tgt_tok_emb = TokenEmbedding(vocab_out_size, args.emb_dim)
        self.positional_encoding = PositionalEncoding(
            args.emb_dim, dropout=args.dropout)

    def subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
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

    def _sample(self, src, src_mask, max_len, start_symbol):
        output = torch.zeros((src.size(0), max_len))
        for i in range(src.size(0)):
            memory = self.encode(src[i].unsqueeze(0), src_mask[i].unsqueeze(0))
            ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src[i].unsqueeze(0).data)
            for j in range(max_len - 1):
                out = self.decode(
                    memory, src_mask[i].unsqueeze(0), ys, self.subsequent_mask(ys.size(1)).type_as(src[i].unsqueeze(0).data)
                )
                prob = self.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
                ys = torch.cat(
                    [ys, torch.zeros(1, 1).type_as(src[i].unsqueeze(0).data).fill_(next_word)], dim=1
                )
            output[i] = ys.squeeze(0)
        return output
