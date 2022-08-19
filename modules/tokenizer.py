from abc import abstractmethod

import json
import re
from collections import Counter
#imports for testing module
# import argparse


class Tokenizer(object):
    def __init__(self, args, column=None):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.ann = json.loads(open(self.ann_path, 'r').read())

        assert column is not None, "column value can't be None, Please provide column name while initializing " \
                                       "tokenizer "
        assert column in ['report', 'impression'], "column value can be either 'report' or 'impression'"

        print('Building vocabulary with threshold {}'.format(args.threshold))
        self.token2idx, self.idx2token = self.create_vocabulary(column)
        print('Vocabulary length {}'.format(self.get_vocab_size()))

    @abstractmethod
    def create_vocabulary(self, column=None):
        raise NotImplementedError

    @abstractmethod
    def decode(_self, ids):
        raise NotImplementedError


    def clean_report(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.idx2token)

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out

class Tokenizer_R2G(Tokenizer):
    def __int__(self, args, column=None):
        super(Tokenizer_R2G, self).__int__(args, column)

    def __call__(self, txt):
        tokens = self.clean_report(txt).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def create_vocabulary(self, column=None):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example[column]).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                pass
        return txt


class Tokenizer_others(Tokenizer):
    def __int__(self, args, column=None):
        super(Tokenizer_others, self).__int__(args, column)

    def __call__(self, txt):
        tokens = self.clean_report(txt).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [2] + ids + [3]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 3:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            elif idx == 3:
                break
        return txt

class Tokenizer_SAT(Tokenizer_others):
    def __init__(self, args, column):
        super(Tokenizer_SAT, self).__init__(args, column)

    def create_vocabulary(self, column=None):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example[column]).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold]
        vocab.sort()
        token2idx, idx2token = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}, {0: '<pad>', 1: '<unk>', 2 : '<bos>', 3: '<eos>'}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 4
            idx2token[idx + 4] = token
        return token2idx, idx2token

class Tokenizer_SS(Tokenizer_others):
    def __init__(self, args, column):
        super(Tokenizer_SS, self).__init__(args, column)

    def create_vocabulary(self, column=None):
        total_tokens = []

        for example in self.ann['train']:
            tokens = self.clean_report(example[column]).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold]
        vocab.sort()
        token2idx, idx2token = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3}, {0: '<unk>', 1: '<pad>', 2 : '<bos>', 3: '<eos>'}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 4
            idx2token[idx + 4] = token
        return token2idx, idx2token

# testing area below!!!

# def parse_args():
#     parser = argparse.ArgumentParser()
#
#     # Tokenizer input settings
#     parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
#     parser.add_argument('--ann_path', type=str, default='D:\\TU Berlin\\Thesis\\Codes\\Master\\data\\annotated.json',
#                         help='the path to the '
#                              'directory '
#                              'containing the '
#                              'data.')
#     args = parser.parse_args()
#     return args
#
#
# if __name__ == '__main__':
#     # parse arguments
#     args = parse_args()
#
#     # create tokenizer
#     tokenizer = Tokenizer_image(args, 'impression')
#     print(tokenizer.idx2token)
