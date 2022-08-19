import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from modules.dataset import ImageDataset, SSDataset # STSDataset, SSIODataset

# imports for testing module
# import argparse
# from tokenizer import Tokenizer


class ImageDataLoader(DataLoader):
    def __init__(self, args, split, shuffle, tokenizer):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        self.dataset = ImageDataset(self.args, self.split, self.tokenizer, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, images, reports_ids, reports_masks, seq_lengths = zip(*data)
        images = torch.stack(images, 0)
        max_seq_length = max(seq_lengths)

        targets = np.zeros((len(reports_ids), max_seq_length), dtype=int)
        targets_masks = np.zeros((len(reports_ids), max_seq_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            targets[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(reports_masks):
            targets_masks[i, :len(report_masks)] = report_masks

        return images_id, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks), torch.LongTensor(seq_lengths).unsqueeze(1)

class SSDataLoader(DataLoader):
    def __init__(self, args, split, shuffle, tokenizer_in, tokenizer_out):
        self.args = args
        self.split = split
        self.shuffle = shuffle
        self.tokenizer_in = tokenizer_in
        self.tokenizer_out = tokenizer_out
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size

        self.dataset = SSDataset(self.args, self.split, self.tokenizer_in, self.tokenizer_out)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        study_ids, reports_ids, impressions_ids, rep_lengths, imp_lengths, masks = zip(*data)

        max_rep_length = max(rep_lengths)
        max_imp_length = max(imp_lengths)

        reports = np.ones((len(reports_ids), max_rep_length), dtype=int)
        impressions = np.ones((len(impressions_ids), max_imp_length), dtype=int)
        targets_masks = np.zeros((len(impressions_ids), max_imp_length), dtype=int)

        for i, report_ids in enumerate(reports_ids):
            reports[i, :len(report_ids)] = report_ids

        for i, impression_ids in enumerate(impressions_ids):
            impressions[i, :len(impression_ids)] = impression_ids

        for i, impression_masks in enumerate(masks):
            targets_masks[i, :len(impression_masks)] = impression_masks

        return study_ids, torch.LongTensor(reports), torch.LongTensor(impressions), torch.FloatTensor(targets_masks)

# class STSDataLoader(DataLoader):
#     def __init__(self, args, split, shuffle, tokenizer):
#         self.args = args
#         self.split = split
#         self.shuffle = shuffle
#         self.tokenizer = tokenizer
#         self.num_workers = args.num_workers
#         self.batch_size = args.batch_size
#
#         self.dataset = STSDataset(self.args, self.split, self.tokenizer)
#
#         self.init_kwargs = {
#             'dataset': self.dataset,
#             'batch_size': self.batch_size,
#             'shuffle': self.shuffle,
#             'collate_fn': self.collate_fn,
#             'num_workers': self.num_workers
#         }
#         super().__init__(**self.init_kwargs)
#
#     @staticmethod
#     def collate_fn(data):
#         study_ids, reports_ids, impressions_ids, rep_lengths, imp_lengths = zip(*data)
#
#         max_rep_length = max(rep_lengths)
#         max_imp_length = max(imp_lengths)
#
#         reports = np.zeros((len(reports_ids), max_rep_length), dtype=int)
#         impressions = np.zeros((len(impressions_ids), max_imp_length), dtype=int)
#
#         for i, report_ids in enumerate(reports_ids):
#             reports[i, :len(report_ids)] = report_ids
#
#         for i, impression_ids in enumerate(impressions_ids):
#             impressions[i, :len(impression_ids)] = impression_ids
#
#         return study_ids, torch.LongTensor(reports), torch.LongTensor(impressions)
#
#
# class SSIODataLoader(DataLoader):
#     def __init__(self, args, split, shuffle, tokenizer_in, tokeniner_out):
#         self.args = args
#         self.split = split
#         self.shuffle = shuffle
#         self.tokenizer_in = tokenizer_in
#         self.tokenizer_out = tokeniner_out
#         self.num_workers = args.num_workers
#         self.batch_size = args.batch_size
#
#         self.dataset = SSIODataset(self.args, self.split, self.tokenizer_in, self.tokenizer_out)
#
#         self.init_kwargs = {
#             'dataset': self.dataset,
#             'batch_size': self.batch_size,
#             'shuffle': self.shuffle,
#             'collate_fn': self.collate_fn,
#             'num_workers': self.num_workers
#         }
#         super().__init__(**self.init_kwargs)
#
#     @staticmethod
#     def collate_fn(data):
#         study_ids, reports_ids, impressions_ids, rep_lengths, imp_lengths = zip(*data)
#
#         max_rep_length = max(rep_lengths)
#         max_imp_length = max(imp_lengths)
#
#         reports = np.zeros((len(reports_ids), max_rep_length), dtype=int)
#         impressions = np.zeros((len(impressions_ids), max_imp_length), dtype=int)
#
#         for i, report_ids in enumerate(reports_ids):
#             reports[i, :len(report_ids)] = report_ids
#
#         for i, impression_ids in enumerate(impressions_ids):
#             impressions[i, :len(impression_ids)] = impression_ids
#
#         return study_ids, torch.LongTensor(reports), torch.LongTensor(impressions)

# testing area below!!!
#
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
#     parser.add_argument('--column-type', type=int, default=1,
#                         choices={1, 2},
#                         help='please select column type'
#                              '1 - tokenizer with one column, '
#                              '2 - tokenizer with two columns.')
#
#     # Dataset input settings
#     parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
#     parser.add_argument('--image_dir', type=str, default='D:\\TU Berlin\\Thesis\\Codes\\Master\\data\\images',
#                         help='the path to the directory containing the data.')
#
#     # Dataloader input settings
#     parser.add_argument('--num_workers', type=int, default=1, help='the number of workers for dataloader.')
#     parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')
#
#     args = parser.parse_args()
#     return args
#
#
# if __name__ == '__main__':
#     # parse arguments
#     args = parse_args()
#
#     # split
#     split = 'val'
#
#     # create tokenizer
#     # tokenizer = Tokenizer(args)
#     tokenizer_in = Tokenizer(args, '1', 'report')
#     tokenizer_out = Tokenizer(args, '1', 'impression')
#
#     # create dataset for R2G
#     val_loader = SSDataLoader(args, split, False, tokenizer_in, tokenizer_out)
#     for id, r, i, rm in val_loader:
#         print(r.shape)

