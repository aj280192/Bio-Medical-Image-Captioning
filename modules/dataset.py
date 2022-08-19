import os
import json
from PIL import Image
from torch.utils.data import Dataset


# # imports for testing module
# import argparse
# from tokenizer import Tokenizer
# from torchvision import transforms


class BaseDataset(Dataset):
    def __init__(self, args, split):
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.examples = self.ann[self.split]

    def __len__(self):
        return len(self.examples)


class ImageDataset(BaseDataset):
    def __init__(self, args, split, tokenizer, transform=None):
        super().__init__(args, split)

        self.image_dir = args.image_dir
        self.tokenizer = tokenizer
        self.transform = transform

        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['impression'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

class SSDataset(BaseDataset):

    def __init__(self, args, split, tokenizer_in, tokenizer_out):
        super().__init__(args, split)

        self.tokenizer_in = tokenizer_in
        self.tokenizer_out = tokenizer_out

        for i in range(len(self.examples)):
            self.examples[i]['report_ids'] = tokenizer_in(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['impression_ids'] = tokenizer_out(self.examples[i]['impression'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['impression_ids'])

    def __getitem__(self, idx):
        example = self.examples[idx]
        study_id = example['study_id']
        report_ids = example['report_ids']
        impression_ids = example['impression_ids']
        masks = example['mask']
        rep_length = len(report_ids)
        imp_length = len(impression_ids)
        sample = (study_id, report_ids, impression_ids, rep_length, imp_length, masks)
        return sample

# class BaseSSDataset(BaseDataset):
#
#     def __init__(self, args, split):
#         super().__init__(args, split)
#
#     def __getitem__(self, idx):
#         example = self.examples[idx]
#         study_id = example['study_id']
#         report_ids = example['report_ids']
#         impression_ids = example['impression_ids']
#         rep_length = len(report_ids)
#         imp_length = len(impression_ids)
#         sample = (study_id, report_ids, impression_ids, rep_length, imp_length)
#         return sample


# class STSDataset(BaseSSDataset):
#     def __init__(self, args, split, tokenizer):
#         super().__init__(args, split)
#
#         self.tokenizer = tokenizer
#
#         for i in range(len(self.examples)):
#             # check if max length needed for both report and impression
#             self.examples[i]['report_ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
#             self.examples[i]['impression_ids'] = tokenizer(self.examples[i]['impression'])[:self.max_seq_length]
#

# class SSIODataset(BaseSSDataset):
#     def __init__(self, args, split, tokenizer_in, tokenizer_out):
#         super().__init__(args, split)
#
#         self.tokenizer_in = tokenizer_in
#         self.tokenizer_out = tokenizer_out
#
#         for i in range(len(self.examples)):
#             # check if max length needed for both report and impression
#             self.examples[i]['report_ids'] = tokenizer_in(self.examples[i]['report'])[:self.max_seq_length]
#             self.examples[i]['impression_ids'] = tokenizer_out(self.examples[i]['impression'])[:self.max_seq_length]

# testing area below!!!

# def parse_args():
#
#     parser = argparse.ArgumentParser()
#
#     # Tokenizer input settings
#     parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
#     parser.add_argument('--ann_path', type=str, default='D:\\TU Berlin\\Thesis\\Codes\\Master\\data\\annotated.json',
#                         help='the path to the '
#                              'directory '
#                              'containing the '
#                              'data.')
#     parser.add_argument('--column-type', type=int, default=2,
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
#     tokenizer = Tokenizer(args)
#     # tokenizer_in = Tokenizer(args,'report')
#     # tokenizer_out = Tokenizer(args,'impression')
#
#     # create dataset for R2G
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406),
#                              (0.229, 0.224, 0.225))])
#
#     dataset = R2GDataset(args, split, tokenizer, transform)
#     print(dataset[1])
