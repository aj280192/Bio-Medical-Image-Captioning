import torch
import torch.nn as nn
import json
from modules.tokenizer import Tokenizer_R2G, Tokenizer_SAT, Tokenizer_SS
from modules.dataloaders import ImageDataLoader, SSDataLoader
from modules.loss import compute_loss, SimpleLossCompute
from modules.optimizers import r2g_optimizer, normal_optimizer, s2s_optimizer, sat_optimizer
from models.r2gen import R2GenModel
from models.sat import SATModel
from models.s2s import S2SModel


# testing imports
# import argparse
# import numpy as np


def dynamic_flow(args):

    config_all = json.loads(open(args.config_path, 'r').read())
    config = config_all[args.model_type]
    args.model_type = config['model_type']


    if config['model_type'] != "S2S":

        tokenizer = eval(config['tokenizer'])(args, config['columns'])

        train_dataloader = eval(config['dataloader_class'])(args, split='train', shuffle=True, tokenizer=tokenizer)
        test_dataloader = eval(config['dataloader_class'])(args, split='test', shuffle=False, tokenizer=tokenizer)
        val_dataloader = eval(config['dataloader_class'])(args, split='val', shuffle=False, tokenizer=tokenizer)

        model = eval(config['model_class'])(args, tokenizer)

        # if config['model_type'] == 'R2G':
        #     print('setting compute')
        #     criterion = compute_loss
        # else:
        #     criterion = nn.CrossEntropyLoss(ignore_index=0)

    else:

        tokenizer_in = eval(config['tokenizer'])(args, config['columns'][0])
        tokenizer_out = eval(config['tokenizer'])(args, config['columns'][1])

        train_dataloader = eval(config['dataloader_class'])(args, split='train', shuffle=True,
                                                            tokenizer_in=tokenizer_in,
                                                            tokenizer_out=tokenizer_out)
        test_dataloader = eval(config['dataloader_class'])(args, split='test', shuffle=False, tokenizer_in=tokenizer_in,
                                                           tokenizer_out=tokenizer_out)
        val_dataloader = eval(config['dataloader_class'])(args, split='val', shuffle=False, tokenizer_in=tokenizer_in,
                                                          tokenizer_out=tokenizer_out)

        model = eval(config['model_class'])(args, tokenizer_in, tokenizer_out)

        # criterion = SimpleLossCompute(len(tokenizer_out.idx2token))

    if config['model_type'] != 'SAT':
        criterion = compute_loss
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = eval(config['opt'])(args, model)

    return model, train_dataloader, test_dataloader, val_dataloader, criterion, optimizer

# testing code below

# def parse_agrs():
#     parser = argparse.ArgumentParser()
#
#     # Data input settings
#     parser.add_argument('--image_dir', type=str, default='D:\\TU Berlin\\Thesis\\Codes\\Master\\data\\images',
#                         help='the path to the directory containing the data.')
#     parser.add_argument('--ann_path', type=str, default='D:\\TU Berlin\\Thesis\\Codes\\Master\\data\\annotated.json',
#                         help='the path to the directory containing the data.')
#
#     # Configuration path
#     parser.add_argument('--config_path', type=str, default='D:\\TU Berlin\\Thesis\\Codes\\Master\\config.json')
#
#     # Data loader settings
#     parser.add_argument('--model_type', type=str, choices=['1', '2', '3', '4'], default='1', help='the model to be used.'
#                                                                                         '1. R2G model'
#                                                                                         '2. SAT model'
#                                                                                         '3. STS model'
#                                                                                         '4. SSIO model')
#     parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
#     parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
#     parser.add_argument('--num_workers', type=int, default=1, help='the number of workers for dataloader.')
#     parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')
#
#     # Model settings (for visual extractor)
#     parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
#     parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
#                         help='whether to load the pretrained visual extractor')
#
#     # Model settings (for Transformer)
#     parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
#     parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
#     parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
#     parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
#     parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
#     parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
#     parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
#     parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
#     parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
#     parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
#     parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
#     parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
#     # for Relational Memory
#     parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
#     parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
#     parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')
#
#     # Sample related
#     parser.add_argument('--sample_method', type=str, default='beam_search',
#                         help='the sample methods to sample a report.')
#     parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
#     parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
#     parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
#     parser.add_argument('--group_size', type=int, default=1, help='the group size.')
#     parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
#     parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
#     parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')
#
#     # Optimization
#     parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
#     parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
#     parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
#     parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
#     parser.add_argument('--amsgrad', type=bool, default=True, help='.')
#
#     # Others
#     parser.add_argument('--seed', type=int, default=9233, help='.')
#
#     args = parser.parse_args()
#     return args
#
# if __name__ == '__main__':
#     # parse arguments
#     args = parse_agrs()
#
#     # fix random seeds
#     torch.manual_seed(args.seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(args.seed)
#
#     # model and loaders for training.
#     model, train_dataloader, val_dataloader, test_dataloader, cri, opt = dynamic_flow(args)
