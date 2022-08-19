import os
from abc import abstractmethod

import time
import torch
import pandas as pd
from numpy import inf

from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence

class BaseTrainer(object):
    def __init__(self, model, criterion, metric_ftns, bert_metrics, optimizer, args):
        self.args = args

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(args.n_gpu)
        self.model = model.to(self.device)
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.bert_scorer = bert_metrics
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.save_period = self.args.save_period

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1
        self.checkpoint_dir = os.path.join(args.save_dir, args.model_type)

        # only used for SAT model loss
        self.alpha_c = args.alpha_c

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

    @abstractmethod
    def _train_epoch_image(self, epoch):
        raise NotImplementedError

    @abstractmethod
    def _train_epoch_ss(self, epoch):
        raise NotImplementedError

    def train(self):
        not_improved_count = 0
        for epoch in tqdm(range(self.start_epoch, self.epochs + 1)):

            if self.args.model_type == 'R2G':
                result = self._train_epoch_r2g(epoch)
            elif self.args.model_type == 'SAT':
                result = self._train_epoch_sat(epoch)
            else:
                result = self._train_epoch_ss(epoch)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self._record_best(log)

            # print logged informations to the screen
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def eval(self):

        if self.args.model_type == 'R2G':
            self.eval_r2g()

        elif self.args.model_type == 'SAT':
            self.eval_sat()

        else:
            self.eval_ss()


    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
        self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)
        record_path = os.path.join(self.args.record_dir, 'mimic_cxr_' + self.args.model_type + '.csv')
        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)
        record_table = record_table.append(self.best_recorder['val'], ignore_index=True)
        record_table = record_table.append(self.best_recorder['test'], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, bert_metrics, optimizer, args, lr_scheduler, train_dataloader, test_dataloader,
                 val_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, bert_metrics,  optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

    def _train_epoch_r2g(self, epoch):

        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, reports_masks, _) in enumerate(tqdm(self.train_dataloader)):
            images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(
                self.device)

            output = self.model(images, reports_ids, mode='train')

            loss = self.criterion(output, reports_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, _) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, _) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log

    def _train_epoch_sat(self, epoch):

        train_loss = 0
        self.model.train()
        for batch_idx, (images_id, images, reports_ids, _, report_len) in enumerate(tqdm(self.train_dataloader)):

            images, reports_ids, report_len = images.to(self.device), reports_ids.to(self.device), report_len.to(self.device)

            scores, caps_sorted, decode_lengths, alphas = self.model(images, reports_ids, report_len, mode='train')

            # since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # remove timesteps that we didn't decode at, or are pads
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # calculate the loss
            loss = self.criterion(scores, targets) + self.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            train_loss += loss.item()

            # backprop
            self.optimizer.zero_grad()
            loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)

            # update weights
            self.optimizer.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images_id, images, reports_ids, _, report_len) in enumerate(self.val_dataloader):

                images, reports_ids = images.to(self.device), reports_ids.to(self.device)

                output = self.model(images, targets_len= report_len, mode='sample')

                reports = self.model.tokenizer.decode_batch(output)

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, _, report_len) in enumerate(self.test_dataloader):

                images, reports_ids = images.to(self.device), reports_ids.to(self.device)

                output = self.model(images, targets_len= report_len, mode='sample')

                reports = self.model.tokenizer.decode_batch(output)

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        # self.lr_scheduler.step(log['val_BLEU_4'])
        self.lr_scheduler.step()
        print(self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[1]["lr"])

        return log

    def _train_epoch_ss(self, epoch): # work under progress.

        train_loss = 0
        # train_tokens = 0

        self.model.train()
        for batch_idx, (study_id, reports_ids, impression_ids, reports_masks) in enumerate(tqdm(self.train_dataloader)):

            reports_ids, impression_ids, reports_masks = reports_ids.to(self.device), impression_ids.to(self.device), reports_masks.to(self.device)

            output = self.model(reports_ids, impression_ids, mode='train')

            loss = self.criterion(output, impression_ids, reports_masks)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        #     n_tokens = (impression_ids[:, 1:] != 1).data.sum()
        #
        #     loss, loss_node = self.criterion(output, impression_ids[:, 1:], n_tokens)
        #
        #     train_loss += loss.item()
        #     train_tokens += n_tokens.cpu()
        #
        #     loss_node.backward()
        #     self.optimizer.step()
        #     self.lr_scheduler.step()
        #     self.optimizer.zero_grad()
        #
        # log = {'train_loss': train_loss / train_tokens}

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (study_id, reports_ids, impression_ids, _) in enumerate(self.val_dataloader):

                reports_ids, impression_ids = reports_ids.to(self.device), impression_ids.to(self.device)

                output = self.model(reports_ids, impression_ids, mode='sample')
                impressions = self.model.tokenizer_out.decode_batch(output)
                ground_truths = self.model.tokenizer_out.decode_batch(impression_ids[:, 1:].cpu().numpy())
                val_res.extend(impressions)
                val_gts.extend(ground_truths)
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (study_id, reports_ids, impression_ids, _) in enumerate(self.test_dataloader):
                reports_ids, impression_ids = reports_ids.to(self.device), impression_ids.to(self.device)

                output = self.model(reports_ids, impression_ids, mode='sample')
                impressions = self.model.tokenizer_out.decode_batch(output)
                ground_truths = self.model.tokenizer_out.decode_batch(impression_ids[:, 1:].cpu().numpy())
                test_res.extend(impressions)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        self.lr_scheduler.step()

        return log

    def eval_r2g(self):

        log = {}
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'model_best.pth'))

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print("model's state dict loaded")

        with torch.no_grad():
            test_ids, test_gts, test_res = [], [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks, _) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(
                    self.device), reports_masks.to(self.device)

                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                test_ids.extend(images_id)
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            test_met['BERTscore'] = self.bert_scorer(test_gts, test_res)

            log.update(**{'test_' + k: v for k, v in test_met.items()})

            path = os.path.join(self.args.record_dir, 'mimic_cxr_' + self.args.model_type + '_report' + '.csv')

            df = pd.DataFrame({'image_id': test_ids, 'actual report': test_gts, 'generated report': test_res})
            df.to_csv(path, index=False)

            print(log)

        return log

    def eval_sat(self):

        log = {}
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'model_best.pth'))

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print("model's state dict loaded")

        with torch.no_grad():
            test_ids, test_gts, test_res = [], [], []
            for batch_idx, (images_id, images, reports_ids, _, report_len) in enumerate(self.test_dataloader):

                images, reports_ids = images.to(self.device), reports_ids.to(self.device)

                output = self.model(images, targets_len= report_len, mode='sample')

                reports = self.model.tokenizer.decode_batch(output)

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                test_ids.extend(images_id)
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            test_met['BERTscore'] = self.bert_scorer(test_gts, test_res)

            log.update(**{'test_' + k: v for k, v in test_met.items()})

            path = os.path.join(self.args.record_dir, 'mimic_cxr_' + self.args.model_type + '_report' + '.csv')

            df = pd.DataFrame({'image_id': test_ids, 'actual report': test_gts, 'generated report': test_res})
            df.to_csv(path, index=False)

            print(log)

        return log

    def eval_ss(self):

        log = {}
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, 'model_best.pth'))

        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        print("model's state dict loaded")

        with torch.no_grad():
            test_ids, test_gts, test_res = [], [], []
            for batch_idx, (study_id, reports_ids, impression_ids, _) in enumerate(self.test_dataloader):
                reports_ids, impression_ids = reports_ids.to(self.device), impression_ids.to(self.device)

                output = self.model(reports_ids, impression_ids, mode='sample')
                impressions = self.model.tokenizer_out.decode_batch(output)
                ground_truths = self.model.tokenizer_out.decode_batch(impression_ids[:, 1:].cpu().numpy())

                test_ids.extend(study_id)
                test_res.extend(impressions)
                test_gts.extend(ground_truths)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            test_met['BERTscore'] = self.bert_scorer(test_gts, test_res)

            log.update(**{'test_' + k: v for k, v in test_met.items()})

            path = os.path.join(self.args.record_dir, 'mimic_cxr_' + self.args.model_type + '_report' + '.csv')

            df = pd.DataFrame({'study_id': test_ids, 'actual report': test_gts, 'generated report': test_res})
            df.to_csv(path, index=False)

            print(log)

        return log
