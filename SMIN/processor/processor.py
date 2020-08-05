import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import pdb
# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO


class Processor(IO):
    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        # if self.arg.phase == 'train':
        #     self.load_pretrained_weights()
        #     self.frozen_parameters()
        # else:
        #     self.load_weights()
        if self.arg.phase != 'train':
            self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()

    def init_environment(self):
        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass

    def load_pretrained_weights(self):
        print('-----------------------------> load pretrained weights')

        pretrained_dict = torch.load(self.arg.pretrained_weights)
        # pdb.set_trace()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() #if k.split('.')[0] != 'fcn'}
                           if k.split('.')[0] == 'st_gcn_networks' and int(k.split('.')[1]) > 0}
        model_dict = self.model.state_dict()
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def frozen_parameters(self):
        print('----------------------------->frozen parameters')
        for name, module in self.model._modules.items():
            if name == 'st_gcn_networks':
                for sub_name, sub_module in module._modules.items():
                    if sub_name == '0':
                        continue
                    else:
                        print(name, '-->', sub_name)
                        for param in sub_module.parameters():
                            param.requires_grad = False
            else:
                continue

    def load_data(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)
        if self.arg.gallery_feeder_args:
            self.data_loader['gallery'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.gallery_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers = self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))

            self.data_loader['probe'] = torch.utils.data.DataLoader(
                dataset = Feeder(**self.arg.probe_feeder_args),
                batch_size = self.arg.test_batch_size,
                shuffle=False,
                num_workers= self.arg.num_worker * torchlight.ngpu(
                   self.arg.device))

    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info = '\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)
            self.io.print_log(info)
            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def start(self):
        # pdb.set_trace()
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # training phase
        if self.arg.phase == 'train':
            best_rank1 = -np.inf
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train()
                self.io.print_log('Done.')

                # save model
                # if ((epoch + 1) % self.arg.save_interval == 0) or (
                #         epoch + 1 == self.arg.num_epoch):
                #     filename = 'epoch{}_model.pt'.format(epoch + 1)
                #     self.io.save_model(self.model, filename)

                # evaluation
                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch):
                    self.io.print_log('Eval epoch: {}'.format(epoch))
                    rank1 = self.test()
                    is_best = rank1 > best_rank1
                    if is_best: best_rank1 = rank1
                    if is_best:
                        filename = 'best_model.pt'
                        self.io.save_model(self.model, filename)

                    self.io.print_log('Done.')
        # test phase
        elif self.arg.phase == 'test':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test()
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')

    @staticmethod
    def get_parser(add_help=False):
        parser = argparse.ArgumentParser(add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False,
                            help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+',
                            help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100,
                            help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10,
                            help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5,
                            help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for training')
        parser.add_argument('--gallery_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')
        parser.add_argument('--probe_feeder_args', action=DictAction, default=dict(),
                            help='the arguments of data loader for test')
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default='./gcn/best_model_25.pt', help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                            help='the name of weights which will be ignored in the initialization')
        # endregion yapf: enable

        return parser