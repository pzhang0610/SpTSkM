import argparse
import numpy as np
import pickle
# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
from torchlight import str2bool
from .processor import Processor
from .evaluate import evaluate
from .hard_mine_triplet_loss import TripletLoss,CrossEntropyLabelSmooth
import pdb


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') !=- 1:
        m.weight.data.normal_(0.0, 0.001)
        m.bias.data.fill_(0)


class REC_Processor(Processor):

    def load_model(self):
        print('-----------------------------> load recognition/model')
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        # self.loss = nn.CrossEntropyLoss()
        self.loss = CrossEntropyLabelSmooth(num_classes=self.arg.model_args['num_class'], use_gpu=True)
        self.criterion_rank = TripletLoss(margin=0.3)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        print('--------------------------->>>recognition/train')
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        # pdb.set_trace()
        for data, label in loader:
            # get data
            # pdb.set_trace()
            # if torch.any(torch.isnan(data)):
            #     print(label)
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            # pdb.set_trace()

            # forward
            output, feat = self.model(data)
            xent_loss = self.loss(output, label)
            rank_loss = self.criterion_rank(feat, label)
            loss = 0.8*xent_loss + 0.2*rank_loss
            # loss = rank_loss
            # print(loss)


            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss'] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, ranks=[1, 5, 10, 20]):
        self.model.eval()
        gallery_loader = self.data_loader['gallery']
        gf, g_pids= [], []
        for data, label in gallery_loader:
            # pdb.set_trace()
            # print(camid)
            data = data.float().to(self.dev)

            with torch.no_grad():
                feature = self.model.extract_feature(data)
            gf.append(feature.data.cpu())
            g_pids.extend(label)
            # g_camids.extend(camid)

        # pdb.set_trace()
        gf = torch.stack(gf).squeeze()
        g_pids = np.asarray(g_pids)
        # g_camids = np.asarray(g_camids)

        print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

        qf, q_pids = [], []
        probe_loader = self.data_loader['probe']
        for data, label in probe_loader:
            data = data.float().to(self.dev)
            with torch.no_grad():
                feature = self.model.extract_feature(data)
            qf.append(feature.data.cpu())
            q_pids.extend(label)
            # q_camids.extend(camid)

        qf = torch.stack(qf).squeeze()
        q_pids = np.asarray(q_pids)
        # q_camids = np.asarray(q_camids)
        # pdb.set_trace()

        print("Extracted features for probe set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

        print("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
        # distmat = 1 - torch.mm(qf, gf.t())
        # distmat = distmat.cpu().numpy()

        # with open('results/smin_score_2.pkl', 'wb') as f:
        #     pickle.dump(distmat, f)

        print("Computing CMC and mAP")
        # pdb.set_trace()
        cmc, mAP = evaluate(distmat, q_pids, g_pids)

        print("Results ----------")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        print("------------------")

        return cmc[0]

    # def test(self, evaluation=True):
    #     print('--------------------------->>>recognition/test')
    #     self.model.eval()
    #     loader = self.data_loader['test']
    #     loss_value = []
    #     result_frag = []
    #     label_frag = []
    #
    #     for data, label in loader:
    #
    #         # get data
    #         data = data.float().to(self.dev)
    #         label = label.long().to(self.dev)
    #
    #         # inference
    #         with torch.no_grad():
    #             output, feature = self.model(data)
    #         result_frag.append(feature.data.cpu().numpy())
    #
    #         # get loss
    #         if evaluation:
    #             loss = self.loss(output, label)
    #             loss_value.append(loss.item())
    #             label_frag.append(label.data.cpu().numpy())
    #
    #     self.result = np.concatenate(result_frag)
    #     # pdb.set_trace()
    #     if evaluation:
    #         self.label = np.concatenate(label_frag)
    #         self.epoch_info['mean_loss'] = np.mean(loss_value)
    #         self.show_epoch_info()
    #
    #         # show top-k accuracy
    #         for k in self.arg.show_topk:
    #             self.show_topk(k)
    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5, 10, 20], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--pretrained_weights', type=str, default=None, help='weights for fine tune')
        # endregion yapf: enable

        return parser

