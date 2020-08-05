import argparse
import os
import os.path as osp
import pickle
import random
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from data import init_dataset, VideoDataset, RandomIdentitySampler, BatchSampler
from data import video_transforms, volume_transforms
from model.motionnet import CelebNet
from model.utils import AverageMeter, now
from model.eval_metrics import evaluate
from model.hard_mine_triplet_loss import TripletLoss, CrossEntropyLabelSmooth
import pdb

def get_data(name,  height, width, batch_size, num_instance, seq_len=15, test_batch=1, workers=8):
    data_src = init_dataset(name=name)

    train_transformer = video_transforms.Compose([video_transforms.RandomExpandCrop(height, width, interpolation='bilinear'),
                                            video_transforms.RandomHorizontalFlip()])

    normalizer = video_transforms.Compose([volume_transforms.ClipToTensorList()])

    test_transformer = video_transforms.Resize(size=(width, height), interpolation='bilinear')

    train_loader = DataLoader(VideoDataset(data_src.train, seq_len=seq_len, sample='random',
                                           transform=train_transformer, normalization=normalizer),
                              sampler=RandomIdentitySampler(data_src.train, num_instances=num_instance),
                              batch_size=batch_size, num_workers=workers, pin_memory=True, drop_last=True)

    gallery_loader = DataLoader(VideoDataset(data_src.gallery, seq_len=seq_len, sample='dense',
                                             transform=test_transformer, normalization=normalizer),
                                batch_size=test_batch, num_workers=workers, pin_memory=True, drop_last=False)

    query_loader = DataLoader(VideoDataset(data_src.query, seq_len=seq_len, sample='dense',
                                             transform=test_transformer, normalization=normalizer),
                                batch_size=test_batch, num_workers=workers, pin_memory=True, drop_last=False)

    return train_loader, gallery_loader, query_loader, data_src.num_train_pids

# def check_mem():
#     mem = os.popen(
#         '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().split(
#         ",")
#
#     return mem

def main(args):
    # total, used = check_mem()
    #
    # total = int(total)
    # used = int(used)
    #
    # max_mem = int(total * 0.9)
    # block_mem = max_mem - used
    #
    # x = torch.rand((256, 1024, block_mem)).cuda()
    # x = torch.randn((2,2)).cuda()
    # pdb.set_trace()
    logs_path = osp.join(args.ckpt_path, 'logs.txt')
    if not osp.exists(args.ckpt_path):
        os.makedirs(args.ckpt_path)
    with open(logs_path, 'w') as f:
        f.write(str(vars(args)))
    f.close()

    cudnn.benchmark = True
    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, gallery_loader, query_loader, num_classes = get_data(args.dataset, args.height, args.width, args.batch_size,
                                                          args.num_instance, args.seq_len, args.test_batch,
                                                          args.workers)

    model = CelebNet(num_classes, args.num_features, dropout=args.dropout, set_pooling=args.set_pooling)
    model = nn.DataParallel(model).to(devices)

    if args.evaluate:
        print('{} Evaluating...'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        checkpoint = torch.load(osp.join(args.ckpt_path, 'model_900.pth.tar'))
        model.load_state_dict(checkpoint['state_dict'])
        rank1 = test(model, query_loader, gallery_loader, out_feature=True)
        return

    if args.resume and osp.exists(args.resume):
        print("{} Loaded checkpoint '{}'".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.resume))
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])

    if args.save_params:
        print("{} Loaded checkpoint '{}'".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.resume))
        pretrained_dict = torch.load(args.resume)['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        torch.save({'state_dict': model_dict}, 'imgs_dict.pth.tar')
        model.load_state_dict(model_dict)
        rank1 = test(model, query_loader, gallery_loader, out_feature='imgs')
        return

    # img_base_param_ids = set(map(id, model.module.img_base.parameters()))
    # mask_base_param_ids = set(map(id, model.module.mask_base.parameters()))
    # base_param_ids = img_base_param_ids | mask_base_param_ids
    #
    # img_base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.img_base.parameters())
    # mask_base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.mask_base.parameters())
    # new_params = [p for p in model.parameters() if
    #               id(p) not in base_param_ids]
    # # pdb.set_trace()
    # param_groups = [
    #     {'params': img_base_params_need_for_grad, 'lr_mult': 0.1},
    #     {'params': mask_base_params_need_for_grad, 'lr_mult': 1.0},
    #     {'params': new_params, 'lr_mult': 1.0}
    # ]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.SGD(param_groups, lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay,
    #                             nesterov=True)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs_decay, gamma=args.gamma)
    # def adjust_lr(epoch):
    #     step_size = args.epochs_decay
    #     lr = args.lr * (0.1 ** (epoch // step_size))
    #
    #     for g in optimizer.param_groups:
    #         g['lr'] = lr * g.get('lr_mult', 1)

    # criterion_xent = nn.CrossEntropyLoss()
    # criterion_xent.cuda()
    criterion_xent = CrossEntropyLabelSmooth(num_classes=num_classes, use_gpu=True)
    criterion_rank = TripletLoss(margin=args.margin)

    best_rank1 = - np.inf
    for epoch in range(args.start_epoch, args.max_epoch):
        # adjust_lr(epoch)
        scheduler.step()
        print("{} Epoch {}/{}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, args.max_epoch))
        train(model, train_loader, criterion_xent, criterion_rank, optimizer)
        if epoch+1 >= args.eval_start and ((epoch + 1) % args.eval_step == 0 or (epoch + 1) == args.max_epoch):
            # save middle models

            print("{} Testing...".format(now()))
            # rank1 = test(model, query_loader, gallery_loader, out_feature=True)
            # is_best = rank1 > best_rank1
            # if is_best: best_rank1 = rank1
            state_dict = model.state_dict()
            if not osp.exists(args.ckpt_path):
                os.makedirs(args.ckpt_path)
            # if is_best:
            #     ckpt_name = osp.join(args.ckpt_path, 'best_model.pth.tar')
            #     torch.save({'state_dict': state_dict,
            #                 'epoch': epoch + 1}, ckpt_name)
            ckpt_name = osp.join(args.ckpt_path, 'model_{}.pth.tar'.format(epoch+1))
            torch.save({'state_dict': state_dict,
                        'epoch': epoch + 1}, ckpt_name)


def train(model, data_loader, criterion_xent, criterion_rank, optimizer):
    model.train()
    losses = AverageMeter()
    # pdb.set_trace()
    for idx, (imgs, masks, labels, camids) in enumerate(data_loader):
        # pdb.set_trace()
        # print(imgs.shape, '----->', labels)
        imgs = imgs.float().cuda()
        masks = masks.float().cuda()
        labels = labels.long().cuda()

        logit_imgs, logit_masks, logit_glob, feat = model(imgs, masks, out_feature=False)

        # img_labels = labels.repeat(args.seq_len, 1).permute(1, 0).contiguous().view(1, -1).squeeze()
        loss_imgs = criterion_xent(logit_imgs, labels)
        loss_masks = criterion_xent(logit_masks, labels)
        loss_globs = criterion_xent(logit_glob, labels)
        loss_rank = criterion_rank(feat, labels)
        loss = 0.5 * (0.4 * loss_imgs + 0.1 * loss_masks + 0.5 * loss_globs) + 0.5 * loss_rank
        # loss = 0.5 * (0.3*loss_imgs + 0.3*loss_masks +  0.4*loss_globs) + 0.5 * loss_rank
        # loss = 0.4*loss_imgs + 0.1*loss_masks + 0.5*loss_globs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss, labels.size(0))
        if (idx + 1) % args.print_freq == 0:
            print(
            "{} Batch [{}/{}]\t lr: {}, ImgLoss {:.6f} MaskLoss {:.6f} GlobLoss {:.6f} RankLoss {:.6f} Loss {:.6f}({:.6f})"
                .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), idx + 1, len(data_loader),
                        optimizer.param_groups[0]['lr'], loss_imgs, loss_masks, loss_globs, loss_rank,
                        losses.val, losses.avg))

        if idx == len(data_loader) - 1:
            break


def test(model, query_loader, gallery_loader, out_feature=True, ranks=[1, 5, 10, 20]):
    model.eval()
    gf, g_pids, g_camids = [], [], []
    with torch.no_grad():
        for idx, (imgs, masks, labels, camids) in enumerate(gallery_loader):
            # print(imgs.shape, '----->', labels.item())
            # pdb.set_trace()
            #b, n, s, c, h, w = imgs.size()
            if imgs.shape[1] >= 20:
                # pdb.set_trace()
                select = sorted(random.sample(range(imgs.shape[1]), 20))
                imgs = imgs[:, select, :, :, :, :]
                masks = masks[:, select, :, :, :]
            # with torch.no_grad():
            imgs = imgs.float().cuda()
            masks = masks.float().cuda()
            # b, n, s, c, h, w = imgs.size()
            # assert (b==1)
            # imgs = imgs.view(b*n, s, c, h, w)
            features = model(imgs, masks, out_feature)
            # features  = features.view(n, -1)
            # features = torch.max(features, 0)[0]
            features = torch.mean(features, 0)
            features = features.data.cpu()#.clone().to('cpu')
            gf.append(features)
            g_pids.extend(labels)
            g_camids.extend(camids)

            # del imgs
            # del masks
            # del features
    gf = torch.stack(gf)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    print("Extracted features for gallery set, obtained {}-by-{} matrix".format(gf.size(0), gf.size(1)))

    qf, q_pids, q_camids = [], [], []
    with torch.no_grad():
        for idx, (imgs, masks, labels, camids) in enumerate(query_loader):
            if imgs.shape[1] >=20:
                select = sorted(random.sample(range(imgs.shape[1]), 20))
                imgs = imgs[:, select, :, :, :, :]
                masks = masks[:, select, :, :, :]
            #with torch.no_grad():
            imgs = imgs.float().cuda()
            masks = masks.float().cuda()

            features = model(imgs, masks, out_feature)
            # features, _ = torch.max(features, 0)
            # features = torch.max(features, 0)[0]
            features = torch.mean(features, 0)
            features = features.data.cpu()#.clone().to('cpu')
            qf.append(features)
            q_pids.extend(labels)
            q_camids.extend(camids)
            # del imgs
            # del masks
            # del features
    qf = torch.stack(qf)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)
    print("Extracted features for query set, obtained {}-by-{} matrix".format(qf.size(0), qf.size(1)))

    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.cpu().numpy()
    with open('results/ssin_score_mars.pkl', 'wb') as f:
        pickle.dump(distmat, f)

    print("Computing CMC and mAP")
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

    print("Results ----------")
    print("mAP: {:.1%}".format(mAP))
    print("CMC curve")
    for r in ranks:
        print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
    print("------------------")

    return cmc[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--dataset', type=str, default='mars', help='Name of the dataset')
    # parser.add_argument('--num_classes', type=int, default=50, help='classes of the training set')
    parser.add_argument('--height', type=int, default=256, help='height of image')
    parser.add_argument('--width', type=int, default=128, help='width of image')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch samples for training, includes batch_size/num_instance identities')
    parser.add_argument('--num_instance', type=int, default=4, help='number of instances for each identity in a batch')
    parser.add_argument('--seq_len', type=int, default=4, help='number of frames of each instance')
    parser.add_argument('--test_batch', type=int, default=1,
                        help='number of test samples in a batch')
    parser.add_argument('--workers', type=int, default=16, help='number of workers to load data')

    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate or not, if specified, eval phase will be conducted.')
    parser.add_argument('--resume', type=str, default=None,
                        help='evaluate or not, if specified, eval phase will be conducted.')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate')  # 0.08(17.9), 0.008(25.1), 0.003(25.2), 0.0003(15.8), 0.001(18.6), 0.005(23.4)--> dropout=0.6
    parser.add_argument('--weight_decay', type=float, default=5e-04, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='weight decay')
    parser.add_argument('--epochs_decay', type=int, default=200, help='step size for lr_scheduler')
    parser.add_argument('--gamma', type=float, default=0.1, help='step size for lr_scheduler')

    parser.add_argument('--num_features', type=int, default=1024, help='num_features')
    parser.add_argument('--set_pooling', type=str, default='max', help='type of pooling, max or mean')
    parser.add_argument('--dropout', type=float, default=0.5, help='type of pooling, max or mean')
    parser.add_argument('--margin', type=float, default=0.3, help='margin of triplet loss')

    parser.add_argument('--start_epoch', type=int, default=0, help='the epoch to start training')
    parser.add_argument('--max_epoch', type=int, default=1000, help='max epoch for training')
    parser.add_argument('--print_freq', type=int, default=10, help='steps for print logs')
    parser.add_argument('--eval_step', type=int, default=50, help='epochs for evaluate')
    parser.add_argument('--eval_start', type=int, default=100, help='epochs for evaluate')

    parser.add_argument('--save_params', action='store_true', help='save params')
    parser.add_argument('--ckpt_path', type=str, default='./img_logs/adam', help='path to store checkpoint')

    args = parser.parse_args()
    main(args)
