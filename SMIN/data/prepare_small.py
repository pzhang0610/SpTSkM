import os
import os.path as osp
import numpy as np
import pickle
from scipy.io import loadmat
import pandas as pd
from tqdm import tqdm
import pdb

def index_dir(path, suffix=None):
    if suffix is None:
        return [osp.join(path, i) for i in os.listdir(path)]
    else:
        return [osp.join(path, i) for i in os.listdir(path) if i.endswith(suffix)]


def load_dataset_info(path):
    info = pd.read_csv(path, sep=',')
    return sorted(list(set(info['pid'])))


class CVID_reID(object):
    def __init__(self, data_path, data_split_path, shuffle=True):
        self.data_path = data_path
        self.identities = load_dataset_info(data_split_path)
        # pdb.set_trace()
        if shuffle:
            np.random.shuffle(self.identities)

    def index_train(self, train_name, train_label):
        print('Writing data for training...\n')
        train_identities = self.identities
        _label = []
        _label_name = []
        _label_idx = []
        _skl_data = []
        for pid_name in tqdm(train_identities, 'ID: '):
            pid_path = osp.join(self.data_path, '%04d' % pid_name)
            records = index_dir(pid_path)
            records.sort()
            for idx_rec in records:
                skeletons = index_dir(idx_rec, '.mat')
                skeletons.sort()
                num_skl = len(skeletons)
                num_tracklet = int(np.ceil(num_skl/30))
                for tracklet in range(num_tracklet):
                    if tracklet == num_tracklet - 1:
                        if len(skeletons[tracklet*30:]) >=8:
                            tracklet_frms = skeletons[-30:]
                        else:
                            break
                    else:
                        tracklet_frms = skeletons[tracklet*30:(tracklet+1)*30]

                    _seq = np.zeros(shape=(4, 30, 14))
                    for idx, mat in enumerate(tracklet_frms):
                        data = loadmat(mat)
                        skl_data = data['new_joint']
                        # skl_data = data['joints']
                        _seq[:, idx, :] = skl_data.transpose()
                    # _label.append(['%04d' % pid_name, int(pid_name)])
                    _label_name.append('%04d' % pid_name)
                    _label_idx.append(int(pid_name))
                    _skl_data.append(_seq)
        _skl_data = np.stack(_skl_data)
        _label.append(_label_name)
        _label.append(_label_idx)
        np.save(train_name, _skl_data)
        with open(train_label, mode='wb') as f:
            pickle.dump(_label, f)
        f.close()

    def index_test(self, test_name, test_label, gallery_num, is_gallery=True):
        print('Writing data for testing...\n')
        train_identities = self.identities
        _label = []
        _label_name = []
        _label_idx = []
        _skl_data = []
        num_skl_gallery=[]

        for pid_name in train_identities:#tqdm(train_identities, 'ID'):
            pid_path = osp.join(self.data_path, '%04d' % pid_name)
            records = index_dir(pid_path)
            records.sort()
            assert gallery_num < len(records)
            assert gallery_num > 0
            if is_gallery:
                records = records[:gallery_num]
            else:
                records = records[gallery_num:]

            for idx_rec in records:
                skeletons = index_dir(idx_rec, '.mat')
                skeletons.sort()
                num_skl = len(skeletons)
                num_tracklet = int(np.ceil(num_skl/30))
                # pdb.set_trace()
                for tracklet in range(num_tracklet):
                    if tracklet == num_tracklet - 1:
                        if len(skeletons[tracklet*30:]) >=8:
                            tracklet_frms = skeletons[-30:]
                            num_skl_gallery.append(len(skeletons[tracklet*30:]))
                        else:
                            break
                    else:
                        tracklet_frms = skeletons[tracklet*30:(tracklet+1)*30]
                        num_skl_gallery.append(30)

                    _seq = np.zeros(shape=(4, 30, 14))
                    for idx, mat in enumerate(tracklet_frms):
                        data = loadmat(mat)
                        skl_data = data['new_joint']
                        # skl_data = data['joints']
                        _seq[:, idx, :] = skl_data.transpose()
                    _label_name.append('%04d' % pid_name)
                    _label_idx.append(int(pid_name))
                    _skl_data.append(_seq)

        print(num_skl_gallery)
        # pdb.set_trace()
        _skl_data = np.stack(_skl_data)
        _label.append(_label_name)
        _label.append(_label_idx)
        np.save(test_name, _skl_data)
        with open(test_label, mode='wb') as f:
            pickle.dump(_label, f)
        f.close()

    def index_all_identities(self):
        return index_dir(self.data_path)


if __name__ == "__main__":
    # data_path = '../../../dataset/motionset/part_joints_3d_conf'
    # data_path = '../../../dataset/motionset/norm_img_part_joint3d_conf'
    data_path = '../../../dataset/motionset/selected_skeletons'
    #
    train_split_path = '../../../dataset/celebset/info/train_050.csv'
    train_name = './small/train_data_small.npy'
    train_label = './small/train_label_small.pkl'
    trainset = CVID_reID(data_path, train_split_path)
    trainset.index_train(train_name, train_label)

    test_split_path = '../../../dataset/celebset/info/test_040.csv'
    testset = CVID_reID(data_path, test_split_path, shuffle=False)
    gallery_name = './small/gallery_data_small.npy'
    gallery_label = './small/gallery_label_small.pkl'
    testset.index_test(gallery_name, gallery_label,gallery_num=3, is_gallery=True)

    probe_name = './small/probe_data_small.npy'
    probe_label = './small/probe_label_small.pkl'
    testset.index_test(probe_name, probe_label, gallery_num=3, is_gallery=False)