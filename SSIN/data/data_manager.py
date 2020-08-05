from __future__ import print_function, absolute_import
import os
import glob
import os.path as osp
import numpy as np
import pandas as pd
from scipy.io import loadmat
import pdb


class MotionSet(object):
    # Change it to your data path
    data_root = '/data/mars/img'
    track_train_info_path = '/data/maars/info/train_050.csv'
    track_test_info_path = '/data/mars/info/test_040.csv'

    def __init__(self, min_seq_len=8, gallery_num=4):
        self._check_before_run()
        train_ids = self.load_dataset_info(self.track_train_info_path)
        test_ids = self.load_dataset_info(self.track_test_info_path)

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(train_ids, relabel=True, min_seq_len=min_seq_len, gallery_num=0, is_gallery=False, is_probe=False)

        query, num_query_tracklets, num_query_pids, num_imgs_query = \
            self._process_data(test_ids, relabel=False, min_seq_len=min_seq_len, gallery_num=gallery_num, is_gallery=False, is_probe=True)

        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
            self._process_data(test_ids, relabel=False, min_seq_len=min_seq_len, gallery_num=gallery_num, is_gallery=True, is_probe=False)
        # pdb.set_trace()

        num_imgs_per_tracklet = num_train_imgs + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> -CelebSet loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids


    def _check_before_run(self):
        if not osp.exists(self.data_root):
            raise RuntimeError("'{}' is not available".format(self.data_root))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))

    def load_dataset_info(self, path):
        info = pd.read_csv(path, sep=',')
        return sorted(list(set(info['pid'])))

    def _process_data(self, pid_list, relabel=False, min_seq_len=8, gallery_num=0, is_gallery=False, is_probe=False):
        num_pids = len(pid_list)
        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for id_idx in range(len(pid_list)):
            p_id = pid_list[id_idx]
            if relabel: p_id = pid2label[p_id]
            p_id_path = osp.join(self.data_root, '%04d'%pid_list[id_idx])
            seq_list = os.listdir(p_id_path)
            seq_list.sort()

            assert gallery_num < len(seq_list)
            if is_gallery and gallery_num > 0:
                seq_list = seq_list[:gallery_num]
            if is_probe and gallery_num < len(seq_list):
                seq_list = seq_list[gallery_num:]
            for seq_idx in range(len(seq_list)):
                imgs = glob.glob(osp.join(p_id_path, seq_list[seq_idx], '*.png'))
                imgs.sort()

                cur_track_index = None
                tracklet_img_names = []
                tracklet_mask_names = []
                for img_idx, img_absname in enumerate(imgs):

                    mask_absname = img_absname.replace('img','mask')
                    img_name = osp.basename(img_absname)
                    track_index = img_name.split('T')[1][:4]
                    if cur_track_index != track_index: #or img_idx == len(imgs)-1:
                        # if img_idx == len(imgs)-1 and cur_track_index == track_index:
                        #     tracklet_img_names.append(img_absname)
                        #     tracklet_mask_names.append(mask_absname)
                        if img_idx!= 0 and len(tracklet_img_names) >= min_seq_len:
                            tracklets.append((tuple(tracklet_img_names), tuple(tracklet_mask_names), p_id, seq_list[seq_idx], cur_track_index))
                            num_imgs_per_tracklet.append(len(tracklet_img_names))
                        tracklet_img_names = []
                        tracklet_mask_names = []
                        cur_track_index = track_index
                    tracklet_img_names.append(img_absname)
                    tracklet_mask_names.append(mask_absname)
                if len(tracklet_img_names) >= min_seq_len:
                    tracklets.append(
                        (tuple(tracklet_img_names), tuple(tracklet_mask_names), p_id, seq_list[seq_idx], cur_track_index))
                    num_imgs_per_tracklet.append(len(tracklet_img_names))
        # pdb.set_trace()
        num_tracklets = len(tracklets)
        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


class MARS(object):
    """
        MARS
        Reference:
        Zheng et al. MARS: A Video Benchmark for Large-Scale Person Re-identification. ECCV 2016.

        Dataset statistics:
        # identities: 1261
        # tracklets: 8298 (train) + 1980 (query) + 9330 (gallery)
        # cameras: 6
        Args:
            min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
        """
    root = '/data/pzhang1/PycharmProjects/CelebrityReID/CelebReID/dataset/mars'
    train_name_path = osp.join(root, 'info/train_name.txt')
    test_name_path = osp.join(root, 'info/test_name.txt')
    track_train_info_path = osp.join(root, 'info/tracks_train_info.mat')
    track_test_info_path = osp.join(root, 'info/tracks_test_info.mat')
    query_IDX_path = osp.join(root, 'info/query_IDX.mat')

    def __init__(self, min_seq_len=30):
        self._check_before_run()

        # prepare meta data
        train_names = self._get_names(self.train_name_path)
        test_names = self._get_names(self.test_name_path)
        track_train = loadmat(self.track_train_info_path)['track_train_info']  # numpy.ndarray (8298, 4)
        track_test = loadmat(self.track_test_info_path)['track_test_info']  # numpy.ndarray (12180, 4)
        query_IDX = loadmat(self.query_IDX_path)['query_IDX'].squeeze()  # numpy.ndarray (1980,)
        query_IDX -= 1  # index from 0
        track_query = track_test[query_IDX, :]
        gallery_IDX = [i for i in range(track_test.shape[0]) if i not in query_IDX]
        track_gallery = track_test[gallery_IDX, :]

        train, num_train_tracklets, num_train_pids, num_train_imgs = \
            self._process_data(train_names, track_train, home_dir='bbox_train', relabel=True, min_seq_len=min_seq_len)

        query, num_query_tracklets, num_query_pids, num_query_imgs = \
            self._process_data(test_names, track_query, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        gallery, num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = \
            self._process_data(test_names, track_gallery, home_dir='bbox_test', relabel=False, min_seq_len=min_seq_len)

        num_imgs_per_tracklet = num_train_imgs + num_query_imgs + num_gallery_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> MARS loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.test_name_path):
            raise RuntimeError("'{}' is not available".format(self.test_name_path))
        if not osp.exists(self.track_train_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_train_info_path))
        if not osp.exists(self.track_test_info_path):
            raise RuntimeError("'{}' is not available".format(self.track_test_info_path))
        if not osp.exists(self.query_IDX_path):
            raise RuntimeError("'{}' is not available".format(self.query_IDX_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, names, meta_data, home_dir=None, relabel=False, min_seq_len=0):
        assert home_dir in ['bbox_train', 'bbox_test']
        num_tracklets = meta_data.shape[0]
        pid_list = list(set(meta_data[:, 2].tolist()))
        num_pids = len(pid_list)

        if relabel: pid2label = {pid: label for label, pid in enumerate(pid_list)}
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(num_tracklets):
            data = meta_data[tracklet_idx, ...]
            start_index, end_index, pid, camid = data
            if pid == -1: continue  # junk images are just ignored
            assert 1 <= camid <= 6
            if relabel: pid = pid2label[pid]
            camid -= 1  # index starts from 0
            img_names = names[start_index - 1:end_index]

            # make sure image names correspond to the same person
            pnames = [img_name[:4] for img_name in img_names]
            assert len(set(pnames)) == 1, "Error: a single tracklet contains different person images"

            # make sure all images are captured under the same camera
            camnames = [img_name[5] for img_name in img_names]
            assert len(set(camnames)) == 1, "Error: images are captured under different cameras!"

            # append image names with directory information
            img_paths = [osp.join(self.root, home_dir, img_name[:4], img_name) for img_name in img_names]
            mask_paths = [osp.join(self.root, home_dir+'_mask', img_name[:4], img_name[:-4]+'.png') for img_name in img_names]
            if len(img_paths) >= min_seq_len:
                img_paths = tuple(img_paths)
                mask_paths = tuple(mask_paths)
                tracklets.append((img_paths, mask_paths, pid, camid))
                num_imgs_per_tracklet.append(len(img_paths))

        num_tracklets = len(tracklets)
        # pdb.set_trace()

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet


__factory = {
    'motionset': MotionSet,
    'mars': MARS
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)


if __name__ == '__main__':
    data_src = init_dataset('mars')
    # for index, (tracklet, p_id, seq_id, tracklet_id) in enumerate(data_src.train):
    #     print(p_id)

    print(data_src.train[0][1])

