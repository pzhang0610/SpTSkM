from __future__ import print_function, absolute_import
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import cv2
import pdb


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            # print(img_path)
            img = np.asarray(cv2.imread(img_path))[:, :, ::-1]  # BGR --> RGB, img_height * img_width * 3
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def read_mask(mask_path):
    got_mask = False
    while not got_mask:
        try:
            # print(mask_path)
            mask = np.asarray(cv2.threshold(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)[1])
            got_mask = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(mask_path))
            pass
    return mask

class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, normalization=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.normalization = normalization

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, mask_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample == 'random':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = range(num)
            rand_end = max(0, len(frame_indices) - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, len(frame_indices))

            indices = list(frame_indices[begin_index:end_index])

            for index in indices:
                if len(indices) >= self.seq_len:
                    break
                indices.append(index)
            indices=np.array(indices)
            img_masks = []
            for index in indices:
                index=int(index)
                img_path = img_paths[index]
                mask_path = mask_paths[index]
                # print(img_path, '=======>',mask_path)
                # pdb.set_trace()
                img = read_image(img_path)
                mask = read_mask(mask_path)
                img_mask = np.concatenate((img, np.expand_dims(mask, axis=2)), axis=2)
                img_masks.append(img_mask)
            if self.transform is not None:
                img_masks = self.transform(img_masks)
            imgs = [img_masks[i][:, :, :3] for i in range(self.seq_len)]
            masks = [torch.from_numpy(np.squeeze(img_masks[i][:, :, 3:]/255.0)) for i in range(self.seq_len)]

            if self.normalization is not None:
                imgs = self.normalization(imgs)

            imgs = torch.stack(imgs, dim=0)
            masks = torch.stack(masks, dim=0)
            return imgs, masks, pid, camid

        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, batch_size needs to be set to 1.
            This sampling strategy is used in test phase.
            """
            cur_index = 0
            frame_indices = range(num)
            indices_list = []
            while num-cur_index > self.seq_len:
                indices_list.append(frame_indices[cur_index:cur_index+self.seq_len])
                cur_index+=self.seq_len
            last_seq=list(frame_indices[cur_index:])
            for index in last_seq:
                if len(last_seq) >= self.seq_len:
                    break
                last_seq.append(index)
            indices_list.append(last_seq)
            imgs_list = []
            masks_list = []
            for indices in indices_list:
                img_masks = []
                for index in indices:
                    index = int(index)
                    img = read_image(img_paths[index])
                    mask = read_mask(mask_paths[index])
                    img_mask = np.concatenate((img, np.expand_dims(mask, axis=2)), axis=2)
                    img_masks.append(img_mask)

                if self.transform is not None:
                    img_masks = self.transform(img_masks)
                imgs = [img_masks[i][:, :, :3] for i in range(self.seq_len)]
                masks = [torch.from_numpy(np.squeeze(img_masks[i][:, :, 3:] / 255.0)) for i in range(self.seq_len)]

                if self.normalization is not None:
                    imgs = self.normalization(imgs)
                imgs = torch.stack(imgs, dim=0)
                masks = torch.stack(masks, dim=0)

                imgs_list.append(imgs)
                masks_list.append(masks)

            img_tensor = torch.stack(imgs_list, dim=0)
            mask_tensor = torch.stack(masks_list, dim=0)

            return img_tensor, mask_tensor, pid, camid
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))



