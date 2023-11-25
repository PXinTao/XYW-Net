import glob
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


def read_from_pair_txt(path, filename):
    pfile = open(os.path.join(path, filename))
    filenames = pfile.readlines()
    pfile.close()

    filenames = [f.strip() for f in filenames]
    filenames = [c.split(' ') for c in filenames]
    filenames = [(os.path.join(path, c[0].replace('/', '/')),
                  os.path.join(path, c[1].replace('/', '/'))) for c in filenames]
    return filenames


def read_Pascal_Context(path, filename):
    pfile = open(os.path.join(path, filename))
    filenames = pfile.readlines()
    pfile.close()

    img_root = 'D:\DataSet\PASCAL_Context\JPEGImages'
    gt_root = 'D:\DataSet\PASCAL_Context\label_new'
    filenames = [(os.path.join(img_root, f[:-1] + '.jpg'), os.path.join(gt_root, f[:-1] + '.jpg')) for f in filenames]

    return filenames


def read_Pascal_VOC12(path, filename):
    pfile = open(os.path.join(path, filename))
    filenames = pfile.readlines()
    pfile.close()

    img_root = r'D:\DataSet\VOCdevkit\VOC2012_trainval\JPEGImages'
    gt_root = r'D:\DataSet\VOCdevkit\VOC2012_trainval\boundaries'
    filenames = [(os.path.join(img_root, f[:-1] + '.jpg'), os.path.join(gt_root, f[:-1] + '.png')) for f in filenames]

    return filenames


class BSDS_500(Dataset):
    def __init__(self, root, flag='train', VOC=False, transform=None):

        if flag == 'train':
            if VOC:
                filenames = read_from_pair_txt(root['BSDS-VOC'], 'bsds_pascal_train_pair_s5_c.lst')

            else:
                filenames = read_from_pair_txt(root['BSDS'], 'train_pair.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]


        elif flag == 'test':
            self.im_list = glob.glob(os.path.join(root['BSDS'], r'test/*.jpg'))
            self.im_list = [im_list.replace('\\', '/') for im_list in self.im_list]
            self.gt_list = [path.split('/')[-1][:-4] for path in self.im_list]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label = Image.fromarray(label.astype(np.float32) / np.max(label))
        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class PASCAL_VOC12(Dataset):
    def __init__(self, root, flag='train', transform=None):
        if flag == 'train':
            filenames = read_Pascal_VOC12(root['PASCAL-VOC12'], 'train_s5.txt')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]

        elif flag == 'test':
            filenames = read_Pascal_VOC12(root['PASCAL-VOC12'], 'Segmentation_val_2012.txt')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1].split('.')[0].split('\\')[-1] for im_name in filenames]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label = Image.fromarray(label.astype(np.float32) / 255.0)
            if np.max(label) == 0:
                print(self.im_list[item])

        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class PASCAL_Context(Dataset):
    def __init__(self, root, flag='train', transform=None):
        if flag == 'train':
            filenames = read_Pascal_Context(root['PASCAL-Context'], 'train_s5.txt')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]

        elif flag == 'test':
            filenames = read_Pascal_Context(root['PASCAL-Context'], 'test_new.txt')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [path.split('\\')[-1][:-4] for path in self.im_list]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label = Image.fromarray(label.astype(np.float32) / 255.0)
            if np.max(label) == 0:
                print(self.im_list[item])

        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class NYUD(Dataset):
    def __init__(self, root, flag='train', rgb=True, transform=None):
        if flag == 'train':
            if rgb:
                filenames = read_from_pair_txt(root['NYUD-V2'], 'image-train.lst')
            else:
                filenames = read_from_pair_txt(root['NYUD-V2'], 'hha-train.lst')
            self.im_list = [im_name[0] for im_name in filenames]
            self.gt_list = [im_name[1] for im_name in filenames]

        elif flag == 'test':
            if rgb:
                self.im_list = glob.glob(os.path.join(root['NYUD-V2'], r'test\Images\*.png'))
            else:
                self.im_list = glob.glob(os.path.join(root['NYUD-V2'], r'test\HHA\*.png'))
            self.gt_list = [path.split('\\')[-1][:-4] for path in self.im_list]

        self.length = self.im_list.__len__()
        self.transform = transform
        self.flag = flag

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        image = Image.open(self.im_list[item])
        if self.flag == 'train':
            label = np.array(Image.open(self.gt_list[item]).convert('L'))
            label = Image.fromarray(label.astype(np.float32) / np.max(label))
        elif self.flag == 'test':
            label = Image.open(self.im_list[item])

        sample = {'images': image, 'labels': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

