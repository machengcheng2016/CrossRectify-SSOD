"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import random

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# note: if you used our download scripts, this should be right
VOC_ROOT = osp.join(HOME, "Data/voc/VOCdevkit")


class VOCAnnotationTransform_uda(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection_uda(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """
    # image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
    # image_sets = [('2007', 'trainval'), ('2014','only_voc')],
    # image_sets = [('2012', 'trainval'), ('2014', 'COCO')],

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 img_transform_base=None, 
                 img_transform_additional=None, 
                 img_transform_tail=None,
                 target_transform=VOCAnnotationTransform_uda(),
                 dataset_name='VOC0712'):
                 
        self.root = root
        self.coco_root = ''
        self.image_set = image_sets
        self.img_transform_base = img_transform_base
        self.img_transform_additional = img_transform_additional
        self.img_transform_tail = img_transform_tail
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        self.unlabel_ids = list()
        for (year, name) in image_sets:
            if year == '2007':
                if name == 'trainval':
                    rootpath = osp.join(self.root, 'VOC' + year)
                    for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                        self.ids.append((rootpath, line.strip()))
            else:
                if name == 'trainval':
                    rootpath = osp.join(self.root, 'VOC' + year)
                    for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                        self.unlabel_ids.append((rootpath, line.strip()))
                        #self.ids.append((rootpath, line.strip()))
                else:
                    rootpath = osp.join(self.coco_root)
                    for line in open(osp.join(rootpath, name + '.txt')):
                        self.unlabel_ids.append((rootpath, line.strip()))

        self.ids = random.sample(self.ids, len(self.ids))
        self.unlabel_ids = random.sample(self.unlabel_ids, len(self.unlabel_ids))
        self.ids = self.ids + self.unlabel_ids


    def __getitem__(self, index):
        # im, gt, h, w = self.pull_item(index)
        im_weak, im_strong, gt, h, w, semi = self.pull_item(index)

        # return im, gt
        return im_weak, im_strong, gt, semi


    def __len__(self):
        return len(self.ids)


    def pull_item(self, index):
        img_id = self.ids[index]

        if img_id[0][(len(img_id[0]) - 7):] == 'VOC2007':
            img = cv2.imread(self._imgpath % img_id)
            height, width, channels = img.shape
            target = ET.parse(self._annopath % img_id).getroot()
            if self.target_transform is not None:
                target = self.target_transform(target, width, height)
            target = np.array(target)
            semi = np.array([1])
        elif img_id[0][(len(img_id[0]) - 7):] == 'VOC2012':
            img = cv2.imread(self._imgpath % img_id)
            height, width, channels = img.shape
            target = np.zeros([1, 5])
            semi = np.array([0])
        else:
            img = cv2.imread('%s/%s' % img_id)
            height, width, channels = img.shape
            target = np.zeros([1, 5])
            semi = np.array([0])

        #if self.img_transform_base is not None:
        img_base, boxes, labels = self.img_transform_base(img,      target[:, :4], target[:, 4])
        target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        
        img_weak, _, _ = self.img_transform_tail(img_base, None, None)
        img_weak = img_weak[:, :, (2, 1, 0)]  # to rgb
        
        img_strong, _, _ = self.img_transform_additional(img_base, None, None)
        img_strong, _, _ = self.img_transform_tail(img_strong, None, None)
        img_strong = img_strong[:, :, (2, 1, 0)]  # to rgb

        return torch.from_numpy(img_weak).permute(2, 0, 1), torch.from_numpy(img_strong).permute(2, 0, 1), target, height, width, semi


    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
