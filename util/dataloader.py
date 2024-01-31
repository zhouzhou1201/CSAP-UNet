import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A #数据增强
import cv2


class SkinDataset(data.Dataset):
    """
    dataloader for skin lesion segmentation tasks
    """
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)
        self.size = len(self.images)

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor()])
        
        self.transform = A.Compose( #数据增强
            [
                A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, rotate_limit=25, p=0.5, border_mode=0),#平移、缩放、旋转
                A.ColorJitter(),#改变图像的属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
                A.HorizontalFlip(),#水平翻转
                A.VerticalFlip()#垂直翻转
            ]
        )

    def __getitem__(self, index):
        
        image = self.images[index]
        gt = self.gts[index]
        gt = gt/255.0

        transformed = self.transform(image=image, mask=gt)
        image = self.img_transform(transformed['image'])
        gt = self.gt_transform(transformed['mask'])
        return image, gt

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, batchsize, shuffle=True, num_workers=0, pin_memory=True):#num_workers=4

    dataset = SkinDataset(image_root, gt_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root):
        self.images = np.load(image_root)
        self.gts = np.load(gt_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.images[self.index]
        image = self.transform(image).unsqueeze(0)
        gt = self.gts[self.index]
        gt = gt/255.0
        self.index += 1

        return image, gt



if __name__ == '__main__':
    path = 'data/'
    tt = SkinDataset(path+'data_train.npy', path+'mask_train.npy')

    for i in range(50):
        img, gt = tt.__getitem__(i)

        img = torch.transpose(img, 0, 1)
        img = torch.transpose(img, 1, 2)
        img = img.numpy()
        gt = gt.numpy()

        plt.imshow(img)#用于接收处理图像，但无法展示，与plt.show()不同
        plt.savefig('vis/'+str(i)+".jpg")#存储文件
 
        plt.imshow(gt[0])
        plt.savefig('vis/'+str(i)+'_gt.jpg')
