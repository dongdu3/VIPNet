import os
import numpy as np
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image

class ShapeNet(data.Dataset):
    def __init__(self,
                 img_root='/media/administrator/Dong1/ShapeNetCore.v1/renderingimg/',
                 filelist_root='/media/administrator/Dong1/ShapeNetCore.v1/train_val_test_list/',
                 cat='03001627', mode='train', view_pick=False):

        self.img_root = img_root
        self.filelist_root = filelist_root
        self.cat = cat
        self.mode = mode

        self.img_dir = os.path.join(self.img_root, self.cat)

        self.list_file = os.path.join(self.filelist_root, self.cat+'_'+mode+'.txt')
        fnames = open(self.list_file, 'r').readlines()
        fnames = [f.strip() for f in fnames]

        self.data_paths = []
        for name in fnames:
            img_view_path = os.path.join(self.img_dir, name, 'rendering')
            with open(os.path.join(img_view_path, 'renderings.txt'), 'r') as f:
                view_list = f.readlines()
                if not view_pick:
                    for view_name in view_list:
                        view_name = view_name.strip()
                        img_path = os.path.join(img_view_path, view_name)
                        self.data_paths.append((img_path, name, view_name.split('.')[0]))
                else:
                    view_name = view_list[0].strip()
                    img_path = os.path.join(img_view_path, view_name)
                    self.data_paths.append((img_path, name, view_name.split('.')[0]))
                f.close()

        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # # RandomResizedCrop or RandomCrop
        # self.dataAugmentation = transforms.Compose([
        #     transforms.RandomCrop(127),
        #     transforms.RandomHorizontalFlip(),
        # ])
        # self.validating = transforms.Compose([
        #     transforms.CenterCrop(127),
        # ])

    def __getitem__(self, index):
        img_path, name, view_id = self.data_paths[index]
        img = Image.open(img_path).convert('RGB')
        img_data = self.img_transform(img)

        return img_data, name, view_id

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    dataset = ShapeNet(mode='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    for i, data in enumerate(dataloader, 0):
        img, name, view_id = data
        print(img)
        print(name)
        print(view_id)
        break
