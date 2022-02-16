import os
import numpy as np
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image
import binvox_rw

class ShapeNet(data.Dataset): ## resolution: partial 16 to 64 (whole 64 to 256)
    def __init__(self,
                 img_root='/media/administrator/Dong1/ShapeNetCore.v1/renderingimg/',
                 vox256_root='/media/administrator/Dong1/ShapeNetCore.v1/vox256/',
                 filelist_root='/media/administrator/Document/project/svr/comparison_quanlity/test_list/',
                 cat='03001627', patch_num=64, view_pick=False):

        self.img_root = img_root
        self.vox256_root = vox256_root
        self.filelist_root = filelist_root
        self.cat = cat
        self.patch_num = patch_num

        self.img_dir = os.path.join(self.img_root, self.cat)
        self.vox256_dir = os.path.join(self.vox256_root, self.cat)

        self.idx = [i for i in range(0, 64)]
        begin_idx = []
        for i in range(0, 64, 16):
            for j in range(0, 64, 16):
                for k in range(0, 64, 16):
                    begin_idx.append([i, j, k])
        self.begin_idx64 = np.array(begin_idx, dtype=np.int32)
        self.begin_idx256 = self.begin_idx64*4
        # print('begin_idx64:', self.begin_idx64)
        # print('begin_idx256:', self.begin_idx256)

        self.list_file = os.path.join(self.filelist_root, self.cat+'.txt')
        fnames = open(self.list_file, 'r').readlines()
        fnames = [f.strip() for f in fnames]

        self.data_paths = []
        for name in fnames:
            vox256_path = os.path.join(self.vox256_dir, name + '.binvox')
            img_view_path = os.path.join(self.img_dir, name, 'rendering')
            with open(os.path.join(img_view_path, 'renderings.txt'), 'r') as f:
                view_list = f.readlines()[0: 2]     # just pick the first 5 views
                if not view_pick:
                    for view_name in view_list:
                        view_name = view_name.strip()
                        img_path = os.path.join(img_view_path, view_name)
                        self.data_paths.append((img_path, vox256_path, name, view_name.split('.')[0]))
                else:
                    view_name = view_list[0].strip()
                    img_path = os.path.join(img_view_path, view_name)
                    self.data_paths.append((img_path, vox256_path, name, view_name.split('.')[0]))
                f.close()


        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        img_path, vox256_path, name, view_id = self.data_paths[index]

        # for image
        img = Image.open(img_path).convert('RGB')
        img_data = self.img_transform(img)

        # for vox256
        fp = open(vox256_path, 'rb')
        vox_gt = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
        vox_gt = torch.from_numpy(vox_gt).type(torch.FloatTensor)
        fp.close()

        return img_data, vox_gt, name, view_id

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    dataset = ShapeNet()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader, 0):
        img, vox_gt, name, view_id = data
        print(img)
        print(vox_gt)
        print(img.shape)
        print(vox_gt.shape)
        print(name)
        print(view_id)
        break
