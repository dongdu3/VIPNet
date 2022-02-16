import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import binvox_rw
from point_transforms import *

class ShapeNet(data.Dataset):
    def __init__(self,
                 img_root='/media/administrator/Dong1/ShapeNetCore.v1/renderingimg/',
                 vox_root='/media/administrator/Dong1/ShapeNetCore.v1/vox64/',
                 pc_root='/media/administrator/Dong1/ShapeNetCore.v1/pc/',
                 pc_occ_root='/media/administrator/Dong1/ShapeNetCore.v1/pc_occ/',
                 filelist_root='/media/administrator/Dong1/ShapeNetCore.v1/train_val_test_list/',
                 cat='03001627', mode='train', n_pc_sample=4096, view_pick=False,
                 occ_packbits=True, n_pc_occ_subsample=1024, occ_data_with_transforms=False):

        self.img_root = img_root
        self.vox_root = vox_root
        self.pc_root = pc_root
        self.pc_occ_root = pc_occ_root
        self.filelist_root = filelist_root
        self.cat = cat
        self.mode = mode
        self.n_pc_sample = n_pc_sample
        self.occ_packbits = occ_packbits
        self.occ_data_with_transforms = occ_data_with_transforms
        if mode == 'train':
            self.pc_occ_transform = SubsamplePoints(n_pc_occ_subsample)
        else:
            self.pc_occ_transform = None

        self.img_dir = os.path.join(self.img_root, self.cat)
        self.vox_dir = os.path.join(self.vox_root, self.cat)
        self.pc_dir = os.path.join(self.pc_root, self.cat)
        self.pc_occ_dir = os.path.join(self.pc_occ_root, self.cat)

        self.list_file = os.path.join(self.filelist_root, self.cat+'_'+mode+'.txt')
        fnames = open(self.list_file, 'r').readlines()
        fnames = [f.strip() for f in fnames]

        self.data_paths = []
        for name in fnames:
            pc_occ_path = os.path.join(self.pc_occ_dir, name+'.npz')
            pc_path = os.path.join(self.pc_dir, name + '.npz')
            vox_path = os.path.join(self.vox_dir, name+'.binvox')
            img_view_path = os.path.join(self.img_dir, name, 'rendering')
            with open(os.path.join(img_view_path, 'renderings.txt'), 'r') as f:
                view_list = f.readlines()
                if not view_pick:
                    for view_name in view_list:
                        view_name = view_name.strip()
                        img_path = os.path.join(img_view_path, view_name)
                        self.data_paths.append((img_path, vox_path, pc_path, pc_occ_path, name, view_name.split('.')[0]))
                else:
                    view_name = view_list[0].strip()
                    img_path = os.path.join(img_view_path, view_name)
                    self.data_paths.append((img_path, vox_path, pc_path, pc_occ_path, name, view_name.split('.')[0]))
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
        # view image data
        img_path, vox_path, pc_path, pc_occ_path, name, view_id = self.data_paths[index]
        img = Image.open(img_path).convert('RGB')
        img_data = self.img_transform(img)

        # binary voxel data
        fp = open(vox_path, 'rb')
        vox_data = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
        vox_size = vox_data.shape[0]
        vox_data = np.array(vox_data).reshape((1, vox_size, vox_size, vox_size))
        vox_data = torch.from_numpy(vox_data).type(torch.FloatTensor)
        fp.close()

        # point cloud data
        pc_data = np.load(pc_path)['points']
        idx = list(range(0, pc_data.shape[0]))
        np.random.shuffle(idx)
        pc_data = pc_data[idx[0: self.n_pc_sample]]
        pc_data = torch.from_numpy(pc_data).type(torch.FloatTensor)

        # point cloud with occupancy value data
        pc_occ_dict = np.load(pc_occ_path)
        points = pc_occ_dict['points'].astype(np.float32)
        occupancies = pc_occ_dict['occupancies']
        if self.occ_packbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)
        occ_data = {
            None: points,
            'occ': occupancies,
        }

        if self.occ_data_with_transforms:
            occ_data['loc'] = pc_occ_dict['loc'].astype(np.float32)
            occ_data['scale'] = pc_occ_dict['scale'].astype(np.float32)

        if self.pc_occ_transform is not None:
            occ_data = self.pc_occ_transform(occ_data)

        return img_data, vox_data, pc_data, occ_data, name, view_id

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    # from torchvision.utils import save_image

    dataset = ShapeNet(mode='val')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for i, data in enumerate(dataloader, 0):
        img, vox, pc, occ, name, view_id = data
        print(img.shape)
        print(vox.shape)
        print(pc.shape)
        print(occ[None])
        print(occ[None].shape)
        print(occ['occ'])
        print(occ['occ'].shape)
        print(name)
        print(view_id)

        # save_image(img.squeeze(0).cpu(), name[0] + '_' + view_id[0] + '.png')

        break
