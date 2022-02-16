import os
import numpy as np
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
from PIL import Image
import binvox_rw

class ShapeNet(data.Dataset):
    def __init__(self,
                 img_root='/media/administrator/Dong1/ShapeNetCore.v1/renderingimg/',
                 vox_root='/media/administrator/Dong1/ShapeNetCore.v1/vox64/',
                 filelist_root='/media/administrator/Dong1/ShapeNetCore.v1/train_val_test_list/',
                 cat='03001627', mode='train', view_pick=False):

        self.img_root = img_root
        self.vox_root = vox_root
        self.filelist_root = filelist_root
        self.cat = cat
        self.mode = mode

        self.img_dir = os.path.join(self.img_root, self.cat)
        self.vox_dir = os.path.join(self.vox_root, self.cat)

        self.list_file = os.path.join(self.filelist_root, self.cat+'_'+mode+'.txt')
        fnames = open(self.list_file, 'r').readlines()
        fnames = [f.strip() for f in fnames]

        self.data_paths = []
        for name in fnames:
            vox_path = os.path.join(self.vox_dir, name+'.binvox')
            img_view_path = os.path.join(self.img_dir, name, 'rendering')
            with open(os.path.join(img_view_path, 'renderings.txt'), 'r') as f:
                view_list = f.readlines()
                if not view_pick:
                    for view_name in view_list:
                        view_name = view_name.strip()
                        img_path = os.path.join(img_view_path, view_name)
                        self.data_paths.append((img_path, vox_path, name, view_name.split('.')[0]))
                else:
                    view_name = view_list[0].strip()
                    img_path = os.path.join(img_view_path, view_name)
                    self.data_paths.append((img_path, vox_path, name, view_name.split('.')[0]))
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
        img_path, vox_path, name, view_id = self.data_paths[index]
        img = Image.open(img_path).convert('RGB')
        img_data = self.img_transform(img)

        fp = open(vox_path, 'rb')
        vox_data = binvox_rw.read_as_3d_array(fp, fix_coords=True).data
        vox_size = vox_data.shape[0]
        vox_data = np.array(vox_data).reshape((1, vox_size, vox_size, vox_size))
        vox_data = torch.from_numpy(vox_data).type(torch.FloatTensor)
        fp.close()

        return img_data, vox_data, name, view_id

    def __len__(self):
        return len(self.data_paths)

if __name__ == '__main__':
    # dataset = ShapeNet(mode='val')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    # for i, data in enumerate(dataloader, 0):
    #     img, vox, name, view_id = data
    #     print(img)
    #     print(vox)
    #     print(name)
    #     print(view_id)
    #     break

    dataset = ShapeNet(mode='train', view_pick=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    n_process = 0
    n_vox = 64*64*64
    n_total_time = 0
    for i, data in enumerate(dataloader, 0):
        _, vox, name, _ = data
        vox = vox.data.squeeze().numpy()
        vox = np.array(vox).reshape((64, 64, 64))
        idx_vec = np.where(vox > 0.5)
        time = 1. * (n_vox-len(idx_vec[0]))/len(idx_vec[0])
        n_total_time += time
        n_process += 1
        print(n_process, name, time)

    print('average_time:', n_total_time/n_process)      # average_time: 22.293324235902226
