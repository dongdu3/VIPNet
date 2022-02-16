from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
# import time,datetime
from dataset_img2vox import *
from model import *
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--voxelSize', type=int, default=64, help='volumetric solution')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

# Creat test dataloader
dataset_test = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                        vox_root=os.path.join(opt.dataRoot, 'vox64'),
                        filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                        cat=opt.cat, mode='test', view_pick=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset_test)
print('test set num', len_dataset)

cudnn.benchmark = True

model_path = os.path.join(opt.model, opt.cat)

# Create network
network_img2vox = VoxelNetwork64(z_dim=128, gf_dim=128)
network_img2vox.cuda()
network_img2vox.load_state_dict(torch.load(os.path.join(model_path, 'img2vox64.pt')))

fw_iou_pre = open(os.path.join(model_path, 'iou_img2vox_' + str(opt.voxelSize) + '.txt'), 'w')

total_n = 0
total_iou_pre = 0

network_img2vox.eval()
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, vox_gt, name, view_id = data
        img = img.cuda()
        _, vox_pre_sigmoid = network_img2vox(img)

        vox_gt = vox_gt.data.numpy()
        vox_pre_sigmoid = vox_pre_sigmoid.cpu().data.numpy()

        for j in range(0, opt.batchSize):
            vox_pre = vox_pre_sigmoid[j, 0, :, :, :]
            iou_pre = compute_iou(vox_pre, vox_gt[j, 0, :, :, :])

            #
            total_n += 1
            total_iou_pre += iou_pre
            fw_iou_pre.write(str(iou_pre) + '\n')

            print('testing %d/%d, iou_pre: %f' % (total_n, len_dataset, iou_pre))

    fw_iou_pre.close()

print('Testing done!')
print('average_iou_pre:', total_iou_pre/total_n)
