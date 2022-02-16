from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
# import time,datetime
from common import *
from dataset_img2vox import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--voxelSize', type=int, default=64, help='voxel resolution')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--model_name', type=str, default='vipnet_finetune', help='model name')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691_156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

cudnn.benchmark = True

# Creat testing dataloader
# using different point cloud data
dataset_test = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                        vox_root=os.path.join(opt.dataRoot, 'vox64'),
                        filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                        cat=opt.cat, mode='test', view_pick=False)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset_test)
print('testing set num', len_dataset)

model_path = os.path.join(opt.model, opt.cat)

# Create network
network = SVRNetworkOccGuide()
network.cuda()
network.load_state_dict(torch.load(os.path.join(model_path, opt.model_name+'.pt')))
network.eval()

fw_iou_ref = open(os.path.join(model_path, 'iou_' + opt.model_name + '_' + str(opt.voxelSize) + '.txt'), 'w')

total_n = 0
total_iou_ref = 0

with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, vox_gt, name, view_id = data
        img = img.cuda()

        _, vox_ref_sigmoid = network.predict(img)

        vox_gt = vox_gt.data.numpy()
        vox_ref_sigmoid = vox_ref_sigmoid.cpu().data.numpy()

        for j in range(0, opt.batchSize):
            vox_ref = vox_ref_sigmoid[j, 0, :, :, :]
            iou_ref = compute_iou(vox_ref, vox_gt[j, 0, :, :, :])

            #
            total_n += 1
            total_iou_ref += iou_ref
            fw_iou_ref.write(str(iou_ref) + '\n')

            print('testing %d/%d, iou_ref: %f' % (total_n, len_dataset, iou_ref))

    fw_iou_ref.close()

print('Testing done!')
print('average_iou_ref:', total_iou_ref / total_n)