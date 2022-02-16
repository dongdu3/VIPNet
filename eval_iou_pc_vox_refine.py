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
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

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
network_img2vox = VoxelNetwork64(z_dim=128, gf_dim=128)
network_img2vox.cuda()
network_img2vox.load_state_dict(torch.load(os.path.join(model_path, 'img2vox64.pt')))
network_img2vox.eval()

network_occnet = OccupancyNetwork()
network_occnet.cuda()
network_occnet.load_state_dict(torch.load(os.path.join(opt.model, 'occnet.pt')))
network_occnet.eval()

network_img2pc = PointSetGenerationNetwork(z_dim=256, n_pc=1024)
network_img2pc.cuda()
network_img2pc.load_state_dict(torch.load(os.path.join(model_path, 'img2pc.pt')))
network_img2pc.eval()

network_vox_refine = VoxelRefineNetwork()
network_vox_refine.cuda()
network_vox_refine.load_state_dict(torch.load(os.path.join(model_path, 'vox64_refine.pt')))
network_vox_refine.eval()

fw_iou_ref = open(os.path.join(model_path, 'iou_pc_vox_refine_' + str(opt.voxelSize) + '.txt'), 'w')

total_n = 0
total_iou_ref = 0

with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, vox_gt, name, view_id = data
        img = img.cuda()

        pc = network_img2pc(img)

        _, vox_init_sigmoid = network_img2vox(img)
        pc_occ_sigmoid = network_occnet.predict(img, pc)
        vox_update = network_vox_refine.voxel_updater(vox_init_sigmoid, pc, pc_occ_sigmoid)

        _, vox_ref_sigmoid = network_vox_refine(vox_update)

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