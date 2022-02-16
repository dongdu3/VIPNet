from __future__ import print_function
import argparse
import random
import torch.backends.cudnn as cudnn
from dataset_img2vox import *
from model import *
from torchvision.utils import save_image
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--thres', type=float, default=0.5, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--test', type=str, default='test_pc_vox_refine', help='test results path')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

opt.manualSeed = random.randint(1, 10000)  # fix seed
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

# create path
model_path = os.path.join(opt.model, opt.cat)
test_path = os.path.join(opt.test, opt.cat)
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Creat testing dataloader
# using different point cloud data
dataset_test = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                        vox_root=os.path.join(opt.dataRoot, 'vox64'),
                        filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                        cat=opt.cat, mode='test', view_pick=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset_test)
print('testing set num', len_dataset)

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

# Create Loss Module
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([2.]).cuda())  # nn.BCEWithLogitsLoss()

with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, vox_gt, name, view_id = data
        img = img.cuda()
        vox_gt = vox_gt.cuda()
        pc = network_img2pc(img)

        vox_init, vox_init_sigmoid = network_img2vox(img)
        pc_occ_sigmoid = network_occnet.predict(img, pc)
        vox_update = network_vox_refine.voxel_updater(vox_init_sigmoid, pc, pc_occ_sigmoid)

        vox_ref, vox_ref_sigmoid = network_vox_refine(vox_update)

        loss_init = criterion(vox_init, vox_gt)
        loss_ref = criterion(vox_ref, vox_gt)

        vox_init_sigmoid = vox_init_sigmoid[0, 0, :, :, :]
        vox_init_sigmoid = vox_init_sigmoid.cpu().data.squeeze().numpy() + (1-opt.thres)
        vox_init_sigmoid = np.array(vox_init_sigmoid, dtype=np.uint8).reshape((64, 64, 64))

        vox_ref_sigmoid = vox_ref_sigmoid[0, 0, :, :, :]
        vox_ref_sigmoid = vox_ref_sigmoid.cpu().data.squeeze().numpy() + (1-opt.thres)
        vox_ref_sigmoid = np.array(vox_ref_sigmoid, dtype=np.uint8).reshape((64, 64, 64))

        vox_gt = vox_gt.cpu().data.squeeze().numpy()
        vox_gt = np.array(vox_gt, dtype=np.uint8).reshape((64, 64, 64))

        save_image(img.squeeze(0).cpu(), os.path.join(test_path, name[0] + '_' + view_id[0] + '.png'))

        pc = np.array(pc.cpu().data.squeeze().numpy()).reshape((-1, 3))
        write_pts_obj(pc, os.path.join(test_path, name[0] + '_' + view_id[0] + '_pre.obj'))

        write_binvox_file(vox_init_sigmoid, os.path.join(test_path, name[0] + '_' + view_id[0] + '_init.binvox'))
        write_binvox_file(vox_ref_sigmoid, os.path.join(test_path, name[0] + '_' + view_id[0] + '_ref.binvox'))
        write_binvox_file(vox_gt, os.path.join(test_path, name[0] + '_' + view_id[0] + '_gt.binvox'))

        print('testing %s, view name %s, loss_init %f, loss_ref %f' %
              (name[0], view_id[0], loss_init.item(), loss_ref.item()))

        if i > 19:
            break

print('Testing done!')
