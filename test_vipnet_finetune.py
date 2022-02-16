from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
from dataset_img2vox import *
from model import *
from torchvision.utils import save_image
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--thres', type=float, default=0.5, help='thres')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--voxelSize', type=int, default=64, help='voxel resolution')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--vipnet', type=str, default='vipnet_finetune.pt', help='model path')
parser.add_argument('--test', type=str, default='test_vipnet_finetune', help='test path')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

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
network = SVRNetworkOccGuide()
network.cuda()
network.load_state_dict(torch.load(os.path.join(model_path, opt.vipnet)))
network.eval()

# Create Loss Module
criterion = nn.BCEWithLogitsLoss()

with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, vox_gt, name, view_id = data
        img = img.cuda()
        vox_gt = vox_gt.cuda()

        vox_init, vox_init_sigmoid, pc, vox_ref, vox_ref_sigmoid = network.predict(img, b_return_mid=True)

        loss_init = criterion(vox_init, vox_gt)
        loss_ref = criterion(vox_ref, vox_gt)

        vox_init_sigmoid = vox_init_sigmoid[0, 0, :, :, :]
        vox_init_sigmoid = vox_init_sigmoid.cpu().data.squeeze().numpy() + (1.-opt.thres)
        vox_init_sigmoid = np.array(vox_init_sigmoid, dtype=np.uint8).reshape((64, 64, 64))

        vox_ref_sigmoid = vox_ref_sigmoid[0, 0, :, :, :]
        vox_ref_sigmoid = vox_ref_sigmoid.cpu().data.squeeze().numpy() + (1.-opt.thres)
        vox_ref_sigmoid = np.array(vox_ref_sigmoid, dtype=np.uint8).reshape((64, 64, 64))

        # vox_gt = vox_gt.cpu().data.squeeze().numpy()
        # vox_gt = np.array(vox_gt, dtype=np.uint8).reshape((64, 64, 64))

        save_image(img.squeeze(0).cpu(), os.path.join(test_path, name[0] + '_' + view_id[0] + '.png'))

        pc = np.array(pc.cpu().data.squeeze().numpy()).reshape((-1, 3))
        write_pts_obj(pc, os.path.join(test_path, name[0] + '_' + view_id[0] + '_pre.obj'))

        write_binvox_file(vox_init_sigmoid, os.path.join(test_path, name[0] + '_' + view_id[0] + '_init.binvox'))
        write_binvox_file(vox_ref_sigmoid, os.path.join(test_path, name[0] + '_' + view_id[0] + '_ref.binvox'))
        # write_binvox_file(vox_gt, os.path.join(opt.test, name[0] + '_' + view_id[0] + '_gt.binvox'))

        print('testing %s, view name %s, loss_init %f, loss_ref %f' % (
            name[0], view_id[0], loss_init.item(), loss_ref.item()))

        if i > 19:
            break

print('Testing done!')