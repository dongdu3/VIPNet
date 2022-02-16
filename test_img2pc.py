from __future__ import print_function
import sys
import time
import argparse
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from dataset_img2pc import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--test', type=str, default='test_img2pc', help='test results path')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

# Creat test dataloader
# using different point cloud data
dataset_test = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                        pc_root=os.path.join(opt.dataRoot, 'pc'),
                        filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                        cat=opt.cat, mode='test', view_pick=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
len_dataset_test = len(dataset_test)
print('test set num', len_dataset_test)

model_path = os.path.join(opt.model, opt.cat)
test_path = os.path.join(opt.test, opt.cat)
if not os.path.exists(test_path):
    os.makedirs(test_path)

cudnn.benchmark = True

# Create network
network_img2pc = PointSetGenerationNetwork(z_dim=256, n_pc=1024)
network_img2pc.cuda()
network_img2pc.load_state_dict(torch.load(os.path.join(model_path, 'img2pc.pt')))

# Create Loss Module
sys.path.append('./utils/chamferdistance/')
import dist_chamfer as ext
distChamfer =  ext.chamferDist()

def write_pts_obj(pred, filename): # pred shape: [N, 3]
    with open(filename, 'w') as f:
        for i in range(0, pred.shape[0]):
            f.write('v ' + str(pred[i][0]) + ' ' + str(pred[i][1]) + ' ' + str(pred[i][2]) + '\n')
        f.close()

total_time = 0
test_num = 0
network_img2pc.eval()
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, pc, name, view_id = data
        img = img.cuda()
        pc = pc.cuda()

        start_time = time.time()
        pc_pre = network_img2pc(img)*10./9.
        cost_time = time.time() - start_time

        print('time cost:', cost_time)
        if i > 0:
            total_time += cost_time
            test_num += 1

        dist1, dist2 = distChamfer(pc, pc_pre)
        loss = torch.mean(dist1) + torch.mean(dist2)

        pc = np.array(pc.cpu().data.squeeze().numpy()).reshape((-1, 3))
        pc_pre = np.array(pc_pre.cpu().data.squeeze().numpy()).reshape((-1, 3))

        save_image(img.squeeze(0).cpu(), os.path.join(test_path, name[0] + '_' + view_id[0] + '.png'))
        write_pts_obj(pc_pre, os.path.join(test_path, name[0] + '_' + view_id[0] + '_pre.obj'))
        write_pts_obj(pc, os.path.join(test_path, name[0] + '_' + view_id[0] + '_gt.obj'))

        print('testing %s, view name %s, loss %f' % (name[0], view_id[0], loss.item()))

        if i > 29:
            break

print('average time:', total_time, test_num, total_time/test_num)