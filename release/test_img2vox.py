from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
import time
from torchvision.utils import save_image
from dataset_img2vox import *
from model import *
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--test', type=str, default='test_img2vox', help='test results path')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

# Creat test dataloader
dataset_test = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                        vox_root=os.path.join(opt.dataRoot, 'vox64'),
                        filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                        cat=opt.cat, mode='test', view_pick=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=opt.batchSize,
                                              shuffle=False, num_workers=int(opt.workers))
print('test set num', len(dataset_test))

cudnn.benchmark = True
len_dataset_test = len(dataset_test)

model_path = os.path.join(opt.model, opt.cat)
test_path = os.path.join(opt.test, opt.cat)
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Create network
network_img2vox = VoxelNetwork64(z_dim=128, gf_dim=128)
network_img2vox.cuda()
network_img2vox.load_state_dict(torch.load(os.path.join(model_path, 'img2vox64.pt')))

# Create Loss Module
criterion = nn.BCEWithLogitsLoss()

test_time = 0
test_num = 0
network_img2vox.eval()
with torch.no_grad():
    for i, data in enumerate(dataloader_test, 0):
        img, vox, name, view_id = data
        img = img.cuda()
        vox = vox.cuda()

        print('prediction', name[0], '...')
        start_time = time.time()  # start time

        vox_pre, vox_pre_sigmoid = network_img2vox(img)

        cost_time = time.time() - start_time
        print('time cost:', cost_time)
        if i > 0:
            test_time += cost_time
            test_num += 1

        loss = criterion(vox_pre, vox)

        vox_pre_sigmoid = vox_pre_sigmoid[0, 0, :, :, :].cpu().data.squeeze().numpy()
        output = vox_pre_sigmoid + 0.5
        vox = vox.cpu().data.squeeze().numpy()
        output = np.array(output, dtype=np.uint8).reshape((64, 64, 64))
        vox = np.array(vox, dtype=np.uint8).reshape((64, 64, 64))

        save_image(img.squeeze(0).cpu(), os.path.join(test_path, name[0] + '_' + view_id[0] + '.png'))
        write_binvox_file(output, os.path.join(test_path, name[0] + '_' + view_id[0] + '_pre.binvox'))
        write_binvox_file(vox, os.path.join(test_path, name[0] + '_' + view_id[0] + '_gt.binvox'))

        mesh = extract_mesh(vox_pre_sigmoid, threshold=0.5, n_face_simp=5000)
        mesh.export(os.path.join(test_path, name[0] + '_' + view_id[0] + '.ply'), 'ply')

        print('testing %s, view name %s, loss %f' % (name[0], view_id[0], loss.item()))

        if i > 19:
            break

print('testing done!')
print('average testing time:', test_time, test_num, test_time/test_num)
