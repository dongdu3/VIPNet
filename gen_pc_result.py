from __future__ import print_function
import os
import time
import argparse
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from PIL import Image
from model import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/media/administrator/Document/project/svr/results', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
# Creat test dataloader
img_root = os.path.join(opt.dataRoot, 'testing_img_sel', opt.cat)

img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

cudnn.benchmark = True

model_path = os.path.join(opt.model, opt.cat)
test_path = os.path.join(opt.dataRoot, 'psg', opt.cat)
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Create network
network_img2pc = PointSetGenerationNetwork(z_dim=256, n_pc=1024)
network_img2pc.cuda()
network_img2pc.load_state_dict(torch.load(os.path.join(model_path, 'img2pc.pt')))

def write_pts_obj(pred, filename): # pred shape: [N, 3]
    with open(filename, 'w') as f:
        for i in range(0, pred.shape[0]):
            f.write('v ' + str(pred[i][0]) + ' ' + str(pred[i][1]) + ' ' + str(pred[i][2]) + '\n')
        f.close()

network_img2pc.eval()
total_num = len(os.listdir(img_root))
n_process = 0
test_time = 0
test_num = 0
with torch.no_grad():
    for name in os.listdir(img_root):
        img = Image.open(os.path.join(img_root, name)).convert('RGB')
        img = img_transform(img)
        img = img.cuda()
        img = img.contiguous().view(1, 3, 224, 224)

        name = name.split('.')[0]

        start_time = time.time()  # start time

        pc_pre = network_img2pc(img)*10./9.

        cost_time = time.time() - start_time
        print('time cost:', cost_time)
        if n_process > 0:
            test_time += cost_time
            test_num += 1

        # dist1, dist2 = distChamfer(pc, pc_pre)
        # loss = torch.mean(dist1) + torch.mean(dist2)

        # pc = np.array(pc.cpu().data.squeeze().numpy()).reshape((-1, 3))
        pc_pre = np.array(pc_pre.cpu().data.squeeze().numpy()).reshape((-1, 3))

        # save_image(img.squeeze(0).cpu(), os.path.join(test_path, name + '.png'))
        write_pts_obj(pc_pre, os.path.join(test_path, name + '.obj'))
        # write_pts_obj(pc, os.path.join(test_path, name + '_gt.obj'))

        n_process += 1
        print('processed %d/%d ...' % (n_process, total_num))

    print('testing done!')
    print('average testing time:', test_time / test_num)