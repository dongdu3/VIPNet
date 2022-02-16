from __future__ import print_function
import argparse
import time
import os
import torch.backends.cudnn as cudnn
from PIL import Image
from model import *
from torchvision.utils import save_image
from common import *
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/media/administrator/Document/project/svr/results', help='data root path')
parser.add_argument('--thres', type=float, default=0.3, help='input batch size')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--voxelSize', type=int, default=64, help='voxel resolution')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--vipnet', type=str, default='vipnet_finetune.pt', help='model path')
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
test_path = os.path.join(opt.dataRoot, 'vipnet_'+str(opt.voxelSize), opt.cat)
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Create network
network = SVRNetworkOccGuide()
network.cuda()
network.load_state_dict(torch.load(os.path.join(model_path, opt.vipnet)))
network.eval()

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

        _, vox_ref_sigmoid = network.predict(img, b_return_mid=False)

        cost_time = time.time() - start_time
        print('time cost:', cost_time)
        if n_process > 0:
            test_time += cost_time
            test_num += 1

        vox_ref_sigmoid = vox_ref_sigmoid[0, 0, :, :, :].cpu().data.squeeze().numpy()

        mesh = extract_mesh(vox_ref_sigmoid, threshold=opt.thres, n_face_simp=5000)
        mesh.export(os.path.join(test_path, name + '.ply'), 'ply')

        output = vox_ref_sigmoid + (1. - opt.thres)
        output = np.array(output, dtype=np.uint8).reshape((64, 64, 64))

        # save_image(img.squeeze(0).cpu(), os.path.join(test_path, name + '.png'))
        write_binvox_file(output, os.path.join(test_path, name + '.binvox'))
        # write_binvox_file(vox_gt, os.path.join(opt.test, name[0] + '_' + view_id[0] + '_gt.binvox'))

        n_process += 1
        print('processed %d/%d ...' % (n_process, total_num))

    print('testing done!')
    print('average testing time:', test_time / test_num)
