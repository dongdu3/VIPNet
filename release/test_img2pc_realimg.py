from __future__ import print_function
import sys
import argparse
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from dataset_img2pc import *
from model import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/media/administrator/Code/don/SVR/VIPNet_SVR_logs_results/real_img', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--test', type=str, default='test_img2pc', help='test results path')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

img_root = os.path.join(opt.dataRoot, opt.cat)
img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

model_path = os.path.join(opt.model, opt.cat)
test_path = os.path.join(opt.dataRoot, opt.cat)
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

fin = open(os.path.join(img_root, 'test_list.txt'), 'r')
network_img2pc.eval()
with torch.no_grad():
    for name in fin.readlines():
        name = name.strip()
        img = Image.open(os.path.join(img_root, name)).convert('RGB')
        img = img_transform(img)
        img = img.cuda()
        img = img.contiguous().view(1, 3, 224, 224)

        name = name.split('.')[0]

        pc_pre = network_img2pc(img)
        pc_pre = np.array(pc_pre.cpu().data.squeeze().numpy()).reshape((-1, 3))

        write_pts_obj(pc_pre, os.path.join(test_path, name + '_pre.obj'))
        print('processed', name)
    fin.close()
print('Done!')