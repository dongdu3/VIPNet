from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
# import time,datetime
from dataset_patch_l2h import *
from model import *
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/data/dudong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--patchNum', type=int, default=64, help='input patch size')
parser.add_argument('--thres', type=float, default=0.5, help='threshold for voxel extractor')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--vipnet', type=str, default='vipnet_finetune.pt', help='model path')
parser.add_argument('--test', type=str, default='test_patch_l2h', help='model path')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

cudnn.benchmark = True

# create path
model_path = os.path.join(opt.model, opt.cat)
test_path = os.path.join(opt.test, opt.cat, 'thres_'+str(opt.thres))
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Creat train/val dataloader
dataset = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                   vox256_root=os.path.join(opt.dataRoot, 'vox256'),
                   filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                   cat=opt.cat, mode='test', patch_num=opt.patchNum, view_pick=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                         shuffle=False, num_workers=int(opt.workers))

len_dataset = len(dataset)
print('testing set num', len_dataset)

# Create network
vipnet = SVRNetworkOccGuide()
vipnet.cuda()
vipnet.load_state_dict(torch.load(os.path.join(model_path, opt.vipnet)))
vipnet.eval()
print("Previous vipnet weights loaded.")

vox_dim = 1
feat_high_dim = 16
feat_low_dim = 32
input_dim = vox_dim + feat_high_dim + feat_low_dim
network_patch_l2h = VoxelSuperResNetwork_16_64(input_dim=input_dim, gf_dim=64)
network_patch_l2h.cuda()
network_patch_l2h.load_state_dict(torch.load(os.path.join(model_path, 'patch_l2h.pt')))
network_patch_l2h.eval()
print('Previous patch_l2h network weights loaded.')

# Create Loss Module
criterion = nn.BCEWithLogitsLoss()

# index
idx = [i for i in range(0, 64)]
begin_idx = []
for i in range(0, 64, 16):
    for j in range(0, 64, 16):
        for k in range(0, 64, 16):
            begin_idx.append([i, j, k])
begin_idx64 = np.array(begin_idx, dtype=np.int32)
begin_idx256 = begin_idx64*4

with torch.no_grad():
    for it, data in enumerate(dataloader, 0):
        img, _, _, name, view_id = data
        img = img.cuda()

        vox_ref_cat_feat_high, feat_low = vipnet.predict_highres_input(img)

        vox_in = torch.zeros(opt.patchNum, vox_dim + feat_high_dim, 16, 16, 16).type(torch.FloatTensor).cuda()
        for i in range(0, opt.patchNum):  # here batch_size = 1
            vox_in[i] = vox_ref_cat_feat_high[0, :, dataset.begin_idx64[idx[i]][0]: dataset.begin_idx64[idx[i]][0] + 16,
                        dataset.begin_idx64[idx[i]][1]: dataset.begin_idx64[idx[i]][1] + 16,
                        dataset.begin_idx64[idx[i]][2]: dataset.begin_idx64[idx[i]][2] + 16]

        del vox_ref_cat_feat_high
        # torch.cuda.empty_cache()

        feat_low = feat_low.repeat(opt.patchNum, 1, 1, 1, 1)
        vox_in_cat = torch.cat([vox_in, feat_low], 1)

        del feat_low, vox_in

        _, vox_pre_sigmoid = network_patch_l2h(vox_in_cat)

        vox_pre_sigmoid_np = np.array(vox_pre_sigmoid.cpu().data.squeeze().numpy() + (1. - opt.thres),
                                      dtype=np.uint8).reshape((opt.patchNum, 64, 64, 64))
        output = np.zeros((256, 256, 256), dtype=np.uint8)
        for i in range(0, opt.patchNum):
            output[begin_idx256[i][0]: begin_idx256[i][0] + 64,
            begin_idx256[i][1]: begin_idx256[i][1] + 64,
            begin_idx256[i][2]: begin_idx256[i][2] + 64] = vox_pre_sigmoid_np[i, :, :, :]

        save_image(img.squeeze(0).cpu(), os.path.join(test_path, name[0] + '_' + view_id[0] + '.png'))
        write_binvox_file(output, os.path.join(test_path, name[0] + '_pre.binvox'), voxel_size=256)

        print('processed', name[0], view_id[0])

        # release the memory
        del img, vox_in_cat, vox_pre_sigmoid, vox_pre_sigmoid_np, output

        torch.cuda.empty_cache()

        if it > 19:
            break

print('Testing done!')