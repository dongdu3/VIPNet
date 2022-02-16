from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
from dataset_img import *
from model import *
from common import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/media/administrator/Dong/ShapeNetCore.v1', help='data root path')
parser.add_argument('--saveRoot', type=str, default='/media/administrator/Document/project/svr/results', help='data root path')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--patchNum', type=int, default=64, help='input patch size')
parser.add_argument('--thres', type=float, default=0.255, help='threshold for voxel extractor')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--vipnet', type=str, default='vipnet_finetune.pt', help='model path')
parser.add_argument('--cat', type=str, default='03001627')
parser.add_argument('--cuda', type=str, default='0')
opt = parser.parse_args()

# cat_set: '03001627', '02691156', '02958343', '04090263', '04256520', '04379243'

os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda

model_path = os.path.join(opt.model, opt.cat)
test_path = os.path.join(opt.saveRoot, 'vipnet_256_wo_pc_testlist', opt.cat)
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Creat train/val dataloader
dataset = ShapeNet(img_root=os.path.join(opt.dataRoot, 'renderingimg'),
                   filelist_root=os.path.join(opt.dataRoot, 'train_val_test_list'),
                   cat=opt.cat, mode='test', view_pick=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
len_dataset = len(dataset)

cudnn.benchmark = True

# Create network
vipnet = SVRNetworkOccGuide()
vipnet.cuda()
vipnet.load_state_dict(torch.load(os.path.join(model_path, opt.vipnet)))
vipnet.eval()
print("Previous vipnet weight loaded...")

vox_dim = 1
feat_high_dim = 16
feat_low_dim = 32
input_dim = vox_dim + feat_high_dim + feat_low_dim
network_patch_l2h = VoxelSuperResNetwork_16_64(input_dim=input_dim, gf_dim=64)
network_patch_l2h.cuda()
network_patch_l2h.load_state_dict(torch.load(os.path.join(model_path, 'patch_l2h.pt')))
network_patch_l2h.eval()
print('Previous patch_l2h network weights loaded.')

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
    for i, data in enumerate(dataloader, 0):
        img, name, view_id = data
        img = img.cuda()
        name = name[0]
        view_id = view_id[0]
        print('processing %d/%d: %s' % (i+1, len_dataset, name))

        vox_ref_cat_feat_high, feat_low = vipnet.predict_highres_input(img)

        vox_in = torch.zeros(opt.patchNum, vox_dim + feat_high_dim, 16, 16, 16).type(torch.FloatTensor).cuda()
        for i in range(0, opt.patchNum):  # here batch_size = 1
            vox_in[i] = vox_ref_cat_feat_high[0, :, begin_idx64[idx[i]][0]: begin_idx64[idx[i]][0] + 16,
                            begin_idx64[idx[i]][1]: begin_idx64[idx[i]][1] + 16,
                            begin_idx64[idx[i]][2]: begin_idx64[idx[i]][2] + 16]

        feat_low = feat_low.repeat(opt.patchNum, 1, 1, 1, 1)
        vox_in_cat = torch.cat([vox_in, feat_low], 1)

        del vox_ref_cat_feat_high, feat_low, vox_in
        # torch.cuda.empty_cache()

        _, vox_pre_sigmoid = network_patch_l2h(vox_in_cat)

        vox_pre_sigmoid_np = np.array(vox_pre_sigmoid.cpu().data.squeeze().numpy()).reshape((opt.patchNum, 64, 64, 64))

        output = np.zeros((256, 256, 256), dtype=np.float32)
        for i in range(0, opt.patchNum):
            output[begin_idx256[i][0]: begin_idx256[i][0] + 64,
            begin_idx256[i][1]: begin_idx256[i][1] + 64,
            begin_idx256[i][2]: begin_idx256[i][2] + 64] = vox_pre_sigmoid_np[i, :, :, :]

        mesh = extract_mesh(output, threshold=opt.thres, n_face_simp=18000)
        mesh.export(os.path.join(test_path, name + '.ply'), 'ply')

        output = np.array(output + (1. - opt.thres), dtype=np.uint8)

        # save_image(img.squeeze(0).cpu(), os.path.join(test_path, name + '.png'))
        # write_binvox_file(output_pre, os.path.join(opt.test, name + '.binvox'), voxel_size=256)
        write_binvox_file(output, os.path.join(test_path, name + '.binvox'), voxel_size=256)

print('testing done!')
