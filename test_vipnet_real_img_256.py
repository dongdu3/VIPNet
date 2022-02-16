from __future__ import print_function
import argparse
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from dataset_patch_l2h import *
from model import *
from common import *
from PIL import Image
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--dataRoot', type=str, default='/media/administrator/Code/don/SVR/VIPNet_SVR_logs_results/real_img', help='data root path')
parser.add_argument('--patchNum', type=int, default=64, help='input patch size')
parser.add_argument('--thres', type=float, default=0.2, help='threshold for voxel extractor')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--model', type=str, default='checkpoint', help='model path')
parser.add_argument('--vipnet', type=str, default='vipnet_finetune.pt', help='model path')
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
vipnet = SVRNetworkOccGuide()
vipnet.cuda()
vipnet.load_state_dict(torch.load(os.path.join(model_path, opt.vipnet)))
vipnet.eval()
print("Previous vipnet weight loaded...")

vox_dim = 1
feat_high_dim = 16
feat_low_dim = 32
input_dim = vox_dim + feat_high_dim + feat_low_dim
network_patch_l2h = VoxelPatchSuperResNetwork(input_dim=input_dim, gf_dim=64)
network_patch_l2h.cuda()
network_patch_l2h.load_state_dict(torch.load(os.path.join(model_path, 'patch_l2h_with_pc_guidance.pt')))
print("loaded vox_l2h_voxfeat_1_pc_guide.pt weights.")

# index
idx = [i for i in range(0, 64)]
begin_idx = []
for i in range(0, 64, 16):
    for j in range(0, 64, 16):
        for k in range(0, 64, 16):
            begin_idx.append([i, j, k])
begin_idx64 = np.array(begin_idx, dtype=np.int32)
begin_idx256 = begin_idx64*4

parchNum_half = opt.patchNum//2
fin = open(os.path.join(img_root, 'test_list.txt'), 'r')
with torch.no_grad():
    for name in fin.readlines():
        name = name.strip()
        img = Image.open(os.path.join(img_root, name)).convert('RGB')
        img = img_transform(img)
        img = img.cuda()
        img = img.contiguous().view(1, 3, 224, 224)

        name = name.split('.')[0]

        _, vox_lr, pc_pre = vipnet.predict(img, b_return_pc=True)
        vox_lr = vox_lr.contiguous().view(64, 64, 64).cpu().data.numpy()

        # save pc_pre
        pc_pre = np.array(pc_pre.cpu().data.squeeze().numpy()).reshape((-1, 3))
        write_pts_obj(pc_pre, os.path.join(test_path, name + '_pc.obj'))

        # save the low-res output as mesh
        mesh = extract_mesh(vox_lr, threshold=opt.thres, n_face_simp=6000)
        mesh.export(os.path.join(test_path, name + '_lr.ply'), 'ply')
        # save the low-res output as voxel
        vox_lr = np.array(vox_lr + (1. - opt.thres), dtype=np.uint8)
        write_binvox_file(vox_lr, os.path.join(test_path, name + '_lr.binvox'), voxel_size=64)

        output_hr = np.zeros((256, 256, 256), dtype=np.float32)
        vox_hr_cat_feat_high, feat_low, pc, pc_occ = vipnet.predict_highres_input(img, b_pc_guide=True)
        vox_in = torch.zeros(opt.patchNum, vox_dim + feat_high_dim, 16, 16, 16).type(torch.FloatTensor).cuda()
        for i in range(0, opt.patchNum):  # here batch_size = 1
            vox_in[i] = vox_hr_cat_feat_high[0, :, begin_idx64[idx[i]][0]: begin_idx64[idx[i]][0] + 16,
                            begin_idx64[idx[i]][1]: begin_idx64[idx[i]][1] + 16,
                            begin_idx64[idx[i]][2]: begin_idx64[idx[i]][2] + 16]

        feat_low = feat_low.repeat(opt.patchNum, 1, 1, 1, 1)
        vox_in_cat = torch.cat([vox_in, feat_low], 1)

        for k in range(0, 2):
            _, _, _, vox_hr_sigmoid = network_patch_l2h(vox_in_cat[k*parchNum_half: k*parchNum_half+parchNum_half],
                                                         idx[k*parchNum_half: k*parchNum_half+parchNum_half], pc, pc_occ)

            vox_hr_sigmoid_np = np.array(vox_hr_sigmoid.cpu().data.squeeze().numpy()).reshape(
                (parchNum_half, 64, 64, 64))

            del vox_hr_sigmoid
            torch.cuda.empty_cache()

            for i in range(k*parchNum_half, k*parchNum_half+parchNum_half):
                output_hr[begin_idx256[i][0]: begin_idx256[i][0] + 64,
                begin_idx256[i][1]: begin_idx256[i][1] + 64,
                begin_idx256[i][2]: begin_idx256[i][2] + 64] = vox_hr_sigmoid_np[i-(k*parchNum_half), :, :, :]

        mesh = extract_mesh(output_hr, threshold=opt.thres, n_face_simp=18000)
        mesh.export(os.path.join(test_path, name + '_hr.ply'), 'ply')

        output_hr = np.array(output_hr + (1. - opt.thres), dtype=np.uint8)

        # save_image(img.squeeze(0).cpu(), os.path.join(test_path, name + '.png'))
        # write_binvox_file(output_pre, os.path.join(opt.test, name + '.binvox'), voxel_size=256)
        write_binvox_file(output_hr, os.path.join(test_path, name + '_hr.binvox'), voxel_size=256)

        print('processed', name)
    fin.close()
print('testing done!')
