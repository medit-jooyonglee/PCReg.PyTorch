import argparse
import json
import platform
import numpy as np
import open3d
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data import CustomData
from models import IterativeBenchmark, IterativeSimilarityBenchmark
from metrics import compute_metrics, summary_metrics, print_train_info
from utils import time_calc
from trainer import torch_utils, vtk_utils, get_logger, time_strftime
# from pc

from pcregmodel.data.cococustom import CocoDetection

visualize = False

def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    ## dataset
    parser.add_argument('--root', help='the data path,',
                        default='D:/workspace/datasets/ModelNet40')
    parser.add_argument('--train_npts', type=int,
                        default=1024,
                        help='the points number of each pc for training')
    ## models training
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--gn', action='store_true',
                        help='whether to use group normalization')
    parser.add_argument('--epoches', type=int, default=400)
    parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--in_dim', type=int, default=6,
                        help='3 for (x, y, z) or 6 for (x, y, z, nx, ny, nz)')
    parser.add_argument('--transform_head', type=int, default=8,
                        help='transform head. for initial-identical-transform-matrix, if negative, random-transform-amtrix')
    parser.add_argument('--niters', type=int, default=40,
                        help='iteration nums in one model forward')
    parser.add_argument('--registration_mode', default='similarity',
                        choices=['rigid', 'similarity'])
    parser.add_argument('--backbone', default='pointnet',
                        choices=['pointnet', 'pointnet2'])
    parser.add_argument('--min_scale', type=float, default=0.9)
    parser.add_argument('--max_scale', type=float, default=1.1)
    parser.add_argument('--max_log_scale', type=float, default=0.35)
    parser.add_argument('--sample_count1', type=int, default=256)
    parser.add_argument('--sample_count2', type=int, default=64)
    parser.add_argument('--resume', type=str, 
                        default=''
                        # default='work_dirs/models_pointnet/checkpoints/test_min_loss.pth'
                        )
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--milestones', type=list, default=[50, 250],
                        help='lr decays when epoch in milstones')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='lr decays to gamma * lr every decay epoch')
    # logs
    parser.add_argument('--saved_path', default='work_dirs/models',
                        help='the path to save training logs and checkpoints')
    parser.add_argument('--saved_frequency', type=int, default=10,
                        help='the frequency to save the logs and checkpoints')
    args = parser.parse_args()
    return args


normal_loss_weight = 1.0


def compute_loss(moving_target, pred_ref_clouds, loss_fn, normal_loss_fn=None):
    losses = []
    discount_factor = 0.5
    prediction_count = len(pred_ref_clouds)
    has_normals = normal_loss_fn is not None and moving_target.shape[-1] >= 6
    for i in range(prediction_count):
        loss = loss_fn(moving_target[..., :3].contiguous(),
                       pred_ref_clouds[i][..., :3].contiguous())
        if has_normals:
            loss = loss + normal_loss_weight * normal_loss_fn(
                pred_ref_clouds[i][..., 3:6].contiguous(),
                moving_target[..., 3:6].contiguous(),
            )
        losses.append(discount_factor**(prediction_count - i)*loss)
    return torch.sum(torch.stack(losses))



def forward_registration(model, batch, loss_fn, normal_loss_fn=None):
    ref_cloud, src_cloud, moving_target, gtR, gtt = \
        [value.cuda() for value in batch[:5]]
    output = model(src_cloud.permute(0, 2, 1).contiguous(),
                   ref_cloud.permute(0, 2, 1).contiguous())
    if len(output) == 3:
        rotation, translation, predictions = output
        scale = None
        scale_loss = ref_cloud.new_tensor(0.0)
        scale_mae = None
    else:
        rotation, translation, scale, predictions = output
        gt_scale = batch[5].cuda().reshape(-1)
        scale_loss = nn.functional.smooth_l1_loss(
            torch.log(scale), torch.log(gt_scale)
        )
        scale_mae = torch.abs(scale - gt_scale).mean().item()
        
    if visualize:
        rr, ss, pp = torch_utils.to_numpy([ref_cloud, src_cloud, predictions], squeeze=True)
        colors = vtk_utils.get_rainbow_color_table(len(pp))
        pp_actor = [vtk_utils.create_points_actor(p0, invert=False ) for p0 in pp]
        
        vtk_utils.change_actor_color(pp_actor, colors)
        vtk_utils.split_show([
            # ss, 
            vtk_utils.create_points_actor(rr, invert=False, point_size=3, color=(0, 1, 0)), 
            vtk_utils.create_points_actor(ss, invert=False, point_size=3, color=(1, 1, 0)), 
            ], 
                             [
            vtk_utils.create_points_actor(rr, invert=False, point_size=3, color=(0, 1, 0)), 
            pp_actor],
            [
            vtk_utils.create_points_actor(pp[-1], invert=False, point_size=3, color=(1, 1, 0)), 
            vtk_utils.create_points_actor(rr, invert=False, point_size=3, color=(0, 1, 0)), 
            # rr
            ])
    dist_loss_weight = 10
    loss = compute_loss(moving_target, predictions, loss_fn, normal_loss_fn) * dist_loss_weight + scale_loss
    return loss, rotation, translation, scale_mae


@time_calc
def train_one_epoch(train_loader, model, loss_fn, optimizer, normal_loss_fn=None):
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    scale_maes = []
    train_loader = iter(train_loader)
    for batch in tqdm(train_loader):
        # try:
        #     batch = next(train_loader)
        # except Exception as e:
        #     print(f"Error occurred while fetching batch {i}: {e}")
        #     continue
            
        gtR, gtt = batch[3].cuda(), batch[4].cuda()
        optimizer.zero_grad()
        loss, R, t, scale_mae = forward_registration(model, batch, loss_fn, normal_loss_fn)
        loss.backward()
        optimizer.step()

        cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
        cur_t_isotropic = compute_metrics(R, t, gtR, gtt)
        losses.append(loss.item())
        r_mse.append(cur_r_mse)
        r_mae.append(cur_r_mae)
        t_mse.append(cur_t_mse)
        t_mae.append(cur_t_mae)
        r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
        t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
        if scale_mae is not None:
            scale_maes.append(scale_mae)
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    if scale_maes:
        results['scale_mae'] = np.mean(scale_maes)
    return results


@time_calc
def test_one_epoch(test_loader, model, loss_fn, normal_loss_fn=None):
    model.eval()
    losses = []
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = [], [], [], [], [], []
    # test_loader = iter(test_loader)
    with torch.no_grad():
        scale_maes = []
        for batch in tqdm(test_loader):

            gtR, gtt = batch[3].cuda(), batch[4].cuda()
            loss, R, t, scale_mae = forward_registration(model, batch, loss_fn, normal_loss_fn)
            cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
            cur_t_isotropic = compute_metrics(R, t, gtR, gtt)

            losses.append(loss.item())
            r_mse.append(cur_r_mse)
            r_mae.append(cur_r_mae)
            t_mse.append(cur_t_mse)
            t_mae.append(cur_t_mae)
            r_isotropic.append(cur_r_isotropic.cpu().detach().numpy())
            t_isotropic.append(cur_t_isotropic.cpu().detach().numpy())
            if scale_mae is not None:
                scale_maes.append(scale_mae)
    model.train()
    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic)
    results = {
        'loss': np.mean(losses),
        'r_mse': r_mse,
        'r_mae': r_mae,
        't_mse': t_mse,
        't_mae': t_mae,
        'r_isotropic': r_isotropic,
        't_isotropic': t_isotropic
    }
    if scale_maes:
        results['scale_mae'] = np.mean(scale_maes)
    return results


def save_args(args, saved_path):
    args_dict = vars(args)
    args_json_path = os.path.join(saved_path, f'args_{time_strftime()}.json')
    with open(args_json_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
        
        
def main():
    args = config_params()
    logger = get_logger()
    print(args)

    setup_seed(args.seed)
    if not os.path.exists(args.saved_path):
        os.makedirs(args.saved_path)
        
    save_args(args, args.saved_path)
    
    summary_path = os.path.join(args.saved_path, 'summary')
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)
    checkpoints_path = os.path.join(args.saved_path, 'checkpoints')
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    estimate_scale = args.registration_mode == 'similarity'
    if platform.system() == 'Linux':
        dataset_path = '/data1/jooyonglee/reverse_tomo/xray_panoramic/cbct_ios_dcm/'
    else:
        dataset_path = 'E:/dataset/reverse_tomosynthesis/kaggle_xrays/cbct_ios_dcm'
    dataset_kwargs = {
        'estimate_scale': estimate_scale,
        'min_scale': args.min_scale,
        'max_scale': args.max_scale,
        'img_folder': dataset_path,
        'ann_file': os.path.join(dataset_path, 'annotations.json'),
        'in_dim': args.in_dim,
    }
    # InnDataset = CustomData
    InnDataset = CocoDetection
    train_set = InnDataset(root=args.root, npts=args.train_npts, train=True, **dataset_kwargs)
    test_set = InnDataset(root=args.root, npts=args.train_npts, train=False, **dataset_kwargs)
    train_loader = DataLoader(train_set, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batchsize, shuffle=False,
                             num_workers=args.num_workers)
    backbone = args.backbone or ('pointnet2' if estimate_scale else 'pointnet')
    model_kwargs = {
        'in_dim': args.in_dim,
        'niters': args.niters,
        'gn': args.gn,
        'backbone': backbone,
        'sample_count1': args.sample_count1,
        'sample_count2': args.sample_count2,
        'transform_head': args.transform_head
    }
    if estimate_scale:
        model_kwargs['max_log_scale'] = args.max_log_scale
        model = IterativeSimilarityBenchmark(**model_kwargs)
    else:
        model = IterativeBenchmark(**model_kwargs)
    model = model.cuda()
    # moving_points is generated by transforming a copy of target_points, so
    # src/ref points share the same order and count: point-wise MSE is exact
    # here, unlike EMD/Chamfer which solve the harder unordered-set case.
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.cuda()
    normal_loss_fn = PairwiseSmoothL1Loss(dim=3).cuda() if args.in_dim >= 6 else None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=args.milestones,
                                                     gamma=args.gamma,
                                                     last_epoch=-1)

    writer = SummaryWriter(summary_path)
    
    if args.resume and os.path.exists(args.resume):
        resume = args.resume
        # from trainer import utils as trainer_utils
        state_dict = torch.load(resume, map_location='cpu')
        try:
            res = model.load_state_dict(state_dict, strict=False)
            print(f'loading complete: {resume} / {res}')
        except RuntimeError as e:
            logger.error('----------------------------------------------------------------')
            logger.error(f'Error loading state_dict from {resume}: {e}')
            logger.error('----------------------------------------------------------------')
            # print('Attempting to load with strict=False...')
            # res = model.load_state_dict(state_dict, strict=False)
            # print(f'loading complete: {resume} / {res}')
        # print(res)
    
    
    test_min_loss, test_min_r_mse_error, test_min_rot_error = \
        float('inf'), float('inf'), float('inf')
    for epoch in range(args.epoches):
        print('=' * 20, epoch + 1, '=' * 20)
        train_results = train_one_epoch(train_loader, model, loss_fn, optimizer, normal_loss_fn)
        print_train_info(train_results)
        test_results = test_one_epoch(test_loader, model, loss_fn, normal_loss_fn)
        print_train_info(test_results)

        if epoch % args.saved_frequency == 0:
            writer.add_scalar('Loss/train', train_results['loss'], epoch + 1)
            writer.add_scalar('Loss/test', test_results['loss'], epoch + 1)
            writer.add_scalar('RError/train', train_results['r_mse'], epoch + 1)
            writer.add_scalar('RError/test', test_results['r_mse'], epoch + 1)
            writer.add_scalar('rotError/train', train_results['r_isotropic'],
                              epoch + 1)
            writer.add_scalar('rotError/test', test_results['r_isotropic'],
                              epoch + 1)
            writer.add_scalar('Lr', optimizer.param_groups[0]['lr'], epoch + 1)
        test_loss, test_r_error, test_rot_error = \
            test_results['loss'], test_results['r_mse'], test_results[
                'r_isotropic']
        if test_loss < test_min_loss:
            saved_path = os.path.join(checkpoints_path, "test_min_loss.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_loss = test_loss
        if test_r_error < test_min_r_mse_error:
            saved_path = os.path.join(checkpoints_path,
                                      "test_min_rmse_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_r_mse_error = test_r_error
        if test_rot_error < test_min_rot_error:
            saved_path = os.path.join(checkpoints_path,
                                      "test_min_rot_error.pth")
            torch.save(model.state_dict(), saved_path)
            test_min_rot_error = test_rot_error
        scheduler.step()





def compute_scores_tensors_pair(tensor1, tensor2, beta=0.1, method='smoothl1'):
    """

    Parameters
    ----------
    tensor1 : (B, N, 6)
    tensor2 : (B, N, 6)

    Returns
    -------
        float scalars average

    """
    assert tensor1.shape[-1] == tensor2.shape[-1] == 6
    # torch.cdist(torch.randn(10, 3), torch.randn(20, 3), )
    # dist = cdist(tensor1[:, :3], tensor2[:, :3])
    pts1, normals1 = tensor1[..., :3], tensor1[..., 3:]
    pts2, normals2 = tensor2[..., :3], tensor2[..., 3:]
    # dist = torch.cdist(pts1, pts2, p=2.0)
    if method == 'smoothl1':
        diff = torch.abs(pts1 - pts2)
        dist = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    elif method == 'l2':
        dist = torch.sum((pts1 - pts2) ** 2, dim=-1)

    # diff = torch.abs(pts1 - pts2)
    # dist = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)

    innerdot = 1 - torch.sum(normals1 * normals2, dim=-1)

    return dist.mean() + innerdot.mean()

# def smoot
class PairwiseSmoothL1Loss(nn.Module):
    def __init__(self, **kwargs):
        super(PairwiseSmoothL1Loss, self).__init__()

        self.beta = kwargs.get('beta', 0.1)
        self.dim = kwargs.get('dim', 3)
        self.pair_metric = kwargs.get('method', 'smoothl1')


    def forward(self, predicted, target):
        """
        Args:
            predicted: (B, N, 3)
            target: (B, N, 3)
        Returns:
            loss: Scalar Smooth L1 loss
        """
        predicted = predicted[..., :self.dim]
        target = target[..., :self.dim]
        if self.pair_metric == 'smoothl1':
            assert predicted.shape == target.shape, "Predicted and target must have the same shape"
            diff = torch.abs(predicted - target)
            loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
            loss = torch.mean(loss)
        elif self.pair_metric == 'innerdot_smoothl1':
            loss = compute_scores_tensors_pair(predicted, target, beta=self.beta, method='smoothl1')
        elif self.pair_metric == 'innerdot_l2':
            loss = compute_scores_tensors_pair(predicted, target, beta=self.beta, method='l2')
        else:
            raise NotImplementedError

        return loss
    

if __name__ == '__main__':
    device = 'cuda:4'
    if torch.cuda.device_count() > 1:
        # torch.cuda.set_device(device)
        torch.cuda.set_device('cuda:4')
    elif torch.cuda.device_count() == 1:
        torch.cuda.set_device('cuda:0')
    main()
