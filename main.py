
from __future__ import print_function
import os
import gc
import argparse
from pickle import FALSE, TRUE, load
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import PcDataset
import numpy as np
from torch.utils.data import DataLoader
from model import DIT
import trimesh
from dataset import STDataset
from hybridDownsample import MyDataset
from public import publicDataset

def _init_(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/' + args.exp_name):
        os.makedirs('checkpoints/' + args.exp_name)
    # if not os.path.exists('models'):
    #     os.makedirs('models')
    os.makedirs(os.path.join('checkpoints', args.exp_name, 'models'), exist_ok=True)
    os.makedirs(os.path.join('checkpoints', args.exp_name, 'tensorboards'), exist_ok=True)



def train(args, net, train_loader, test_loader):
    #print("Use Adam")
    opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

    epoch_factor = args.epochs / 100.0

    scheduler = MultiStepLR(opt,milestones=[int(30*epoch_factor), int(60*epoch_factor), int(80*epoch_factor)], gamma=0.1)

    if args.load_model:
        print('load param from',args.model_path)
        net.load_state_dict(torch.load(args.model_path))
        net.eval()

    if args.eval == True:
        print('Testing begins! q(^_^)p ~~~')
    else:
        print('Training begins! q(^_^)p ~~~')

    for epoch in range(args.epochs):
        lr = scheduler.get_last_lr()
        print(f" Epoch: {epoch}, LR: {lr}")
        if args.eval == False:
            info_train = net._train_one_epoch(epoch=epoch, train_loader=train_loader, opt=opt, args=args)
            gc.collect()
        info_test = net._test_one_epoch(epoch=epoch, test_loader=test_loader)
        if args.eval == False:
            scheduler.step()
        if args.eval == False:
            #net.logger.write(info_train_best)
            torch.save(net.state_dict(), f'./checkpoints/{args.exp_name}/models/model{epoch}.pth')
        else:
            break
        gc.collect()
    net.close()
    if args.eval == True:
        print('Testing completed! /(^o^)/~~~')
    else:
        print('Training completed! /(^o^)/~~~')

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp3', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--train_path', type=str, default='./traindata/unfilteredtrain', metavar='N',
                        help='path for training set')
    parser.add_argument('--test_path', type=str, default='./traindata/unfilteredtest', metavar='N',
                        help='path for the testdata set')
    parser.add_argument('--n_emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--alpha', type=float, default=0.5, metavar='N',
                        help='control geo_scores and feature_scores')
    parser.add_argument('--n_blocks', type=int, default=12, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--n_iters', type=int, default=1, metavar='N',
                        help='Num of iters to run inference')
    parser.add_argument('--discount_factor', type=float, default=0.9, metavar='N',
                        help='Discount factor to compute the loss')
    parser.add_argument('--n_ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--temp_factor', type=float, default=100, metavar='N',
                        help='Factor to control the softmax precision')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--lr', type=float, default=0.00003, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=4327, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle_consistency_loss', type=float, default=0.1, metavar='N',
                        help='cycle consistency loss')
    parser.add_argument('--discrimination_loss', type=float, default=0.5, metavar='N',
                        help='discrimination loss')
    parser.add_argument('--rot_factor', type=float, default=4, metavar='N',
                        help='Divided factor of rotation')
    parser.add_argument('--curvature_sample_ratio', type=float, default=0.5, metavar='N',
                        help='partition of curve sample')
    parser.add_argument('--model_path', type=str, default='models/train_best.pth', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch when training)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch when testing)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--n_points', type=int, default=1000, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--num_sub_points', type=int, default=200, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--corres_mode', action='store_true', default=False,
                        help='decide whether use GMCCE or not')
    parser.add_argument('--GMCCE_Sharp', type=float, default=30, metavar='N',
                        help='The Sharp of GMCCE module')
    parser.add_argument('--GMCCE_Thres', type=float, default=0.6, metavar='N',
                        help='The Threshold of GMCCE module')                
    parser.add_argument('--load_model', action='store_true', default=False,
                        help='decide whether load model or not')    
    parser.add_argument('--token_dim', default=64, type=int, metavar='PCT',
                        help='the token dim')

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = False #关闭CuDNN库的自动优化功能
    torch.backends.cudnn.deterministic = True #使CuDNN库在确定性模式下运行,这样每次运行时结果都是一样的
    torch.backends.cudnn.enabled = True #启用CuDNN库
    torch.manual_seed(args.seed) #设置PyTorch的随机数生成器的种子
    torch.cuda.manual_seed_all(args.seed) #设置所有GPU的随机数生成器的种子
    np.random.seed(args.seed) #设置NumPy的随机数生成器的种子。

    _init_(args)

    train_loader = DataLoader(MyDataset(args.train_path, num_points=args.n_points, partition='train',
                                        rot_factor=args.rot_factor,curvature_sample_ratio=args.curvature_sample_ratio),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(MyDataset(args.test_path, num_points=args.n_points, partition='test',
                                       rot_factor=args.rot_factor,curvature_sample_ratio=args.curvature_sample_ratio),
        batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=2)

    # train_loader = DataLoader(PcDataset(args.train_path,num_points=args.n_points,num_subsampled_points=args.num_sub_points,partition='train',
    #                                            curvature_sample_ratio=args.curvature_sample_ratio,rot_factor=args.rot_factor),
    #                             batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
    # test_loader = DataLoader(PcDataset(args.test_path,num_points=args.n_points,num_subsampled_points=args.num_sub_points,partition='testdata',
    #                                      curvature_sample_ratio=args.curvature_sample_ratio, rot_factor=args.rot_factor),
    #                             batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=2)
    # train_loader = DataLoader(STDataset(args.train_path,num_points=args.n_points,curvature_sample_ratio=args.curvature_sample_ratio),
    #                           batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    # test_loader = DataLoader(STDataset(args.test_path,num_points=args.n_points,curvature_sample_ratio=args.curvature_sample_ratio),
    #                           batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=2)

    # train_loader = DataLoader(publicDataset(args.train_path,num_points=args.n_points,num_subsampled_points=args.num_sub_points,partition='train',
    #                                            gaussian_noise=True,rot_factor=args.rot_factor,down=True),
    #                             batch_size=args.batch_size, shuffle=False, drop_last=True, num_workers=2)
    # test_loader = DataLoader(publicDataset(args.test_path,num_points=args.n_points,num_subsampled_points=args.num_sub_points,partition='test',
    #                                       gaussian_noise=True, rot_factor=args.rot_factor,down=True),
    #                             batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=2)

    net = DIT(args).cuda()
    train(args, net, train_loader, test_loader)
    print('FINISH')



if __name__ == '__main__':
    main()
