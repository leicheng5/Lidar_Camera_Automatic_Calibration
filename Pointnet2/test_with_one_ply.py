"""
Author: Lei Cheng
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from plyfile import PlyData

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def ReadPly(filename, has_normal: bool = True, has_color: bool = False,
               has_label: bool = False):
    plydata = PlyData.read(filename)
    vtx = plydata['vertex']

    output = dict()
    points = np.stack([vtx['x'], vtx['y'], vtx['z']], axis=1)
    output['points'] = points.astype(np.float32)
    if has_normal:
      normal = np.stack([vtx['nx'], vtx['ny'], vtx['nz']], axis=1)
      output['normals'] = normal.astype(np.float32)
    if has_color:
      color = np.stack([vtx['red'], vtx['green'], vtx['blue']], axis=1)
      output['colors'] = color.astype(np.float32)
    if has_label:
      label = vtx['label']
      output['labels'] = label.astype(np.int32)
    return output

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

# def MyDataLoader(root='/xdisk/caos/leicheng/lidar_camera_calibration/octformer/data/ModelNet40', 
#                  file_path='filelist/m40_test.txt', onlyXYZ= True):
#     catfile = os.path.join(root, 'filelist/modelnet40_shape_names.txt')
#     cat = [line.rstrip() for line in open(catfile)]
#     classes = dict(zip(cat, range(len(cat))))
    
#     datapath = np.loadtxt(os.path.join(root, file_path), dtype=str, delimiter=' ')
#     # Extract the categories, paths, and category numbers
#     categories = np.array([path.split('\\')[0] for path in datapath[:, 0]])
#     paths = datapath[:, 0]
#     category_nums = datapath[:, 1].astype(np.int32)
#     for index in tqdm(range(len(datapath)), total=len(datapath)):
#         filename = datapath[index]
#         filename = os.path.join(root, filename)
#         cls = datapath[index][1]
#         label = np.array([cls]).astype(np.int32)
#         if filename.endswith('.ply'):
#           read_ply = ReadPly(filename,has_normal=False)
#           point_set =  read_ply(filename)
#         point_set = point_set['points']          
#         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
#         if onlyXYZ:
#             point_set = point_set[:, 0:3]

#     return point_set, label[0]

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default='pointnet2_cls_msg',  help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    return parser.parse_args()


def test(model, loader, num_class=40, vote_num=1, 
         root='/xdisk/caos/leicheng/lidar_camera_calibration/octformer/data/ModelNet40', 
         file_path='filelist/m40_test.txt', onlyXYZ= True):
    ######### Data Preparation  ##############################
    catfile = os.path.join(root, 'filelist/modelnet40_shape_names.txt')
    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))
    datapath = np.loadtxt(os.path.join(root, file_path), dtype=str, delimiter=' ')
    # Extract the categories, paths, and category numbers
    categories = np.array([path.split('\\')[0] for path in datapath[:, 0]])
    paths = datapath[:, 0]
    category_nums = datapath[:, 1].astype(np.int32)
    ######### Test  ##############################
    mean_correct = []
    classifier = model.eval()
    class_acc = np.zeros((num_class, 3))

    #for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
    for index in tqdm(range(len(datapath)), total=len(datapath)):
        ##### Load Data
        filename = datapath[index][0]
        filename = os.path.join(root, "ModelNet40.ply.normalize", filename)
        filename = filename.replace('\\', '/')
        cls = datapath[index][1]
        target = np.array([cls]).astype(np.int32)  #label
        if filename.endswith('.ply'):
          points = ReadPly(filename, has_normal=False)
        points = points['points']         
        #points[:, 0:3] = pc_normalize(points[:, 0:3])
        if onlyXYZ:
            points = points[:, 0:3]
        ###################################    
        points = np.expand_dims(points, axis=0)
        points = points.transpose(0, 2, 1)
        points = torch.from_numpy(points).to('cuda')
        target = torch.from_numpy(target).to('cuda')
        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()
        vote_pool = torch.zeros(target.size()[0], num_class).cuda()

        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num
        pred_choice = pred.data.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1
        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    return instance_acc, class_acc


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    data_path = 'data/modelnet40_normal_resampled/'

    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=False)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, vote_num=args.num_votes, num_class=num_class)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
