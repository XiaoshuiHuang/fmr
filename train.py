"""
train the feature-metric registration algorithm
Creator: Xiaoshui Huang
Date: 2020-06-19
"""
import model
from data import dataset
import torch
import os
import argparse
import logging

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def parameters(argv=None):
    parser = argparse.ArgumentParser(description='Feature-metric registration')

    # required to check
    parser.add_argument('-data', '--dataset-type', default='7scene', choices=['modelnet', '7scene'],
                        metavar='DATASET', help='dataset type (default: 7scene)')
    parser.add_argument('-o', '--outfile', default='./result/fmr', type=str,
                        metavar='BASENAME', help='output filename (prefix)')  # the result: ${BASENAME}_model_best.pth
    parser.add_argument('--store', default='./result/fmr_model.pth', type=str, metavar='PATH',
                        help='path to the trained model')
    parser.add_argument('--train-type', default=0, type=int,
                        metavar='type', help='unsupervised (0) or semi-supervised (1) training (default: 0)')

    # settings for performance adjust
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--num-points', default=2048, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--mag', default=0.8, type=float,
                        metavar='T', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')

    # settings for on training
    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--max-iter', default=10, type=int,
                        metavar='N', help='max-iter on IC algorithm. (default: 10)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('-l', '--logfile', default='./result/fmr_training.log', type=str,
                        metavar='LOGNAME', help='path to logfile (default: fmr_training.log)')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--start-epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='./result/fmr_model_7scene.pth', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('-i', '--dataset-path', default='', type=str,
                        metavar='PATH', help='path to the input dataset')  # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', default='./data/categories/modelnet40_half1.txt', type=str,
                        metavar='PATH',
                        help='path to the categories to be trained')  # eg. './sampledata/modelnet40_half1.txt'
    parser.add_argument('--mode', default='train', help='program mode. This code is for training')
    parser.add_argument('--uniformsampling', default=False, type=bool, help='uniform sampling points from the mesh')
    args = parser.parse_args(argv)
    return args


def main(args):
    # dataset
    trainset, testset = dataset.get_datasets(args)

    # training
    fmr = model.FMRTrain(dim_k=args.dim_k, num_points=args.num_points, train_type=args.train_type)
    run(args, trainset, testset, fmr)


def run(args, trainset, testset, action):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    model = action.create_model()
    if args.store and os.path.isfile(args.store):
        model.load_state_dict(torch.load(args.store, map_location='cpu'))

    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # optimizer
    min_loss = float('inf')
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    # training
    LOGGER.debug('train, begin')
    for epoch in range(args.start_epoch, args.epochs):
        running_loss = action.train(model, trainloader, optimizer, args.device)
        val_loss = action.validate(model, testloader, args.device)

        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        LOGGER.info('epoch, %04d, %f, %f', epoch + 1, running_loss, val_loss)
        print('epoch, %04d, floss_train=%f, floss_val=%f' % (epoch + 1, running_loss, val_loss))

        if is_best:
            save_checkpoint(model.state_dict(), args.outfile, 'model')

    LOGGER.debug('train, end')


def save_checkpoint(state, filename, suffix):
    torch.save(state, '{}_{}.pth'.format(filename, suffix))


if __name__ == '__main__':
    ARGS = parameters()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)

    main(ARGS)
    LOGGER.debug('done (PID=%d)', os.getpid())
