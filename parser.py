from common import *
from params import *
from utils import *

########################################################################
# Parameters Setting
########################################################################
def parser():
    parser = argparse.ArgumentParser(description='Hello there')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--fold', type=int, default=0, metavar='N',
                        help='Fold number (default: 0)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight-decay', type=float, default=0.0001, metavar='WD',
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=500, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model-name', type=str, default='se_resnext50_32x4d', metavar='M',
                        help='model name (default: se_resnext50_32x4d)')
    parser.add_argument('--dropout-p', type=float, default=0.5, metavar='D',
                        help='Dropout probability (default: 0.5)')
    parser.add_argument('--image-size', type=int, default=224, metavar='IS',
                        help='image size (default: 224)')
    parser.add_argument('--train-dir', type=str, default='./results', metavar='TD',
                        help='Where the training (fine-tuned) checkpoint and logs will be saved to.')
    parser.add_argument('--imbalance', type=int, default=0, metavar='IM',
                        help='Train with imbalance or not')
    parser.add_argument("--workers", type=int, default=8, help="Number of worker running parallel")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--freeze', type=int, default=6)
    parser.add_argument('--data-mode', type=str, default="all_image")

    args = parser.parse_args()

    print ('********************\n# Parameter settings\n********************')
    print(args)

    return args
