from common import *

root_path = '/home/user/Downloads/quickdraw'
DP_DIR = os.path.join(root_path,'../input/shuffle-csvs/')
INPUT_DIR = '../input/quickdraw-doodle-recognition/'

np.random.seed(seed=1988)
tf.set_random_seed(seed=1988)

BASE_SIZE = 256
NCSVS = 100
NCATS = 340

STEPS = 800
EPOCHS = 16
size = 64
batchsize = 680
