from common import *

SHUFFLE_DATA = os.path.join(root_path,'input/shuffle_data/')
INPUT_DIR = os.path.join(root_path,'input/quickdraw-doodle-recognition/')

np.random.seed(seed=1988)
tf.set_random_seed(seed=1988)

BASE_SIZE = 256
NCSVS = 100
NCATS = 340

# size = 64
size = 197
batchsize = 680
