from common import *
from params import *
from utils import *
from model import *

def main():
    # load model
    model = load_model()
    print(model.summary())
    
    
    # show process time
    end = dt.datetime.now()
    print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))


if __name__ == '__main__':
    main()


