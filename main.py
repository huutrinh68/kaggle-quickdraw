from common import *
from params import *
from utils import *
from model import *
from parser import *
from slack_notify import SlackNotifier

# slack notify
notify = SlackNotifier()

def main():
    # setting hyper-parameters
    args = parser()
    
    # load model
    model = load_model()
    print(model.summary())
    
    # nofity training processing
    notify.sendNotifyMessage("Test", "This is test notify")

    # show process time
    end = dt.datetime.now()
    print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
    

if __name__ == '__main__':
    main()

