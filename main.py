from common import *
from params import *
from utils import image_generator_xd
from model import *
from parser import *
from slack_notify import SlackNotifier
from create_shuffle_data import Simplified

# slack notify
notify = SlackNotifier()
# nofity training processing
# notify.sendNotifyMessage("Title", "Message")

# show generated image
def show_img(train_datagen):
    x, y = next(train_datagen)
    n = 8
    fig, axs = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(12, 12))
    for i in range(n**2):
        ax = axs[i // n, i % n]
        (-x[i]+1)/2
        ax.imshow((-x[i, :, :, 0] + 1)/2, cmap=plt.cm.gray)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig('gs.png', dpi=300)
    plt.show()

# show result
def show_result(hist):
    hist_df = pd.concat([pd.DataFrame(hist.history) for hist in hists], sort=True)
    hist_df.index = np.arange(1, len(hist_df)+1)
    fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(16, 10))
    axs[0].plot(hist_df.val_categorical_accuracy, lw=5, label='Validation Accuracy')
    axs[0].plot(hist_df.categorical_accuracy, lw=5, label='Training Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].grid()
    axs[0].legend(loc=0)
    axs[1].plot(hist_df.val_categorical_crossentropy, lw=5, label='Validation MLogLoss')
    axs[1].plot(hist_df.categorical_crossentropy, lw=5, label='Training MLogLoss')
    axs[1].set_ylabel('MLogLoss')
    axs[1].set_xlabel('Epoch')
    axs[1].grid()
    axs[1].legend(loc=0)
    fig.savefig('hist.png', dpi=300)
    plt.show()

def main():
    # setting hyper-parameters
    args = parser()
    
    # create validate data, training data
    valid_df = pd.read_csv(os.path.join(SHUFFLE_DATA, 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=34000)
    x_valid = df_to_image_array_xd(valid_df, size)
    y_valid = keras.utils.to_categorical(valid_df.y, num_classes=NCATS)
    train_datagen = image_generator_xd(size=size, batchsize=batchsize, ks=range(NCSVS - 1))

    # show generated img
    #show_img(train_datagen)

    # define callback
    callbacks = [
    ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=5,
                      min_delta=0.005, mode='max', cooldown=3, verbose=1)
    ]

    # load model
    model = load_model()
    #print(model.summary())

    # train model
    hists = []
    hist = model.fit_generator(
        train_datagen, steps_per_epoch=STEPS, epochs=EPOCHS, verbose=1,
        validation_data=(x_valid, y_valid),
        callbacks = callbacks
    )
    # log train history
    hists.append(hist)

    # show result
    #show_result(hist)

    # validation
    valid_predictions = model.predict(x_valid, batch_size=128, verbose=1)
    map3 = mapk(valid_df[['y']].values, preds2catids(valid_predictions).values)

    # create submission
    test = pd.read_csv(os.path.join(INPUT_DIR, 'test_simplified.csv'))
    test.head()
    x_test = df_to_image_array_xd(test, size)
    test_predictions = model.predict(x_test, batch_size=128, verbose=1)
    top3 = preds2catids(test_predictions)
    cats = Simplified().list_all_categories()
    id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(cats)}
    top3cats = top3.replace(id2cat)
    test['word'] = top3cats['a'] + ' ' + top3cats['b'] + ' ' + top3cats['c']
    submission = test[['key_id', 'word']]
    submission.to_csv('gs_mn_submission_{}.csv'.format(int(map3 * 10**4)), index=False)

    # show process time
    end = dt.datetime.now()
    print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))
    
if __name__ == '__main__':
    main()

