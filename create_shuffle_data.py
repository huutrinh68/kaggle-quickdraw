from common import *
from params import *

""" 
these functions be used to generate shuffled data for training and validating.
"""

class Simplified():
    def __init__(self, input_path=os.path.join(root_path,'input')):
        self.input_path = input_path

    # get category name from file name
    def filename_to_category(self, filename):
        return filename.split('.')[0]

    # get all categories name
    def list_all_categories(self):
        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        return sorted([self.filename_to_category(f) for f in files], key=str.lower)

    # read csv file into dataframe
    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(ast.literal_eval)
        return df

def main():
    path = os.path.join(root_path, 'input')
    simplified = Simplified(path)
    number_csvs = 100
    categories = simplified.list_all_categories()

    # create folder to save suffle data
    os.makedirs(SHUFFLE_DATA, exist_ok=True)

    for y, category in tqdm(enumerate(categories)):
        df = simplified.read_training_csv(category, nrows=30000)
        df['y'] = y
        df['cv'] = (df.key_id // 10 ** 7) % number_csvs
        for k in range(number_csvs):
            filename = os.path.join(SHUFFLE_DATA, 'train_k{}.csv'.format(k))
            chunk = df[df.cv == k]
            # drop column, but when you take mean by row, you must set axis=1
            chunk = chunk.drop(['key_id'], axis=1)

            if y == 0:
                chunk.to_csv(filename, index=False)
            else:
                chunk.to_csv(filename, mode='a', header=False, index=False)

    for k in tqdm(range(number_csvs)):
        filename = os.path.join(SHUFFLE_DATA, 'train_k{}.csv'.format(k))
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['rnd'] = np.random.rand(len(df))
            df = df.sort_values(by='rnd').drop('rnd', axis=1)
            df.to_csv(filename + '.gz', compression='gzip', index=False)
            os.remove(filename)
            print(df.shape)

    end = dt.datetime.now()
    print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))


if __name__ == '__main__':
    main()
