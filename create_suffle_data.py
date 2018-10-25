from common import *

def filename_to_category(filename):
    # filename.csv -> filename
    return filename.split('.')[0]


class Simplified():
    def __init__(self, input_path='./input'):
        self.input_path = input_path

    def list_all_categories(self):
        files = os.listdir(os.path.join(self.input_path, 'train_simplified'))
        return sorted([filename_to_category(f) for f in files], key=str.lower)

    def read_training_csv(self, category, nrows=None, usecols=None, drawing_transform=False):
        df = pd.read_csv(os.path.join(self.input_path, 'train_simplified', category + '.csv'),
                         nrows=nrows, parse_dates=['timestamp'], usecols=usecols)
        if drawing_transform:
            df['drawing'] = df['drawing'].apply(ast.literal_eval)
        return df

path = os.path.join(root_path, 'input')
simplified = Simplified(path)
number_csvs = 100
categories = simplified.list_all_categories()

# create folder to save suffle data
suffle_data = os.path.join(root_path, 'input', 'suffle_data')
os.makedirs(suffle_data, exist_ok=True)

for y, category in tqdm(enumerate(categories)):
    df = simplified.read_training_csv(category, nrows=30000)
    df['y'] = y
    df['cv'] = (df.key_id // 10 ** 7) % number_csvs
    for k in range(number_csvs):
        filename = os.path.join(suffle_data, 'train_k{}.csv'.format(k))
        chunk = df[df.cv == k]
        # drop column, but when you take mean by row, you must set axis=1
        chunk = chunk.drop(['key_id'], axis=1)

        if y == 0:
            chunk.to_csv(filename, index=False)
        else:
            chunk.to_csv(filename, mode='a', header=False, index=False)

for k in tqdm(range(number_csvs)):
    filename = os.path.join(suffle_data, 'train_k{}.csv'.format(k))
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        df['rnd'] = np.random.rand(len(df))
        df = df.sort_values(by='rnd').drop('rnd', axis=1)
        df.to_csv(filename + '.gz', compression='gzip', index=False)
        os.remove(filename)

print(df.shape)

end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))

