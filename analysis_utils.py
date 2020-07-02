import pickle


def get_pickle(file_path):
    with open(file_path, 'rb') as fp:
        x = pickle.load(fp)
    return x
