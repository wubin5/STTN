import pickle

def import_pkl(file_name):
    with open(file_name, 'rb') as f:
        file = pickle.load(f)
    return file

def export_pkl(file_name, file):
    with open(file_name, 'wb') as f:
        pickle.dump(file, f)
    return 0
