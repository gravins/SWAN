from .gpp_dataset import GPPDataset, TASKS
def get_dataset(root: str, task=None):

    assert task is None or task in TASKS

    data_train = GPPDataset(root, name=task, split='train') 
    data_valid = GPPDataset(root, name=task, split='val') 
    data_test = GPPDataset(root, name=task, split='test')
    num_features = data_train.num_features
    num_classes = data_train.num_classes

    return data_train, data_valid, data_test, num_features, num_classes


DATA = ['GraphProp']