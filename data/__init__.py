"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib
import os

import torch.utils.data

from .BaseDataset import BaseDataset
from .ParallelSentencesDataset import ParallelSentencesDataset


def find_dataset_using_name(dataset_type):
    """Import the module "data/[dataset_type]Dataset.py".

    In the file, the class called DatasetTypeDataset() will
    be instantiated. It has to be a subclass of BaseDataset (non nel mio),
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_type + "Dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_type = dataset_type.replace('_', '') + 'Dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_type.lower() and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a class with class name that matches %s in lowercase." % (dataset_filename, target_dataset_type))

    return dataset


def get_option_setter(dataset_type):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_type)
    return dataset_class.modify_commandline_options


def create_dataset(opt, model):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    assert opt.train_percentage+opt.eval_percentage+opt.test_percentage == 1.0

    cachedfiles = ["data_"]  # , "STS2017.en-de.txt.gz", "xnli-en-de.txt.gz"]
    cached_filepath_train = os.path.join(opt.dataroot, opt.name, cachedfiles[0] + 'train.bkp')
    cached_filepath_eval = os.path.join(opt.dataroot, opt.name, cachedfiles[0] + 'eval.bkp')
    #cached_filepath_test = os.path.join(opt.dataroot, cachedfiles[0] + 'test.bkp')

    if os.path.exists(cached_filepath_train):
        train_data_loader = torch.load(cached_filepath_train)
    else:
        train_data_loader = CustomDatasetDataLoader(opt, dataloader=None, dataset_type='train')
        os.mkdir(os.path.join(opt.dataroot, opt.name))
        torch.save(train_data_loader, cached_filepath_train)


    if os.path.exists(cached_filepath_eval):
        eval_data_loader = torch.load(cached_filepath_eval)
    else:
        eval_data_loader = CustomDatasetDataLoader(opt, dataloader=None, dataset_type='eval')
        torch.save(eval_data_loader, cached_filepath_eval)

    '''
    if os.path.exists(cached_filepath_test):
        test_data_loader = torch.load(cached_filepath_test)
    else:
        test_data_loader = CustomDatasetDataLoader(opt, dataloader=None, dataset_type='test')
        torch.save(test_data_loader, cached_filepath_test)
    '''

    return train_data_loader, eval_data_loader#, test_data_loader


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, dataloader=None, dataset_type='train'):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        self.train_perc = opt.train_percentage
        self.eval_perc = opt.eval_percentage
        self.test_perc = opt.test_percentage

        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt, self.train_perc, self.eval_perc, self.test_perc)
        self.dataset.load_data(dataset_type)

        print("dataset [%s] was created" % type(self.dataset).__name__)
        if dataloader is None:
            n_threads = int(opt.num_threads)
            shuffle = not opt.serial_batches
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=shuffle,
                num_workers=n_threads)

        else:
            self.dataloader = dataloader

    def load_data(self):
        return self


    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
