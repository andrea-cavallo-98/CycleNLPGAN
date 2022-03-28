import csv
from typing import List

from torch.utils.data import Dataset
import torch
import logging
import gzip
import os
import random

from tqdm import tqdm

from data import BaseDataset
import urllib.request
import pandas as pd



class MonolingualDataset(BaseDataset):
    """
    Mono-lingual dataset
    """

    def __init__(self, opt, path):
        BaseDataset.__init__(self, opt)
        self.path = path
        self.data = []


    def load_data(self, seed=25):

        filepath = self.path
        sentences = []

        with open(filepath, encoding='utf8') as fIn:

            logging.info("Load " + filepath)

            count = 0
            for line in fIn:
                sentences.append(line.split("\t")[1])
                count += 1
                
        self.data = sentences

        random.seed(seed)
        random.shuffle(self.data)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]





class ParallelDataset(BaseDataset):
    """
    Bi-lingual parallel dataset
    """

    def __init__(self, opt):
        """
        parameters:
            path1: path of the txt file containing one of the languages
            path2: path of the other language
        """

        BaseDataset.__init__(self, opt)
        self.path1 = opt.path1_bi
        self.path2 = opt.path2_bi
        self.data = []
        self.target_language = opt.target_lang


    def load_data(self, seed=25):

        df_en = pd.read_csv(self.path1, sep='\t', header=None, names=["id", "en"])
        df_target = pd.read_csv(self.path2, sep='\t', header=None, names=["id", self.target_language])
        df_en = df_en.set_index("id")
        df_target = df_target.set_index("id")
        df_en_target = df_en.join(df_target)
        df_en_target.dropna(inplace=True)
                
        self.data = df_en_target
        self.data = self.data.sample(frac=1, random_state=seed)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data["en"].iloc[idx], self.data[self.target_language].iloc[idx]







class ParallelSentencesDataset(BaseDataset):
    """
    This dataset reader can be used to read-in parallel sentences, i.e., it reads in a file with tab-seperated sentences with the same
    sentence in different languages. For example, the file can look like this (EN\tDE\tES):
    hello world     hallo welt  hola mundo
    second sentence zweiter satz    segunda oración

    The sentence in the first column will be mapped to a sentence embedding using the given the embedder. For example,
    embedder is a mono-lingual sentence embedding method for English. The sentences in the other languages will also be
    mapped to this English sentence embedding.

    When getting a sample from the dataset, we get one sentence with the according sentence embedding for this sentence.

    teacher_model can be any class that implement an encode function. The encode function gets a list of sentences and
    returns a list of sentence embeddings


    def __init__(self, student_model: SentenceTransformer, teacher_model):
        Parallel sentences dataset reader to train student model given a teacher model
        :param student_model: Student sentence embedding model that should be trained
        :param teacher_model: Teacher model, that provides the sentence embeddings for the first column in the dataset file

        self.student_model = student_model
        self.teacher_model = teacher_model
        self.datasets = []
        self.dataset_indices = []
        self.copy_dataset_indices = []
    """


    def __init__(self, opt, train_perc, eval_perc, test_perc):
        """
        Parallel sentences dataset reader to train student model given a teacher model
        :param opt: options used to create and read the dataset
        """
        BaseDataset.__init__(self, opt)
        self.filepaths_train = ["Tatoeba-en-{}-train.tsv.gz".format(opt.language), "WikiMatrix-en-{}-train.tsv.gz".format(opt.language), "TED2020-en-{}-train.tsv.gz".format(opt.language), "JW300-en-{}.tsv.gz".format(opt.language),]#, "STS2017.en-de.txt.gz", "xnli-en-de.txt.gz"]
        self.filepaths_eval = ["Tatoeba-en-{}-eval.tsv.gz".format(opt.language), "TED2020-en-{}-eval.tsv.gz".format(opt.language), "WikiMatrix-en-{}-eval.tsv.gz".format(opt.language),]#, "STS2017.en-de.txt.gz", "xnli-en-de.txt.gz"]
        #self.cachedfiles = ["ted2020_", "ted2020_", "ted2020_"]#, "STS2017.en-de.txt.gz", "xnli-en-de.txt.gz"]
        self.datasets = []
        self.dataset_indices = []
        self.copy_dataset_indices = []
        self.num_sentences = 0
        self.server = "https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/"

        self.datasets_iterator = []
        #self.train_perc = train_perc
        #self.eval_perc = eval_perc
        #self.test_perc = test_perc

        '''
        for dataset in self.filepaths:
            print("Download", dataset)
            url = self.server+dataset
            dataset_path = os.path.join(self.root, dataset)

            if not os.path.exists(dataset_path):
                urllib.request.urlretrieve(url, dataset_path)
        '''
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the sentences directory



    def load_data(self, dataset_type='train', seed=25):
        """
        Reads in a tab-seperated .txt/.csv/.tsv or .gz file. The different columns contain the different translations of the sentence in the first column

        :param filepath: Filepath to the file
        :param weight: If more that one dataset is loaded with load_data: With which frequency should data be sampled from this dataset?
        :param max_sentences: Max number of lines to be read from filepath
        :param max_sentence_length: Skip the example if one of the sentences is has more characters than max_sentence_length
        :return:
        """
        weight = self.opt.param_weight
        max_sentences = self.opt.max_sentences
        if max_sentences == 0:
            max_sentences = None
        max_sentence_length = self.opt.max_sentence_length

        if dataset_type == "train":
            filepaths = self.filepaths_train
        elif dataset_type == "eval":
            filepaths = self.filepaths_eval

        for filepath in filepaths:
            filepath = os.path.join(self.root, filepath)
            with gzip.open(filepath, 'rt', encoding='utf8') if filepath.endswith('.gz') else open(filepath,
                                                                                                  encoding='utf8') as fIn:

                logging.info("Load " + filepath)
                parallel_sentences = []

                count = 0
                for line in fIn:
                    sentences = line.strip().split("\t")
                    if max_sentence_length is not None and max_sentence_length > 0 and max(
                            [len(sent) for sent in sentences]) > max_sentence_length:
                        continue

                    parallel_sentences.append(sentences)
                    count += 1
                    if max_sentences is not None and max_sentences > 0 and count >= max_sentences:
                        break
            self.add_dataset(parallel_sentences, weight=weight, max_sentences=max_sentences,
                             max_sentence_length=max_sentence_length)


        random.seed(seed)
        random.shuffle(self.datasets)

        dataset_id = 0
        self.datasets = [self.datasets]
        self.dataset_indices.extend([dataset_id] * weight)

    def add_dataset(self, parallel_sentences: List[List[str]], weight: int = 100, max_sentences: int = None,
                    max_sentence_length: int = 128):

        data = []
        sentences_map = {}
        for sentences in parallel_sentences:
            if max_sentence_length is not None and max_sentence_length > 0 and max(
                    [len(sent) for sent in sentences]) > max_sentence_length:
                continue

            source_sentence = sentences[0]
            if source_sentence not in sentences_map:
                sentences_map[source_sentence] = set()

            for sent in sentences:
                if sent != source_sentence:
                    sentences_map[source_sentence].add(sent)
                    data.append([sent, source_sentence])

            if max_sentences is not None and max_sentences > 0 and len(sentences_map) >= max_sentences:
                break

        if len(sentences_map) == 0:
            return


        self.num_sentences += len(data)
        #self.num_sentences += sum([len(sentences_map[sent]) for sent in sentences_map])

        #dataset_id = len(self.datasets)
        self.datasets.extend(data)



    def __len__(self):
        return max([len(dataset) for dataset in self.datasets])

    def __getitem__(self, idx):
        if len(self.copy_dataset_indices) == 0:
            self.copy_dataset_indices = self.dataset_indices.copy()
            random.shuffle(self.copy_dataset_indices)

        dataset_idx = self.copy_dataset_indices.pop()

        A = self.datasets[dataset_idx][idx % len(self.datasets[dataset_idx])][0]

        B = self.datasets[dataset_idx][idx % len(self.datasets[dataset_idx])][1]



        return {'A': A, 'B': B}
