import sacrebleu

import argparse

from torch.utils.data import DataLoader

from models import EncDecModel
import os
import torch
from tqdm import tqdm
import numpy as np

batch_size = 32

parser = argparse.ArgumentParser(description='')
parser.add_argument('--language', required=True, help='Language compared to English')
parser.add_argument('--models_dir', type=str, default='../../checkpoints/translation',
                    help='Root directory of saved models')
parser.add_argument('--models_prefix', type=str, default='latest',
                    help='Prefix in model name')

args = parser.parse_args()

language = args.language
# Model we want to use for bitext mining. LaBSE achieves state-of-the-art performance
prefix = args.models_prefix
model_dir = args.models_dir
modelA_name = prefix + '_net_G_AB'
modelB_name = prefix + '_net_G_BA'
modelA = EncDecModel("Helsinki-NLP/opus-mt-de-en").to("cuda:0")
modelB = EncDecModel("Helsinki-NLP/opus-mt-en-de").to("cuda:0")

language = args.language
# Intput files for BUCC2018 shared task
source_file = "wmt14/" + language + "_en/newstest2014.src." + language
reference_file = "wmt14/" + language + "_en/newstest2014.ref.en"

# metric = load_metric("sacrebleu")

print("Read source file")
source_data = []
with open(source_file, encoding='utf8') as fIn:
    for line in fIn:
        source_data.append(line)

print("Read reference file")
reference_data = []
with open(reference_file, encoding='utf8') as fIn:
    for line in fIn:
        reference_data.append(line)

source_dataloader = DataLoader(
    source_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=1)

reference_dataloader = DataLoader(
    reference_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=1)

print("Translate source data")
translated_source_data = []

for i, source_batch in tqdm(enumerate(source_dataloader), total=len(source_dataloader)):
    # print(source_batch)

    model_prediction = modelA(source_batch)
    # print(model_prediction)
    translated_source_data.extend(model_prediction)

print(translated_source_data[:10])
print(reference_data[:10])
bleu = sacrebleu.raw_corpus_bleu(translated_source_data, [reference_data]).score

print("BLEU score: " + str(bleu))