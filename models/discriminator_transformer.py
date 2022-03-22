from abc import ABC

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import json
from typing import List, Dict, Optional
import os


class DiscriminatorTransformer(torch.nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, model_args: Dict = {}, cache_dir: Optional[str] = None, out_dim=2):
        #super(DiscriminatorTransformer, self).__init__(model_name_or_path)
        super(DiscriminatorTransformer, self).__init__()

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name_or_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2, return_dict=True)
        self.config = self.model.config
        #self.config_class = self.model.config_class
        self.config_keys = ['max_seq_length']
        self.dtype = self.model.dtype
        self.config_class = self.model.config_class
        self.max_seq_length = max_seq_length

    def forward(self, features, label):
        """Returns token_embeddings, cls_token"""
        embedding = self.tokenizer(features, return_tensors='pt', truncation=True,
                max_length=self.max_seq_length, padding='max_length').to(self.model.device)

        #labels = self.get_target_tensor(torch.zeros(embedding["input_ids"].size()[0]), label)
        labels = torch.tensor(label).to(self.model.device)

        features = self.model(**embedding, labels=labels)
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.auto_model.config.hidden_size


    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
            config = json.load(fIn)
        return DiscriminatorTransformer(model_name_or_path=input_path, **config)



    def get_target_tensor(self, prediction, label):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """


        target_tensor = torch.tensor(label).to(self.model.device)
        return target_tensor.expand_as(prediction)
