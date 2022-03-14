import torch
from torch import nn, Tensor
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer, AutoConfig, MarianTokenizer, MarianMTModel, AutoModelForSeq2SeqLM, MBart50TokenizerFast
import transformers
import json
from typing import List, Dict, Optional, Tuple, Union
import os
import numpy as np
import logging
from models.Pooling import Pooling
from transformers.tokenization_utils import BatchEncoding



class EncDecModel(nn.Module):
    """Huggingface AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.
    """
    def __init__(self, model_name_or_path: str, max_seq_length: int = 128, task="translation", 
                model_args: Dict = {}, cache_dir: Optional[str] = None, freeze_encoder=False, 
                source_lang = "en", target_lang = "vi"):
        super(EncDecModel, self).__init__()
        self.config_keys = ['max_seq_length']
        self.max_seq_length = max_seq_length
        self.source_lang = source_lang
        self.target_lang = target_lang

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)

        self.tokenizer_en = MarianTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.tokenizer_target = MarianTokenizer.from_pretrained("CLAck/en-vi", cache_dir=cache_dir)

        self.config = self.model.config
        self.config_class = self.model.config_class
        #self.device = self.model.device
        self.dtype = self.model.dtype
        self.task = task

        self.output_attentions = True
        #self.output_hidden_states = True
        self.config.output_attentions = True
        #self.config.output_hidden_states = True

        self.freeze_encoder = freeze_encoder

        self.add_pooling_layer()

    def forward(self, sentences, target_sentences=None, partial_value=False, generate_sentences=True):

        if self.source_lang == "en":
            embeddings = self.tokenizer_en(sentences, padding='max_length', max_length=self.max_seq_length, truncation=True, return_tensors='pt')
        else:
            embeddings = self.tokenizer_target(sentences, padding='max_length', max_length=self.max_seq_length, truncation=True, return_tensors='pt')
        
        embeddings = embeddings.to(self.model.device)
        pooling_attention_mask = embeddings.attention_mask
        
        outputs = self.model(**embeddings, return_dict=True, labels = embeddings.input_ids)
        
        if generate_sentences:
            output_sentences = self.model.generate(**embeddings)
            output_sentences = self.decode(output_sentences)
        else:
            output_sentences = []
       

        if partial_value:

            sentence_embedding = torch.zeros([len(sentences), self.get_word_embedding_dimension()], dtype=torch.float32).to(self.model.device)
            for i in range(len(sentences)):
                params = dict()
                params["token_embeddings"] = outputs.encoder_last_hidden_state[i]
                params["attention_mask"] = pooling_attention_mask[i]
                sentence_embedding[i] = self.embedding_pooling(params)["sentence_embedding"]

            if target_sentences is not None:
                return output_sentences, sentence_embedding, outputs.loss
            else:
                return output_sentences, sentence_embedding, 0.0
        else:
            del embeddings

            if target_sentences is not None:
                return output_sentences, outputs.loss
            else:
                return output_sentences

    def get_word_embedding_dimension(self) -> int:
        return self.config.hidden_size

    def get_sentence_embedding_dimension(self) -> int:
        return self.config.hidden_size

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path: str):
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        #with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w') as fOut:
        #    json.dump(self.get_config_dict(), fOut, indent=2)


    @staticmethod
    def load(input_path: str, task, freeze_encoder):
        #with open(os.path.join(input_path, 'sentence_bert_config.json')) as fIn:
        #    config = json.load(fIn)

        return EncDecModel(model_name_or_path=input_path, task=task, freeze_encoder=freeze_encoder) #, **config)


    def generate(self, text):
        if self.source_lang == "en":
            encod = self.tokenizer_en(text, return_tensors="pt", padding=True).to(self.model.device)
        else:
            encod = self.tokenizer_target(text, return_tensors="pt", padding=True).to(self.model.device)
        
        output = self.model.generate(**encod)
        return output


    def get_encoder(self):
        return self.model.get_encoder()


    def decode(self, tokens): 
        list_sentences = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        list_sentences = [''.join(l).replace('▁', ' ') for l in list_sentences]
        return list_sentences
        #return self.dest_tokenizer.batch_decode(tokens, skip_special_tokens=True)


    def train(self, mode=True):
        self.training = mode
        self.model.training = mode
        self.model.base_model.training = mode
        self.model.base_model.encoder.training = mode
        self.model.base_model.decoder.training = mode
        self.model.train(mode)
        self.model.base_model.train(mode)
        if self.freeze_encoder is True:
            self.model.base_model.encoder.training = False
            self.model.base_model.decoder.train(mode)
            self.model.base_model.encoder.eval()
            for param in self.model.base_model.encoder.parameters():
                param.requires_grad = False
        return self

    def eval(self):
        self.training = False
        self.model.training = False
        self.model.base_model.training = False
        self.model.base_model.encoder.training = False
        self.model.base_model.decoder.training = False
        self.model.eval()


    def redefine_config(self):
        self.config.architectures[0] = "MixedModel"
        self.config.encoder_attention_heads = self.model.base_model.encoder.config.num_attention_heads
        #self.config.hidden_size = None
        #self.config.hidden_size = self.model.base_model.encoder.config.hidden_size
        self.config.encoder_layers = self.model.base_model.encoder.config.num_hidden_layers

    def encode(self, sentences: Union[str, List[str], List[int]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               is_pretokenized: bool = False,
               num_workers: int = 0) -> Union[List[Tensor], np.ndarray, Tensor]:

        """
                Computes sentence embeddings
                :param sentences: the sentences to embed
                :param batch_size: the batch size used for the computation
                :param show_progress_bar: Output a progress bar when encode sentences
                :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
                :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
                :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
                :param is_pretokenized: DEPRECATED - No longer used, will be removed in the future
                :param device: Which torch.device to use for the computation
                :param num_workers: DEPRECATED - No longer used, will be removed in the future
                :return:
                   By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
                """

        device = "cuda:0"
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = (
                    logging.getLogger().getEffectiveLevel() == logging.INFO or logging.getLogger().getEffectiveLevel() == logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True


        self.to(device)

        all_embeddings = []
        length_sorted_idx = np.argsort([len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):

            with torch.no_grad():
                sentences_batch = sentences_sorted[start_index:start_index + batch_size]
                _, out_features = self.forward(sentences_batch, partial_value=True, generate_sentences=False)
                embeddings = out_features

                if output_value == 'token_embeddings':
                    # Set token embeddings to 0 for padding tokens
                    input_mask = out_features['attention_mask']
                    input_mask_expanded = input_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    embeddings = embeddings * input_mask_expanded

                embeddings = embeddings.detach()

                # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                if convert_to_numpy:
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings


    def batch_encode_plus(self, sentences, padding='max_length', verbose=True):
        train_input_ids = self.tokenizer.batch_encode_plus(
                    sentences,
                    return_tensors='pt',
                    max_length=self.max_seq_length,
                    padding=padding,
                    #pad_to_max_length=True,
                    truncation=True,
                )
        return train_input_ids

    def add_pooling_layer(self):
        if not hasattr(self, 'embedding_pooling'):# and self.task == "reconstruction":
            self.embedding_pooling = Pooling(self.get_word_embedding_dimension(),
                                 pooling_mode_mean_tokens=True,
                                 pooling_mode_cls_token=False,
                                 pooling_mode_max_tokens=False)

