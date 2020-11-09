import os
import re
import logging
import numpy as np
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


from bs4 import BeautifulSoup
from string import punctuation
from soynlp.normalizer import repeat_normalize
from emoji import emojize, demojize, get_emoji_regexp
from itertools import chain

# from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction

from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    ElectraTokenizer,
    BertTokenizer,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification,
)
from models.tokenization_hanbert import HanBertTokenizer
from models.tokenization_kobert import KoBertTokenizer
from models.tokenization_kocharelectra import KoCharElectraTokenizer
from models.data.processor_ner import SentimentExtractionProcessor
from models.modeling_plm_crf import BertCFRForTokenClassification, ElectraCRFForTokenClassification


MODEL_PATH_MAP = { 
    "kobert": "monologg/kobert",
    "hanbert": "models/HanBert-54kN-torch",
    "koelectra-base": "monologg/koelectra-base-discriminator",
    "koelectra-small": "monologg/koelectra-small-discriminator",
    "koelectra-base-v2": "monologg/koelectra-base-v2-discriminator",
    "koelectra-small-v2": "monologg/koelectra-small-v2-discriminator",
    "kocharelectra-base": "monologg/kocharelectra-base-discriminator",
    
    "hanbert-crf" : "models/HanBert-54kN-torch",
    "koelectra-base-crf" : "monologg/koelectra-base-discriminator",
    "kocharelectra-base-crf" : "monologg/kocharelectra-base-discriminator",
    
    "kcbert-base" : "beomi/kcbert-base",
    "kcbert-large" : "beomi/kcbert-large",
    
}


CONFIG_CLASSES = {
    "kobert": BertConfig,
    "hanbert": BertConfig,
    "koelectra-base": ElectraConfig,
    "koelectra-small": ElectraConfig,
    "koelectra-base-v2": ElectraConfig,
    "koelectra-small-v2": ElectraConfig,
    "kocharelectra-base": ElectraConfig,
    
    "hanbert-crf" : BertConfig,
    "koelectra-base-crf" : ElectraConfig,
    "kocharelectra-base-crf" : ElectraConfig,
    
    "kcbert-base" : BertConfig,
    "kcbert-large" : BertConfig,
}

TOKENIZER_CLASSES = {
    "kobert": KoBertTokenizer,
    "hanbert": HanBertTokenizer,
    "koelectra-base": ElectraTokenizer,
    "koelectra-small": ElectraTokenizer,
    "koelectra-base-v2": ElectraTokenizer,
    "koelectra-small-v2": ElectraTokenizer,
    "kocharelectra-base": KoCharElectraTokenizer,
    
    "hanbert-crf" : HanBertTokenizer,
    "koelectra-base-crf" : ElectraTokenizer,
    "kocharelectra-base-crf" : KoCharElectraTokenizer,
    
    "kcbert-base" : BertTokenizer,
    "kcbert-large" : BertTokenizer,
    
}
MODEL_FOR_TOKEN_CLASSIFICATION = {
    "kobert": BertForTokenClassification,
    "hanbert": BertForTokenClassification,
    "koelectra-base": ElectraForTokenClassification,
    "koelectra-small": ElectraForTokenClassification,
    "koelectra-base-v2": ElectraForTokenClassification,
    "koelectra-small-v2": ElectraForTokenClassification,
    "kocharelectra-base": ElectraForTokenClassification,
    
    "hanbert-crf" : BertCFRForTokenClassification,
    "koelectra-base-crf" : ElectraCRFForTokenClassification,
    "kocharelectra-base-crf" : ElectraCRFForTokenClassification,
    
    "kcbert-base" : BertForTokenClassification,
    "kcbert-large" : BertForTokenClassification,
}



label_map = SentimentExtractionProcessor().get_labels()

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    label_map = SentimentExtractionProcessor().get_labels()
    preds = np.argmax(predictions, axis=2)
    
    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]
    
    ## hard-coded
    if predictions.shape[-1] == 5: # plm + token-classification ignore_index=-100
        label_map = label_map[1:]
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

    elif predictions.shape[-1] == 6: # index 0이 ignore_index 값임.
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != 0:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])


    return preds_list, out_label_list

def compute_metrics(p: EvalPrediction) -> Dict:
    preds_list, out_label_list = align_predictions(p.predictions, p.label_ids) # list of list
    
    y_true = list(chain(*out_label_list)) # post_process for token-level comparison
    y_pred = list(chain(*preds_list))
    
    y_true = [tok.split('-')[0] for tok in y_true]
    y_pred = [tok.split('-')[0] for tok in y_pred]
    
    return {
        "token-f1" : f1_score(y_true, y_pred, average='macro'),
        "token-acc" : accuracy_score(y_true, y_pred)      
#         "acc" : accuracy_score(out_label_list, preds_list), 
#         "precision": precision_score(out_label_list, preds_list, suffix=True),
#         "recall": recall_score(out_label_list, preds_list, suffix=True),
#         "f1": f1_score(out_label_list, preds_list, suffix=True),
    }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

        
def strpreprocess(string):
    """모델링용 텍스트 전처리"""
    # compile basics
    html = re.compile(r'<\s*a[^>]*>(.*?)<\s*/\s*a>')
    url = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    email = re.compile('([0-9a-zA-Z_]|[^\s^\w])+(@)[a-zA-Z]+.[a-zA-Z)]+')
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    
    # process
    if html.search(string) != None: # html js 처리
        soup = BeautifulSoup(string, "lxml")
        for script in soup(["script", "style"]):
            script.decompose()
        string = soup.get_text()
    string = string.strip()
    string = re.sub(rf'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+',' ', string)
    string = re.sub('&nbsp;',' ', string) #&nbsp; 제거
    string = re.sub('&lt;','<',string) #기타 html특수기호
    string = re.sub('&gt;','>',string) #기타 html특수기호
    string = re.sub('&amp;','&',string) #기타 html특수기호
    string = re.sub('&quot;','""',string) #기타 html특수기호
    string = repeat_normalize(string, num_repeats=3) # repeats
    string = url.sub(' [URL] ', string) # url
    string = email.sub(' [EMAIL] ', string) # email
    string = demojize(string, delimiters=(' :', ': ')) # emoji를 영문으로 변환
    string = re.sub(r'@(\w+)',r' ', string) # Mention 제거
    for ht in re.findall(r'#(\w+)', string): # spacing to hashtag
        p = re.compile(f'#{ht}')
        string = p.sub(f' #{ht} ', string)
    # 요기에 보존리스트 추가하자.
    string = re.sub(r'\s+',' ', string) #white space character 변환 연속 하나로
    
    return string.strip()