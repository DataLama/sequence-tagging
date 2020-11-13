from typing import Dict, List, Optional, Tuple

import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import EvalPrediction


def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_map[label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list, out_label_list


def compute_metrics(p: EvalPrediction) -> Dict:
        preds_list, out_label_list =  align_predictions(p.predictions, p.label_ids)
        
        ## token-level comparison
        y_true = list(chain(*out_label_list))
        y_pred = list(chain(*preds_list))

        y_true = [tok.split('-')[0] for tok in y_true]
        y_pred = [tok.split('-')[0] for tok in y_pred]
        ##
        
        return {
            "token-f1": f1_score(y_true, y_pred, average='macro'),
            "token-acc": accuracy_score(y_true, y_pred)
        }