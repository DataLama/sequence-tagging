import numpy as np
from typing import List, Dict, Union, Optional
from tqdm.auto import tqdm

from transformers import PreTrainedTokenizer, ModelCard, BasicTokenizer
from transformers.pipelines import ArgumentHandler, _ScikitCompat, TokenClassificationArgumentHandler

from os import environ
from psutil import cpu_count
environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
environ["OMP_WAIT_POLICY"] = 'ACTIVE'
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers
from onnxruntime.capi.session import InferenceSession


class PipelineORT(_ScikitCompat):
    
    default_input_names = None
    
    def __init__(
        self,
        model_path: str, # 변경
        tokenizer: PreTrainedTokenizer,
        args_parser: ArgumentHandler = None,
        device: int = -1,
        binary_output: bool = False,
    ):
        self.tokenizer = tokenizer
        self.args_parser = args_parser
        self.device = 'CPUExecutionProvider' if device < 0 else 'CUDAExecutionProvider'
        
        # load model 
        self.model = self._create_model_for_provider(model_path, provider=self.device)
        assert self.device in self.model.get_providers(), f"provider {self.device} is not found, in my model."
        
    
    def _create_model_for_provider(self, model_path: str, provider: str) -> InferenceSession:
        assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

        # Few properties that might have an impact on performances (provided by MS)
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # Load the model as a graph and prepare the CPU backend 
        session = InferenceSession(model_path,sess_options=options,  providers=[provider])
        session.disable_fallback()

        return session
    
    
    def transform(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)

    def predict(self, X):
        """
        Scikit / Keras interface to transformers' pipelines. This method will forward to __call__().
        """
        return self(X=X)
    
    def _parse_and_tokenize(self, inputs, padding=True, add_special_tokens=True, **kwargs):
        """
        Parse arguments and tokenize
        """
        # Parse arguments
        inputs = self.tokenizer(
            inputs,
            add_special_tokens=add_special_tokens,
            return_tensors='np',
            padding=padding,
        )

        return inputs
    
    def __call__(self, *args, **kwargs):
        inputs = self._parse_and_tokenize(*args, **kwargs)
        return self._forward(inputs)

    def _forward(self, inputs, return_tensors=False):
        """
        Internal framework specific forward dispatching
        Args:
            inputs: dict holding all the keyword arguments for required by the model forward method.
            return_tensors: Whether to return native framework (pt/tf) tensors rather than numpy array
        Returns:
            Numpy array
        """
        return self.model.run(None, inputs)[0]

    
class TokenClassificationPipelineORT(PipelineORT):
    """
    Named Entity Recognition pipeline using any :obj:`ModelForTokenClassification`. See the `named entity recognition
    examples <../task_summary.html#named-entity-recognition>`__ for more information.
    This token recognition pipeline can currently be loaded from :func:`~transformers.pipeline` using the following
    task identifier: :obj:`"ner"` (for predicting the classes of tokens in a sequence: person, organisation, location
    or miscellaneous).
    The models that this pipeline can use are models that have been fine-tuned on a token classification task. See the
    up-to-date list of available models on `huggingface.co/models
    <https://huggingface.co/models?filter=token-classification>`__.
    """

    default_input_names = "sequences"

    def __init__(
        self,
        model_path: str, # new
        tokenizer: PreTrainedTokenizer,
        modelcard: Optional[ModelCard] = None,
        framework: Optional[str] = None,
        args_parser: ArgumentHandler = TokenClassificationArgumentHandler(),
        device: int = -1,
        binary_output: bool = False,
        ignore_labels=["O"],
        task: str = "",
        grouped_entities: bool = False,
        ignore_subwords: bool = False,
        idx2label: List = None, # new
    ):
        super().__init__(
            model_path=model_path,
            tokenizer=tokenizer,
            device=device,
            binary_output=binary_output,
        )


        self._basic_tokenizer = BasicTokenizer(do_lower_case=False)
        self._args_parser = args_parser
        self.ignore_labels = ignore_labels
        self.grouped_entities = grouped_entities
        self.ignore_subwords = ignore_subwords
        self.idx2label = idx2label

        if self.ignore_subwords and not self.tokenizer.is_fast:
            raise ValueError(
                "Slow tokenizers cannot ignore subwords. Please set the `ignore_subwords` option"
                "to `False` or use a fast tokenizer."
            )

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.
        Args:
            inputs (:obj:`str` or :obj:`List[str]`):
                One or several texts (or one list of texts) for token classification.
        Return:
            A list or a list of list of :obj:`dict`: Each result comes as a list of dictionaries (one for each token in
            the corresponding input, or each entity if this pipeline was instantiated with
            :obj:`grouped_entities=True`) with the following keys:
            - **word** (:obj:`str`) -- The token/word classified.
            - **score** (:obj:`float`) -- The corresponding probability for :obj:`entity`.
            - **entity** (:obj:`str`) -- The entity predicted for that token/word.
            - **index** (:obj:`int`, only present when ``self.grouped_entities=False``) -- The index of the
              corresponding token in the sentence.
        """
        
        if isinstance(inputs, str):
            inputs, offset_mappings = self._args_parser(inputs, **kwargs)
        else:
            inputs, offset_mappings = self._args_parser(inputs, **kwargs)
            inputs = inputs[0]

        answers = []

        for i, sentence in enumerate(tqdm(inputs, desc="Sentiment-Tagging")):

            # Manage correct placement of the tensors
            tokens = self.tokenizer(
                sentence,
                return_attention_mask=False,
                return_tensors='np',
                truncation=True,
                return_special_tokens_mask=True,
                return_offsets_mapping=self.tokenizer.is_fast,
            )
            if self.tokenizer.is_fast:
                offset_mapping = tokens.pop("offset_mapping")[0]
            elif offset_mappings:
                offset_mapping = offset_mappings[i]
            else:
                offset_mapping = None

            special_tokens_mask = tokens.pop("special_tokens_mask")[0]

            # Forward
            tokens = {k:v for k,v in tokens.items()}
            batch, seq_len = tokens['input_ids'].shape
            tokens['attention_mask'] = np.ones((batch, seq_len), dtype=int)
            entities = self.model.run(None, tokens)[0].squeeze()
            input_ids = tokens["input_ids"].squeeze()

            # softmax for calcuate the probability
            score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
            labels_idx = score.argmax(axis=-1)

            entities = []
            # Filter to labels not in `self.ignore_labels`
            # Filter special_tokens
            filtered_labels_idx = [
                (idx, label_idx)
                for idx, label_idx in enumerate(labels_idx)
                if (self.idx2label[label_idx] not in self.ignore_labels) and not special_tokens_mask[idx]
            ]

            for idx, label_idx in filtered_labels_idx:
                if offset_mapping is not None:
                    start_ind, end_ind = offset_mapping[idx]
                    word_ref = sentence[start_ind:end_ind]
                    word = self.tokenizer.convert_ids_to_tokens([int(input_ids[idx])])[0]
                    is_subword = len(word_ref) != len(word)

                    if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                        word = word_ref
                        is_subword = False
                else:
                    word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))

                entity = {
                    "word": word,
                    "score": score[idx][label_idx].item(),
                    "entity": self.idx2label[label_idx],
                    "index": idx,
                }

                if self.grouped_entities and self.ignore_subwords:
                    entity["is_subword"] = is_subword

                entities += [entity]

            if self.grouped_entities:
                answers += [self.group_entities(entities)]
            # Append ungrouped entities
            else:
                answers += [entities]

        if len(answers) == 1:
            return answers[0]
        return answers

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """
        Group together the adjacent tokens with the same entity predicted.
        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        scores = np.nanmean([entity["score"] for entity in entities])
        tokens = [entity["word"] for entity in entities]

        entity_group = {
            "entity_group": entity,
            "score": np.mean(scores),
            "word": self.tokenizer.convert_tokens_to_string(tokens),
        }
        return entity_group

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """
        Find and group together the adjacent tokens with the same entity predicted.
        Args:
            entities (:obj:`dict`): The entities predicted by the pipeline.
        """

        entity_groups = []
        entity_group_disagg = []

        if entities:
            last_idx = entities[-1]["index"]

        for entity in entities:

            is_last_idx = entity["index"] == last_idx
            is_subword = self.ignore_subwords and entity["is_subword"]
            if not entity_group_disagg:
                entity_group_disagg += [entity]
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
                continue

            # If the current entity is similar and adjacent to the previous entity, append it to the disaggregated entity group
            # The split is meant to account for the "B" and "I" suffixes
            # Shouldn't merge if both entities are B-type
            if (
                (
                    entity["entity"].split("-")[-1] == entity_group_disagg[-1]["entity"].split("-")[-1]
                    and entity["entity"].split("-")[0] != "B"
                )
                and entity["index"] == entity_group_disagg[-1]["index"] + 1
            ) or is_subword:
                # Modify subword type to be previous_type
                if is_subword:
                    entity["entity"] = entity_group_disagg[-1]["entity"].split("-")[-1]
                    entity["score"] = np.nan  # set ignored scores to nan and use np.nanmean

                entity_group_disagg += [entity]
                # Group the entities at the last entity
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]
            # If the current entity is different from the previous entity, aggregate the disaggregated entity group
            else:
                entity_groups += [self.group_sub_entities(entity_group_disagg)]
                entity_group_disagg = [entity]
                # If it's the last entity, add it to the entity groups
                if is_last_idx:
                    entity_groups += [self.group_sub_entities(entity_group_disagg)]

        return entity_groups