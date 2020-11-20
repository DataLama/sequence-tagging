import os
from psutil import cpu_count
os.environ["OMP_NUM_THREADS"] = str(cpu_count(logical=True))
os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'
import numpy as np
import logging
from time import time

from transformers import AutoTokenizer
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions, get_all_providers

from pipeline.onnx_pipeline import TokenClassificationONNXPipeline

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)


def create_model_for_provider(model_path: str, provider: str) -> InferenceSession: 
  
    assert provider in get_all_providers(), f"provider {provider} not found, {get_all_providers()}"

    # Few properties that might have an impact on performances (provided by MS)
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load the model as a graph and prepare the CPU backend 
    session = InferenceSession(model_path, options, providers=[provider])
    session.disable_fallback()

    return session

if __name__=="__main__":
    # setup
    model_path = 'onnx/zh-sentiment-tagging-test.onnx'
    provider = "CUDAExecutionProvider"
    onnx_model = create_model_for_provider(model_path, provider)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-electra-180g-large-discriminator", use_fast=True)
    tokens = tokenizer('今天来回报一个好消息!平时没事的时候也经常逛论坛,看看姐妹们分享的护肤化妆技巧经验什么的,却很少说些什么,这次来发一个帖子,跟大家聊聊护肤的那些事。')
    
    #preprocess
    tokens = {k: np.array(v, ndmin=2) for k, v in tokens.items()}
    batch, seq_len = tokens['input_ids'].shape
    tokens['attention_mask'] = np.ones((batch, seq_len), dtype=int)

    # run
    inference_times = []
        
    for _ in range(100):
        tick = time()
        onnx_model.run(None, tokens)
        tok = time()
        inference_times.append(tok-tick)

    mean = np.mean(inference_times)
    std = np.std(inference_times)
    
    logging.info(f"[*] The Speed of  {provider} => {mean} s ± {std} s")
    