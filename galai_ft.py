# This Source Code Form is subject to the terms of the MIT
# License. If a copy of the same was not distributed with this
# file, You can obtain one at
# https://github.com/akhilpandey95/galsciriff/blob/main/LICENSE.

import os
import gc
import time
import torch
import wandb
import pathlib
import logging
import datasets
import numpy as np
import polars as pl
from pprint import pprint
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig, set_seed
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling

# enable logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# setup wandb
wandb_project = "galsciriff"
wandb.init(project=wandb_project)
logger.info(f"WandDB setup init to {wandb_project}")

# op model dir
output_dir = pathlib.Path("./galactica-sciriff-finetuned")
output_dir.mkdir(exist_ok=True, parents=True)
logger.info(f"Saving FT-model to: {output_dir}")

# set polars threads and new eager engine
os.environ['POLARS_MAX_THREADS'] = "12"
os.environ["POLARS_FORCE_NEW_STREAMING"] = "1"
logger.info(f"Polars concurrency set to {pl.thread_pool_size()} threads.")

# helper function to map data and tokenize data
def tokenize_data(dataset):
    """
    Map function to process SciRIFF dataset and tokenize
    inputs and outputs, and create 'labels'

    --------------------
    Parameters
        arg1 | dataset: TBA
            Train/Val/Test dataset object to be processed

    --------------------
    Returns
        Dataset
    """
    return True

# load the sciriff dataset
data_dir = pathlib.Path("/Users/akhilakella/code/models/sciriff") 

# load DS from local dir
#dataset = datasets.load_dataset("parquet", data_dir=data_dir, trust_remote_code=True)

# /or load the 4096 context window DS
dataset = datasets.load_dataset("allenai/SciRIFF", "4096")
logger.info(f"Loaded SCIRIFF dataset, {dataset}")

# set seed
set_seed(2025)

# get device status
device = torch.device("mps" if torch.mps.is_available() else "cpu")
logger.info("----------------------------------")
logger.info(f"Using {device} to run GALACTICA")
logger.info("----------------------------------")

# init the tookenizer
model_id = "/Users/akhilakella/code/models/galactica-1.3b"
start = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# add special tokens
tokenizer.add_special_tokens({"eos_token": "</s>"})
logger.info(f"Setting </s> EOS token for {model_id}")
tokenizer.pad_token_id = 1
#tokenizer.eos_token_id = tokenizer.encode("</s>")[0]
tokenizer.padding_side = 'left'
tokenizer.model_max_length = 1024

# tokenizer quirks
logger.info(f"EOS token-id: {tokenizer.eos_token_id}")
logger.info(f"PAD token-id: {tokenizer.pad_token_id}")

# load the model
attn_implementation = "sdpa"
model = AutoModelForCausalLM.from_pretrained(model_id, \
                                            trust_remote_code=True, \
                                            torch_dtype=torch.bfloat16, \
                                            low_cpu_mem_usage=True, \
                                            attn_implementation=attn_implementation, \
                                            device_map=device)

# resize tokenizer
logger.info(f"Tokenizer length pre resize: {len(tokenizer)}")
model.resize_token_embeddings(len(tokenizer))
logger.info(f"Tokenizer length after resize: {len(tokenizer)}")

# calc model-tokenizer time
end = time.time()
logger.info(f"Model-tokenizer load time: {end - start} seconds")
logger.info("----------------------------------")

# gather the train and test datasets
train_dataset = dataset["train"]
test_dataset = dataset["test"]
val_dataset = dataset["validation"]

# print sample cut of the dataset
logger.info("Sample Input from the SciRIFF dataset..")
logger.info(f"Input\n: {train_dataset[-1]["input"]}")
logger.info("----------------------------------")
logger.info(f"Output\n: {train_dataset[-1]["output"]}")
logger.info("----------------------------------")

# LoRA config
alpha = 4
r = 4
peft_config = LoraConfig(
    lora_alpha=alpha,
    lora_dropout=0.05,
    r=r,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# create a peft model
min_model = get_peft_model(model, peft_config)
logger.info(f"lora config applied to GALACTICA: r={r}, alpha={alpha}")

#
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to False for causal LM (no masked LM)
)

