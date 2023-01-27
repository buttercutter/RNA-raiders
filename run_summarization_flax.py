#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for summarization.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm

import evaluate
import jax
import jax.numpy as jnp
import optax
import transformers
from filelock import FileLock
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSeq2SeqLM,
    HfArgumentParser,
    is_tensorboard_available,
)
from transformers.utils import get_full_repo_name, is_offline_mode, send_example_telemetry
import pandas as pd

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    # buttercuter added 12/6/22
    per_device_gradient_accumulation_steps: int = field(
        default=1,
        metadata={
            "help": "Number of updates steps to accumulate before performing a backward/update pass."
        },
    )    
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    eval_steps: int = field(default=None, metadata={"help": "Run an evaluation every X steps."})
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    hub_model_id: str = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_token: str = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )

    def __post_init__(self):
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
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
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized and trained. Choose one of"
                " `[float32, float16, bfloat16]`."
            )
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )    


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(default="train_small_20.csv", metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default="eval_small_10.csv",
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default="eval_small_10.csv",
        metadata={"help": "An optional input predict data file to do prediction on (a text file)."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the `max_length` param of `model.generate`, which is used "
                "during evaluation."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to `model.generate`, "
                "which is used during evaluation."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False, drop_last=True):
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete,
    and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch


def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def to_fp32(t):
    return jax.tree_map(lambda x: x.astype(jnp.float32) if x.dtype == jnp.bfloat16 else x, t)


def to_bf16(t):
    return jax.tree_map(lambda x: x.astype(jnp.bfloat16) if x.dtype == jnp.float32 else x, t)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args, framework="flax")

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    to_dtype = to_bf16 if model_args.dtype=="bfloat16" else to_fp32

    # Handle the repository creation
    if training_args.push_to_hub:
        if training_args.hub_model_id is None:
            repo_name = get_full_repo_name(
                Path(training_args.output_dir).absolute().name, token=training_args.hub_token
            )
        else:
            repo_name = training_args.hub_model_id
        repo = Repository(training_args.output_dir, clone_from=repo_name)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            keep_in_memory=False,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    else:

        data_files = {}

        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        
        else:
            data_files["train"] = 'train.csv'
            extension = data_args.train_file.split(".")[-1]            

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]

        else:
            data_files["validation"] = 'validate.csv'
            extension = data_args.validation_file.split(".")[-1]

        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]

        else:
            data_files["test"] = 'test.csv'
            extension = data_args.test_file.split(".")[-1]           

        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html
        dataset = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            keep_in_memory=True,  # https://huggingface.co/docs/datasets/cache#enable-or-disable-caching
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Load pretrained model and tokenizer

    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = FlaxAutoModelForSeq2SeqLM.from_config(
            config,
            seed=training_args.seed,
            dtype=getattr(jnp, model_args.dtype),
        )

    if training_args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length and max_source_lengthfor training.
    max_target_length = data_args.max_target_length
    max_source_length = data_args.max_source_length

    # In Flax, for seq2seq models we need to pass `decoder_input_ids`
    # as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
    # for that dynamically import the `shift_tokens_right` function from the model file
    model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
    shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

    # train_df_filtered = \
    #     filter_file_for_max_tokens_and_add_tasks(model, tokenizer, file_name="train.csv", max_input_length=max_source_length, max_output_length=max_target_length,  
    #                                              experiment_folder="./experiment/", write_to_file="train_filtered.csv", filter_by_token_count=True, shuffle=True, 
    #                                              add_classification_task=False, add_correctness_task_ratio=0, add_fix_task_ratio=0)

    # eval_df_filtered = \
    #     filter_file_for_max_tokens_and_add_tasks(model, tokenizer, file_name="eval.csv", max_input_length=max_source_length, max_output_length=max_target_length,  
    #                                              experiment_folder="./experiment/", write_to_file="eval_filtered.csv", filter_by_token_count=True, shuffle=True, 
    #                                              add_classification_task=False, add_correctness_task_ratio=0, add_fix_task_ratio=0)

    # test_df_filtered = \
    #     filter_file_for_max_tokens_and_add_tasks(model, tokenizer, file_name="test.csv", max_input_length=max_source_length, max_output_length=max_target_length,  
    #                                              experiment_folder="./experiment/", write_to_file="test_filtered.csv", filter_by_token_count=True, shuffle=True, 
    #                                              add_classification_task=False, add_correctness_task_ratio=0, add_fix_task_ratio=0)                                                 

    # Setting padding="max_length" as we need fixed length inputs for jitted functions
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True, return_tensors="np"
        )

        # Setup the tokenizer for targets
        labels = tokenizer(
            text=targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        model_inputs["labels"] = labels["input_ids"]
        decoder_input_ids = shift_tokens_right_fn(
            labels["input_ids"], config.pad_token_id, config.decoder_start_token_id
        )
        model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)

        # We need decoder_attention_mask so we can ignore pad tokens from loss
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        return model_inputs

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = dataset["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )


    def load_dataset_from_csv_file(file, prompt_column, answer_column):
        max_target_length = data_args.val_max_target_length
        this_dataset = load_dataset("csv", data_files=file)
        return_dataset = this_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )


    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        try:
            from flax.metrics.tensorboard import SummaryWriter

            summary_writer = SummaryWriter(log_dir=Path(training_args.output_dir))
        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    batch_size_per_update = train_batch_size * training_args.per_device_gradient_accumulation_steps
    per_device_eval_batch_size = int(training_args.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()
    try:
      if train_batch_size > 0 and train_dataset!=None:
        steps_per_epoch = len(train_dataset) // train_batch_size
      else:
        steps_per_epoch = 0
    except:
      # does not exist
      steps_per_epoch = 0

    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    try:
      linear_decay_lr_schedule_fn = create_learning_rate_fn(
          len(train_dataset),
          train_batch_size,
          training_args.num_train_epochs,
          training_args.warmup_steps,
          training_args.learning_rate,
      )
    except:
      print("")
    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = set(
            [
                layer[-2:]
                for layer_norm_name in layer_norm_candidates
                for layer in flat_params.keys()
                if layer_norm_name in "".join(layer).lower()
            ]
        )
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    try:
      optimizer = optax.adamw(
          learning_rate=linear_decay_lr_schedule_fn,
          b1=training_args.adam_beta1,
          b2=training_args.adam_beta2,
          eps=training_args.adam_epsilon,
          weight_decay=training_args.weight_decay,
          mask=decay_mask_fn,
      )
    except:
      print("")

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rng)

    # label smoothed cross entropy
    def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
        """
        The label smoothing implementation is adapted from Flax's official example:
        https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
        """
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum()
        num_labels = padding_mask.sum()
        return loss, num_labels
  
        
    # Define eval fn
    def eval_step(params, batch, label_smoothing_factor=0.0):
        labels = batch.pop("labels")
        logits = model(**batch, params=params, train=False)[0]

        loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
        num_labels = jax.lax.psum(num_labels, "batch")

        # true loss = total loss / total samples
        loss = jax.lax.psum(loss, "batch")
        loss = jax.tree_util.tree_map(lambda x: x / num_labels, loss)

        metrics = {"loss": loss}
        return metrics

    # Define generation function
    max_length = (
        data_args.val_max_target_length if data_args.val_max_target_length is not None else model.config.max_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs)
        return output_ids.sequences

    def save_checkpoint(epoch, current_step):
        '''
        added by parapraxis on 12/5/22
        '''
        params = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state.params))
        model.save_pretrained(training_args.output_dir, params=params)
        tokenizer.save_pretrained(training_args.output_dir)
        if training_args.push_to_hub:
            repo.push_to_hub(commit_message=f"Saving weights and logs of epoch {epoch}", blocking=False)


    def get_predictions_for_dataset(pred_dataset, prompt_field, correct_answer_field=None, save_csv_file=True):
        '''
        added by parapraxis on 12/5/22
        '''
        logger.info("*** Predict ***")

        pred_metrics = []
        pred_generations = []
        pred_labels = []
        # load the dataset
        
        pred_loader = data_loader(input_rng, pred_dataset, eval_batch_size, drop_last=False)
        pred_steps = math.ceil(len(pred_dataset) / eval_batch_size)
        for _ in tqdm(range(pred_steps), desc="Predicting...", position=2, leave=False):
            # Model forward
            batch = next(pred_loader)
            labels = batch["labels"]

            metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                state.params, batch, min_device_batch=per_device_eval_batch_size
            )
            pred_metrics.append(metrics)

            # generation
            if data_args.predict_with_generate:
                generated_ids = pad_shard_unpad(p_generate_step)(state.params, batch)
                pred_generations.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
                pred_labels.extend(labels)

        # normalize prediction metrics
        pred_metrics = get_metrics(pred_metrics)
        pred_metrics = jax.tree_util.tree_map(jnp.mean, pred_metrics)

        # compute ROUGE metrics
        rouge_desc = ""
        if data_args.predict_with_generate:
            rouge_metrics = compute_metrics(pred_generations, pred_labels)
            pred_metrics.update(rouge_metrics)
            rouge_desc = " ".join([f"Predict {key}: {value} |" for key, value in rouge_metrics.items()])

        # Print metrics
        desc = f"Predict Loss: {pred_metrics['loss']} | {rouge_desc})"
        logger.info(desc)

        # save final metrics in json
        if jax.process_index() == 0:
            rouge_metrics = {f"test_{metric_name}": value for metric_name, value in rouge_metrics.items()}
            path = os.path.join(training_args.output_dir, "test_results.json")
            with open(path, "w") as f:
                json.dump(rouge_metrics, f, indent=4, sort_keys=True)
        # added by parapraxis on 12/5/22
        # save predictions in a new file that is the eval file with the predictions added
        if data_args.predict_with_generate:
            if jax.process_index() == 0:
                # add _results to the file name
                fn_without_ext = os.path.splitext(data_args.test_file)[0]
                if save_csv_file:
                    output_predict_file = fn_without_ext + f"_results_epoch_{epoch}.csv"
                    output_predict_file = os.path.join(training_args.output_dir, output_predict_file)
                test_df = pd.read_csv(data_args.test_file)
                # add decoded predictions using the tokenizer to test_df
                test_df["predictions"] = tokenizer.batch_decode(pred_generations, skip_special_tokens=True)
                if correct_answer_field is not None:
                    # compare predictions to "correct_answer" column after stripping spaces from the ends
                    test_df[correct_answer_field] = test_df[correct_answer_field].str.strip()
                    test_df["predictions"] = test_df["predictions"].str.strip()
                    test_df["correct"] = test_df["predictions"] == test_df[correct_answer_field]
                    # sum the correct items
                    num_correct = test_df["correct"].sum()
                    # get the total number of items
                    num_total = len(test_df)
                    # calculate the accuracy
                    accuracy = num_correct / num_total
                    if save_csv_file:
                        # add number of correct items to the file name
                        output_predict_file = fn_without_ext + f"_results_epoch_{epoch}_correct_{num_correct}.csv"
                        print(f"Saving predictions to {output_predict_file}")
                    print(f"Accuracy: {accuracy}")
                    print(f"Number correct: {num_correct}")
                    print(f"Total number: {num_total}")

                if save_csv_file:
                    test_df.to_csv(output_predict_file, index=False)

                else:
                    return test_df

    def df_to_dataset(df):
        print("loading dataset from pandas dataframe")
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(preprocess_function, batched=True)
        return dataset

    def get_predictions_for_df(data_frame, prompt_column_title, correct_answer_column_title, save_csv_file=True):
        dataset_file = df_to_dataset(data_frame)
        dataset_to_evaluate = load_dataset_from_csv_file(dataset_file)
        get_predictions_for_dataset(dataset_to_evaluate, prompt_column_title, correct_answer_column_title, save_csv_file=save_csv_file)

    def get_predictions_for_file(dataset_file, prompt_column_title, correct_answer_column_title, save_csv_file=True):
        dataset_to_evaluate = load_dataset_from_csv_file(dataset_file)
        get_predictions_for_dataset(dataset_to_evaluate, prompt_column_title, correct_answer_column_title, save_csv_file=save_csv_file)

    def filter_file_for_max_tokens_and_add_tasks(model, tokenizer, file_name, max_input_length, max_output_length, experiment_folder, write_to_file=None, 
                                                 filter_by_token_count=True, shuffle=True, add_classification_task=False, add_correctness_task_ratio=0, add_fix_task_ratio=0):
        """
        This function takes a file_name and filters it for max_tokens and adds additional tasks if requested
        Explanation of parameters:
        file_name: the name of the dataset file to be filtered
        write_to_file: if not empty, the filtered dataset will be written to this file
        shuffle: if True, the dataset will be shuffled
        add_classification_task: if True, the dataset will be doubled and the second half will be a classification task
        add_correctness_task_ratio: if > 0, the dataset will be sampled by this ratio and the sampled data will be used to create a correctness task (i.e., is the answer correct?)
        add_fix_task_ratio: if > 0, the dataset will be sampled by this ratio and the sampled data will be used to create a fix task (i.e., fix the answer)
        """
        
        if "http" in file_name:
            tdf = file_name
        else:
            if "/" not in file_name:
                tdf = f"{experiment_folder}{file_name}"
            else:
                tdf = file_name      

        print(f"filter_file_for_max_tokens {tdf}")
        df = pd.read_csv(tdf)
        if "additional_tasks" not in file_name: # don't run if already added tasks
            if add_correctness_task_ratio > 0:
                print("adding correctness task")
                # sample the add_correctness_task_ratio of the data
                df_correctness_sample = df.sample(frac=add_correctness_task_ratio, random_state=42)
                # run inference on the sampled data
                df_correctness_sample['predictions'] = get_predictions_for_df(model, df_correctness_sample, save_csv_file=False)
                print(f"inference complete for correctness task adding {len(df_correctness_sample)} items")
                # add the correctness task 'prompt' plus 'predictions' and change correct_answer to yes if prediction==correct_answer and no if prediction!=correct_answer
                # make a copy of 'prompt' to prompt_old
                df_correctness_sample['prompt_old'] = df_correctness_sample['prompt']
                df_correctness_sample['prompt'] = "correct?:" + df_correctness_sample['prompt'] + " " + df_correctness_sample['predictions']
                # copy 'correct_answer' to 'correct_answer_board'
                df_correctness_sample['correct_answer_board'] = df_correctness_sample['correct_answer']
                df_correctness_sample['correct_answer'] = df_correctness_sample.apply(lambda row: 'yes' if row['predictions'] == row['correct_answer'] else 'no', axis=1)
                # print head
                print("examples of correctness task")
                print(df_correctness_sample.head())
            if add_fix_task_ratio > 0:
                print("adding fix task")
                if add_correctness_task_ratio > 0:
                    # then use the same sample for the fix task
                    df_fix_sample = df_correctness_sample
                    # use prompt_old for the fix task and correct_answer_board
                    # copy prompt_old to prompt and predictions to correct_answer
                    df_fix_sample['prompt'] = 'fix:' + df_fix_sample['prompt_old']
                    df_fix_sample['correct_answer'] = df_fix_sample['predictions']
                else:
                    # sample the add_fix_task_ratio of the data
                    df_fix_sample = df.sample(frac=add_fix_task_ratio, random_state=42)
                    # run inference on the sampled data
                    print(f"running inference for fix task on {len(df_fix_sample)} items")
                    df_fix_sample['predictions'] = get_predictions_for_df(model, df_fix_sample)
                    df_fix_sample['prompt'] = 'fix:' + df_fix_sample['prompt']
                    df_fix_sample['correct_answer'] = df_fix_sample['predictions']            
                print(f"inference complete for fix task adding {len(df_fix_sample)} items")
                print(df_fix_sample.head())
            if add_classification_task:
                print("adding classification task - doubles the number of training items")
                # create new column id_int that is 'id' without the .json extension and converted to int from hex
                df['id_root_file'] = df['path_data'].apply(lambda x: str(x.split(".")[0].split("_")[0]))
                df['id_int'] = df.groupby('id_root_file').ngroup()
                # convert to string
                df['id_int'] = df['id_int'].apply(lambda x: str(x))
                # iterate over the rows and add the classification task; add "solve:" to the prompt and duplicate the item adding classify to the duplicate
                # duplicate each row and add "classify:" to the prompt (use concat)
                df = pd.concat([df, df]) 
                # reset the index
                df = df.reset_index(drop=True)
                # add "classify:" to the prompt (the newly added items) in the second half of the dataframe
                df.loc[df.index >= len(df)/2, 'prompt'] = "classify: " + df.loc[df.index >= len(df)/2, 'prompt'].astype(str)
                # add "solve:" to the prompt (the original items) in the first half of the dataframe
                df.loc[df.index < len(df)/2, 'prompt'] = "solve: " + df.loc[df.index < len(df)/2, 'prompt'].astype(str)
                # for correct_answer in the second half, set it to df['path_data'] + the next to last string when the id string is split on "/"
                df.loc[df.index >= len(df)/2, 'correct_answer'] = df.loc[df.index >= len(df)/2, 'id'].str.split("/").str[-2] + "/" + df.loc[df.index >= len(df)/2, 'path_data'] + ' ' + df.loc[df.index >= len(df)/2, 'id_int'] # put the id_int at the end of the correct_answer, because this could change if the dataset changes
            
            # need to add these after the classification task, because the classify task adds numbers related to the line number
            if add_correctness_task_ratio > 0:
                df.extend(df_correctness_sample)
            
            if add_fix_task_ratio > 0:
                df.extend(df_fix_sample)
            if add_classification_task or add_correctness_task_ratio > 0 or add_fix_task_ratio > 0:
                # write to csv file
                if write_to_file == None:
                    df.to_csv(f"{experiment_folder}additional_tasks_{file_name}", index=False)
                else:
                    df.to_csv(f"{experiment_folder}additional_tasks_{write_to_file}", index=False)

        if filter_by_token_count:
            print(f"filtering dataset by token count")
            # add the token count columns
            #tokenized_prompts = tokenizer(df['prompt'].tolist(), padding=True, truncation=True, max_length=max_input_length, return_tensors="pt")
            #df['token_count'] = df['prompt'].apply(lambda x: len(tokenized_prompts['input_ids']))
            df['token_count'] = df['prompt'].apply(lambda x: len(tokenizer.encode(x)))
            # same for correct answer
            df['token_count_correct'] = df['correct_answer'].apply(lambda x: len(tokenizer.encode(x)))

            # filter out the examples that are too long
            df = df[df['token_count'] <= max_input_length]
            df = df[df['token_count_correct'] <= max_output_length]
            # fix the index
            df = df.reset_index(drop=True)
        # randomize the dataset order
        if shuffle:
            # shuffle the dataset and do so in a reproducible way
            print("shuffling dataset")
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            #df = df.sample(frac=1).reset_index(drop=True)
        return df


    # Define gradient update step fn
    def train_step(state, batch, label_smoothing_factor=0.0):
        # only one single rng per grad step, with or without accumulation, as the graph should be identical over one effective training batch
        dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
        
        def compute_loss(params, minibatch):
            labels = minibatch.pop("labels")
            logits = state.apply_fn(
                **minibatch,
                params=params,
                dropout_rng=dropout_rng,
                # freeze_feature_encoder=model_args.freeze_feature_encoder,
                train=True,
            )[0]
            loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
            return loss, num_labels

        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)

        if training_args.per_device_gradient_accumulation_steps == 1:
            (loss, num_labels), grad = grad_fn(to_dtype(state.params), batch)

        # Custom gradient accumulation
        else:
            # print("batch = ", batch)
            # See https://github.com/huggingface/transformers/issues/20855
            # add a first dimension over gradient_accumulation_steps for minibatch slices
            batch = jax.tree_map(
                lambda x: x.reshape(
                    training_args.per_device_train_batch_size, training_args.per_device_gradient_accumulation_steps, -1 #*x.shape[1::]
                ),
                batch,
            )

            def accum_minibatch_step(accum_grad, minibatch):
                # compute loss, num labels and grad over minibatch and accumulate
                (loss, num_labels), grad = grad_fn(to_dtype(state.params), minibatch)
                return jax.tree_map(jnp.add, accum_grad, grad), (loss, num_labels)

            # create an initial state for accumulating losses, num labels and gradients
            init_grad = jax.tree_map(jnp.zeros_like, to_dtype(state.params))
            # loop accum minibatch step over the number of gradient accumulation steps
            grad, (loss, num_labels) = jax.lax.scan(accum_minibatch_step, init_grad, batch)

        grad = jax.lax.psum(grad, "batch")
        loss = jax.lax.psum(loss.sum(), "batch")
        total_samples = jax.lax.psum(num_labels.sum(), "batch")
        grad = jax.tree_map(lambda g: g / total_samples, grad)
        loss = jax.tree_map(lambda l: l / total_samples, loss)

        # update state
        new_state = state.apply_gradients(
            grads=grad,
            dropout_rng=new_dropout_rng,
            # to_dtype=to_dtype,
        )

        # compute gradient norms over all layers, total encoder, total decoder and global for detailed monitoring
        layer_grad_norm = jax.tree_map(jnp.linalg.norm, grad)
        '''
        logs = {
            "layer_grad_norm": layer_grad_norm,
            "encoder_grad_norm": jnp.linalg.norm(jax.tree_util.tree_leaves(layer_grad_norm["encoder"])),
            "decoder_grad_norm": jnp.linalg.norm(jax.tree_util.tree_leaves(layer_grad_norm["decoder"])),
        }
        logs["grad_norm"] = jnp.linalg.norm([logs["encoder_grad_norm"], logs["decoder_grad_norm"]])

        # compute parameter norms over all layers, total encoder, total decoder and global for detailed monitoring
        layer_param_norm = jax.tree_map(jnp.linalg.norm, new_state.params)
        logs["layer_param_norm"] = layer_param_norm
        logs["encoder_param_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(layer_param_norm["encoder"]))
        logs["decoder_param_norm"] = jnp.linalg.norm(jax.tree_util.tree_leaves(layer_param_norm["decoder"]))
        logs["param_norm"] = jnp.linalg.norm([logs["encoder_param_norm"], logs["decoder_param_norm"]])
        '''
        metrics = {"loss": loss, "learning_rate": linear_decay_lr_schedule_fn(state.step)}
        #metrics.update(logs)

        metrics = jax.lax.pmean(metrics, axis_name="batch")
        # metrics = to_fp32(metrics)

        return new_state, metrics


    # Create parallel version of the train and eval step
    p_train_step = jax.pmap(
        partial(train_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch", donate_argnums=(0,)
    )
    p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch")
    p_generate_step = jax.pmap(generate_step, "batch")

    # Replicate the train state on each device
    state = state.replicate()

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    # buttercutter added 12/6/22
    logger.info(f"  Total train batch size (w. parallel, distributed & gradient accumulation) = {batch_size_per_update}")
#    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_time = 0
    epochs = tqdm(range(num_epochs), desc=f"Epoch ... (1/{num_epochs})", position=0)
    for epoch in epochs:
        # ======================== Training ================================
        train_start = time.time()

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)
        train_metrics = []

        # Generate an epoch by shuffling sampling indices from the train dataset
        train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)
        steps_per_epoch = len(train_dataset) // train_batch_size
        # train
        for cstep in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            batch = next(train_loader)
            batch = shard(batch)
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)
            # added by parapraxis on 12/5/22
            if cstep % training_args.logging_steps == 0:
                # print data from train_metric
                print("train_metric", train_metric)
            if cstep % training_args.save_steps == 0:
                save_checkpoint(epoch, state.step)

        train_time += time.time() - train_start
        # this was failing so I changed train_metric to train_metrics - parapraxis 11/30/22
        train_metric = unreplicate(train_metric)
        
        # added by parapraxis on 12/5/22
        save_checkpoint(epoch, steps_per_epoch * (epoch + 1))

        epochs.write(
            f"Epoch... ({epoch + 1}/{num_epochs} | Loss: {train_metric['loss']}, Learning Rate:"
            f" {train_metric['learning_rate']})"
        )

        # ======================== Evaluating ==============================
        eval_metrics = []
        eval_preds = []
        eval_labels = []

        eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size, drop_last=False)
        eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
        for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
            # Model forward
            batch = next(eval_loader)
            labels = batch["labels"]

            metrics = pad_shard_unpad(p_eval_step, static_return=True)(
                state.params, batch, min_device_batch=per_device_eval_batch_size
            )
            eval_metrics.append(metrics)

            # generation
            if data_args.predict_with_generate:
                generated_ids = pad_shard_unpad(p_generate_step)(state.params, batch)
                eval_preds.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
                eval_labels.extend(labels)

        # normalize eval metrics
        eval_metrics = get_metrics(eval_metrics)
        eval_metrics = jax.tree_util.tree_map(jnp.mean, eval_metrics)

        # compute ROUGE metrics
        rouge_desc = ""
        if data_args.predict_with_generate:
            rouge_metrics = compute_metrics(eval_preds, eval_labels)
            eval_metrics.update(rouge_metrics)
            rouge_desc = " ".join([f"Eval {key}: {value} |" for key, value in rouge_metrics.items()])

        # Print metrics and update progress bar
        desc = f"Epoch... ({epoch + 1}/{num_epochs} | Eval Loss: {eval_metrics['loss']} | {rouge_desc})"
        epochs.write(desc)
        epochs.desc = desc

        # Save metrics
        if has_tensorboard and jax.process_index() == 0:
            cur_step = epoch * (len(train_dataset) // train_batch_size)
            write_metric(summary_writer, train_metrics, eval_metrics, train_time, cur_step)
        # prediction loop added to the epoch loop by parapraxis on 12/5/22
        # ======================== Prediction loop ==============================
        if training_args.do_predict:
            get_predictions_for_dataset(predict_dataset, text_column, summary_column)


if __name__ == "__main__":
    main()


# # normal operation            
# if __name__ == "__main__":
#     main()