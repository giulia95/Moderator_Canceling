import logging
import os
import sys

from importlib.util import find_spec

import datasets
import determined as det
import transformers

from determined.transformers import DetCallback
from transformers import TrainingArguments
from transformers.hf_argparser import HfArgumentParser

from arguments import DataArguments, ModelArguments
from train import main


logger = logging.getLogger(__name__)

if __name__ == "__main__":
    info = det.get_cluster_info()

    if info is None:
        msg = "This script must be run in a Determined trial."
        raise ValueError(msg)

    hparams = info.trial.hparams if info.task_type != "NOTEBOOK" else None

    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))  # type: ignore

    if hparams is None:
        msg = "No hyperparameters found. Please run this script in a Determined trial."
        raise ValueError(msg)

    model_args, data_args, training_args = parser.parse_dict(hparams)
    training_args.run_name = training_args.run_name + "-" + str(info.trial.trial_id)
    training_args.output_dir = f"output/{training_args.run_name}"

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        training_args.parallel_mode.value == "distributed",
        "fp16" if training_args.fp16 else "bf16" if training_args.bf16 else False,
    )

    logger.info("Data arguments %s", data_args)
    logger.info("Model arguments %s", model_args)
    logger.info("Training arguments %s", training_args)

    # Setup wandb if installed
    if "wandb" in training_args.report_to and find_spec("wandb") is not None:
        os.environ["WANDB_PROJECT"] = "QA_Beavertails_" + model_args.model_name_or_path.split("/")[-1]
        os.environ["WANDB_LOG"] = "false"
        os.environ["WANDB_WATCH"] = "false"
        os.environ["WANDB_NAME"] = training_args.run_name
    else:
        logger.warning("wandb not installed. Install with `pip install wandb` to enable logging to wandb")

    if info.task_type != "NOTEBOOK":
        distributed = det.core.DistributedContext.from_torch_distributed()

        with det.core.init(distributed=distributed) as core_context:
            det_callback = DetCallback(core_context, training_args)
            main(model_args, data_args, training_args, callbacks=[det_callback])
    else:
        main(model_args, data_args, training_args)
