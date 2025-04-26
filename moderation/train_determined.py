import logging
import os

import determined as det

from determined.transformers import DetCallback
from transformers import TrainingArguments
from transformers.hf_argparser import HfArgumentParser

from arguments import DataArguments, ModelArguments
from train import run, setup_logging


os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

    setup_logging(model_args, data_args, training_args)

    if info.task_type != "NOTEBOOK":
        if training_args.deepspeed:
            distributed = det.core.DistributedContext.from_deepspeed()
        else:
            distributed = det.core.DistributedContext.from_torch_distributed()

        with det.core.init(distributed=distributed) as core_context:
            det_callback = DetCallback(core_context, training_args)
            run(model_args, data_args, training_args, callbacks=[det_callback])
    else:
        run(model_args, data_args, training_args)
