from dataclasses import asdict, dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="meta-llama/Llama-Guard-3-8B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether or not to allow loading models from a remote repository on the Hub"},
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "Device map for model parallelism"},
    )
    lora_r: int = field(
        default=16,
        metadata={"help": "The attention dimension to use for LoRA."},
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "The scaling factor for LoRA."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate for LoRA."},
    )
    lora_modules: list[str] | str = field(
        default="all-linear",
        metadata={"help": "The modules to apply LoRA to."},
    )
    completion_only: bool = field(
        default=True,
        metadata={
            "help": "Whether to only use the generated text for the optimization objective. "
            "If False, the entire text is used for the optimization objective."
        },
    )

    def __str__(self) -> str:
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="PKU-Alignment/Beavertails",
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    config_name: str = field(
        default="multilingual",
        metadata={"help": "The name of the dataset configuration to use."},
    )
    train_split: str = field(
        default="330k_train",
        metadata={"help": "The name of the train split to use."},
    )
    test_split: str = field(
        default="330k_test",
        metadata={"help": "The name of the test split to use."},
    )
    eval_split_ratio: float = field(
        default=0.1,
        metadata={"help": "The ratio of the test split to use for evaluation."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    padding: str = field(init=False)
    include_descriptions: bool = field(
        default=False,
        metadata={"help": "Whether to include descriptions in the prompt. "},
    )

    def __post_init__(self) -> None:
        self.padding = "max_length" if self.pad_to_max_length else "do_not_pad"

    def __str__(self) -> str:
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    def __repr__(self) -> str:
        return self.__str__()
