from dataclasses import asdict, dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="microsoft/deberta-v3-large",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    cls_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for the classifier"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Whether or not to allow loading models from a remote repository on the Hub"},
    )
    device_map: str = field(
        default="auto",
        metadata={"help": "Device map for model parallelism"},
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
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    template: str = field(
        default="Question: {question}\n\nAnswer: {answer}",
        metadata={"help": "The prompt template."},
    )
    padding: bool | str = field(init=False)

    def __post_init__(self) -> None:
        self.padding = "max_length" if self.pad_to_max_length else False

    def __str__(self) -> str:
        self_as_dict = asdict(self)
        attrs_as_str = [f"{k}={v},\n" for k, v in sorted(self_as_dict.items())]
        return f"{self.__class__.__name__}(\n{''.join(attrs_as_str)})"

    def __repr__(self) -> str:
        return self.__str__()
