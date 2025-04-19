from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    trust_remote_code: bool = field()
    device_map: str = field()


@dataclass
class DataArguments:
    dataset_name: str = field()
    train_split: str = field()
    test_split: str = field()
    eval_split_ratio: float = field()
    max_seq_length: int = field()
    pad_to_max_length: bool = field()
    template: str = field(default="Question: {question} Answer: {answer}")
    padding: bool | str = field(init=False)

    def __post_init__(self) -> None:
        self.padding = "max_length" if self.pad_to_max_length else False
