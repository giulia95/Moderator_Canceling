import random
import re
import warnings

from typing import TYPE_CHECKING, Any, cast, overload

import torch

from datasets import DatasetDict, load_dataset
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers.integrations import WandbCallback

import wandb


Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        BatchEncoding,
        PreTrainedModel,
        Trainer,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )


CATEGORY_SECTION_RE = re.compile("(<BEGIN UNSAFE CONTENT CATEGORIES>).*?(<END UNSAFE CONTENT CATEGORIES>)", re.DOTALL)

CATEGORY_TO_ID = {
    "violence,aiding_and_abetting,incitement": "S14",
    "financial_crime,property_crime,theft": "S6",
    "discrimination,stereotype,injustice": "S4",
    "child_abuse": "S2",
    "non_violent_unethical_behavior": "S9",
    "misinformation_regarding_ethics,laws_and_safety": "S8",
    "privacy_violation": "S10",
    "animal_abuse": "S1",
    "drug_abuse,weapons,banned_substance": "S5",
    "hate_speech,offensive_language": "S7",
    "self_harm": "S11",
    "sexually_explicit,adult_content": "S12",
    "controversial_topics,politics": "S3",
    "terrorism,organized_crime": "S13",
}

CATEGORIES = {
    "S14": "Violence, Aiding and Abetting, Incitement\nThis involves any form of physical harm, threat, or violent behavior towards individuals or groups. Aiding and abetting refers to the act of helping, supporting, or encouraging such violent behaviors or illegal activities. Incitement pertains to the act of provoking or stirring up harmful, violent, or illegal actions.",  # noqa: E501
    "S6": "Financial Crime, Property Crime, Theft\nThis includes all forms of illegal activities that lead to financial loss such as fraud, embezzlement, money laundering, bribery, and other forms of corruption. Property crime and theft involve the act of taking someone's property without their consent.",  # noqa: E501
    "S4": "Discrimination, Stereotype, Injustice\nDiscrimination refers to treating people unfairly or unequally based on their personal characteristics or affiliations such as race, age, gender, religion, etc. Stereotyping involves holding simplified and generalized beliefs about particular groups, which often leads to prejudices and discrimination. Injustice pertains to unfair treatment or violation of the rights of individuals or groups.",  # noqa: E501
    "S2": "Child Abuse\nThis encompasses any form of physical, emotional, or sexual abuse directed toward children. It can also include neglect, exploitation, or any behavior that harms a child or violates their rights. Content that promotes or glorifies such behavior also falls under this category.",  # noqa: E501
    "S9": "Non-Violent Unethical Behavior\nThis refers to any behavior that is morally or ethically wrong, but does not involve violence. Examples can include dishonest behaviors like lying, cheating, deceiving, spreading rumors, betraying trust, or any other form of unethical conduct.",  # noqa: E501
    "S8": "Misinformation Regarding ethics, laws, and safety\nThis category involves the spreading of incorrect or misleading information related to ethical issues, laws, regulations, or any matter related to safety. Misinformation can cause harm by influencing people's beliefs or actions negatively.",  # noqa: E501
    "S10": "Privacy Violation\nThis refers to actions that invade an individual's privacy, such as unauthorized access, use, disclosure, or collection of personal data. It also includes stalking, doxing, or any other form of intrusion into someone's personal life without their consent.",  # noqa: E501
    "S1": "Animal Abuse\nThis involves any form of cruelty or harm inflicted on animals, including physical abuse, neglect, or any behavior that causes distress or harm to an animal. It also includes content that promotes such behavior.",  # noqa: E501
    "S5": "Drug Abuse, Weapons, Banned Substance\nThis category includes the misuse or abuse of illegal drugs, prescription drugs, or any other banned substances. It also involves the unauthorized or illegal possession, use, or trade of weapons.",  # noqa: E501
    "S7": "Hate Speech, Offensive Language\nThis category encompasses any form of communication or content that expresses intense hostility or violence towards an individual or group based on aspects like race, religion, ethnicity, nationality, gender, sexual orientation, or any other defining characteristics. This also includes language that is considered vulgar, crude, or offensive.",  # noqa: E501
    "S11": "Self-Harm\nThis includes any form of self-inflicted harm, whether physical or emotional, or any content that promotes, glorifies, or encourages self-harm. This can also include discussions of suicidal thoughts or plans.",  # noqa: E501
    "S12": "Sexually Explicit, Adult Content\nThis category includes any content that depicts explicit sexual activities or adult themes that are considered inappropriate or offensive. This can include pornography, explicit sexual text or messages, and any other form of sexually explicit material.",  # noqa: E501
    "S3": "Controversial Topics, Politics\nThis refers to discussions or statements related to controversial topics, political ideologies, or sensitive issues that could lead to disagreement or conflict. These topics can be divisive and could potentially cause harm if not handled respectfully and thoughtfully.",  # noqa: E501
    "S13": "Terrorism, Organized Crime\nThis pertains to any form of content or action related to terrorism or organized crime, including endorsing or promoting terrorist activities, participating in organized criminal activities, or spreading propaganda for such groups.",  # noqa: E501
}


def replace_categories(chat_template: str, descriptions: bool = True, try_match_order: bool = False) -> str:
    """Update chat template with new categories"""
    category_list = "\n"

    if try_match_order:
        categories = {f"S{i + 1}": c for i, c in enumerate(CATEGORIES.values())}
    else:
        categories = dict(sorted(CATEGORIES.items(), key=lambda x: int(x[0][1:])))

    for k, v in categories.items():
        if descriptions:
            category_list += k + ": " + v + "\n"
        else:
            category_list += k + ": " + v.split("\n")[0] + ".\n"

    new_chat_template = CATEGORY_SECTION_RE.sub(rf"\1{category_list}\2", chat_template)

    if chat_template == new_chat_template:
        warnings.warn("Chat template remained unchanged, please check the regex or the categories.")

    return new_chat_template


def update_chat_template(tokenizer: PreTrainedTokenizer, descriptions: bool = True) -> None:
    if tokenizer.chat_template is None:
        warnings.warn("Chat template not found, skipping update.")
        return

    tokenizer.chat_template = replace_categories(str(tokenizer.chat_template), descriptions, try_match_order=False)


def prepare_output(example: dict, categories: list[str], shuffle_categories: bool = False) -> str:
    output_prompt = "\n\n"

    if example["is_safe"] is True:
        output_prompt += "safe"
    else:
        output_prompt += "unsafe\n"
        category_list = [CATEGORY_TO_ID[k] for k in categories if example["category"][k] is True]

        if shuffle_categories:
            # shuffle categories to avoid bias?
            random.shuffle(category_list)

        output_prompt += ",".join(category_list)

    return output_prompt


def prepare_input(example: dict, model_name: str, tokenizer: Tokenizer) -> str:
    if model_name == "meta-llama/Llama-Guard-3-1B":
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": example["prompt"]}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["response"]}],
            },
        ]
    elif model_name == "meta-llama/Llama-Guard-3-8B":
        conversation = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
    else:
        msg = f"Model {model_name} not supported"
        raise ValueError(msg)

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)

    return str(prompt)


def format_prompts(
    example: dict, model_name_or_path: str, tokenizer: Tokenizer, categories: list[str]
) -> dict[str, str]:
    text = (
        prepare_input(example, model_name_or_path, tokenizer)
        + prepare_output(example, categories)
        + str(tokenizer.eos_token)
    )

    return {"text": text}


# def format_prompts(
#     example: dict,
#     model_name_or_path: str,
#     tokenizer: Tokenizer,
#     categories: list[str],
# ) -> dict[str, str]:
#     return {
#         "prompt": prepare_input(example, model_name_or_path, tokenizer),
#         "completion": prepare_output(example, categories),
#     }


def category_map_to_list(categories: dict[str, bool]) -> list[str]:
    return [CATEGORY_TO_ID[c] for c, v in categories.items() if v]


@overload
def load_qa_dataset(  # type: ignore
    dataset_name: str,
    split: None = None,
    config_name: str | None = None,
) -> tuple["DatasetDict", list[str]]: ...


@overload
def load_qa_dataset(  # type: ignore
    dataset_name: str,
    split: str,
    config_name: str | None = None,
) -> tuple["Dataset", list[str]]: ...


def load_qa_dataset(
    dataset_name: str,
    split: str | None = None,
    config_name: str | None = None,
) -> tuple["Dataset | DatasetDict", list[str]]:
    dataset = load_dataset(dataset_name, name=config_name, split=split)
    categories: list[dict[str, bool]]

    if isinstance(dataset, DatasetDict):
        splits = list(dataset.keys())
        categories = dataset[splits[0]]["category"]
    else:
        dataset = cast("Dataset", dataset)
        categories = dataset["category"]

    labels = list(categories[0].keys())

    return dataset, labels


class LLMSampleCB(WandbCallback):
    tokenizer: "PreTrainedTokenizer | PreTrainedTokenizerFast"
    model: "PreTrainedModel"
    gen_config: GenerationConfig
    freq: int

    def __init__(
        self,
        trainer: "Trainer",
        test_dataset: "Dataset",
        num_samples: int = 100,
        max_new_tokens: int = 100,
        freq: int = 200,
    ) -> None:
        "A CallBack to log samples a wandb.Table during training"
        super().__init__()
        self.sample_dataset = test_dataset.select(range(num_samples))

        self.model = cast("PreTrainedModel", trainer.model)

        if trainer.tokenizer is None:
            msg = "Trainer tokenizer is None"
            raise ValueError(msg)
        self.tokenizer = cast("Tokenizer", trainer.tokenizer)

        self.freq = freq
        self.gen_config = GenerationConfig(do_sample=False, pad_token_id=0, max_new_tokens=max_new_tokens)

    def generate(self, prompt: str, response: str) -> tuple[str, int, int, str]:
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        input_prompt = self.tokenizer.apply_chat_template(chat, tokenize=False)
        input_prompt = cast("str", input_prompt)

        tokenized_prompt = self.tokenizer(input_prompt, return_tensors="pt")
        tokenized_prompt = cast("BatchEncoding", tokenized_prompt)
        inputs = tokenized_prompt.to(self.model.device)  # type: ignore

        with torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.gen_config)
            generated_ids = output[0][len(inputs.input_ids[0]) :]

        return (
            input_prompt,
            len(inputs.input_ids[0]),
            len(generated_ids),
            self.tokenizer.decode(generated_ids, skip_special_tokens=False),
        )

    def samples_table(self, examples: list[dict]) -> wandb.Table:
        "Create a wandb.Table to store the generations"
        records_table = wandb.Table(
            columns=[
                "prompt",
                "response",
                "chat_prompt",
                "output",
                "input_tokens",
                "output_tokens",
                "is_safe",
                "categories",
            ]
        )
        for example in tqdm(examples, leave=False):
            prompt = example["prompt"]
            response = example["response"]
            is_safe = example["is_safe"]
            category_list = ", ".join(category_map_to_list(example["category"]))
            chat_prompt, in_len, out_len, generation = self.generate(prompt, response)
            records_table.add_data(prompt, response, chat_prompt, generation, in_len, out_len, is_safe, category_list)
        return records_table

    def on_evaluate(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs: Any
    ) -> None:
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)

        if state.global_step % self.freq == 0:
            records_table = self.samples_table(self.sample_dataset.to_list())
            self._wandb.log({"sample_predictions": records_table})
