from datasets import Dataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING, SYSTEM_MESSAGE_GSM8K, GSM8K_STRING, SYSTEM_MESSAGE_COMPMATH, SYSTEM_MESSAGE_OPENMATH, COMPMATH_STRING
from pathlib import Path
import random
import json

RESAMPLE = False

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Goes up to balance-budget directory
IFEVAL_INPUT_PATH = BASE_DIR / "instruction_following_eval/data/input_data.jsonl"

def random_subset(dataset, n=1000):
    random_subset = random.sample(range(len(dataset)), n)
    return dataset.select(random_subset)

def get_ifeval_test_dataset():
    with open(IFEVAL_INPUT_PATH, "r") as f:
        ifeval_prompts = [json.loads(line) for line in f]

    messages = [
        [
            {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
            {"role": "user", "content": prompt["prompt"]},
        ]
        for prompt in ifeval_prompts
    ]



    prompts = [prompt["prompt"] for prompt in ifeval_prompts]
    dataset = Dataset.from_dict({"messages": messages, "prompt": prompts})
    return dataset


def get_gsm8k_test_dataset(num_prompts=None):
    """Load GSM8K test set with messages, prompt, and reference_answer columns."""
    from datasets import load_dataset
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")

    messages_list = []
    prompts = []
    reference_answers = []

    for row in gsm8k:
        question = row["question"]
        # Reference answer is the number after ####
        answer_text = row["answer"]
        ref_answer = answer_text.split("####")[-1].strip()

        prompt = GSM8K_STRING.format(question=question)
        messages_list.append([
            {"role": "system", "content": SYSTEM_MESSAGE_GSM8K},
            {"role": "user", "content": prompt},
        ])
        prompts.append(prompt)
        reference_answers.append(ref_answer)

    dataset = Dataset.from_dict({
        "messages": messages_list,
        "prompt": prompts,
        "reference_answer": reference_answers,
    })

    if num_prompts is not None:
        dataset = dataset.select(range(min(num_prompts, len(dataset))))

    return dataset


def get_ifbench_test_dataset(num_prompts=None):
    """Load IFBench test set with messages, prompt, instruction_id_list, and kwargs columns."""
    from datasets import load_dataset
    ifbench = load_dataset("allenai/IFBench_test", split="train")

    messages_list = []
    prompts = []
    instruction_id_lists = []
    kwargs_lists = []

    for row in ifbench:
        prompt_text = row["prompt"]
        messages_list.append([
            {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
            {"role": "user", "content": prompt_text},
        ])
        prompts.append(prompt_text)
        instruction_id_lists.append(row["instruction_id_list"])
        kwargs_lists.append(row["kwargs"])

    dataset = Dataset.from_dict({
        "messages": messages_list,
        "prompt": prompts,
        "instruction_id_list": instruction_id_lists,
        "kwargs": kwargs_lists,
    })

    if num_prompts is not None:
        dataset = dataset.select(range(min(num_prompts, len(dataset))))

    return dataset


AMC_NUM_PROMPTS = 40


def get_amc_test_dataset(num_prompts=AMC_NUM_PROMPTS):
    """Load AMC test set with messages, prompt, and reference_answer columns.

    Source is the 2023 AMC 12 (A and B) competition problems with integer answers,
    scored with the same math-verify path as MATH-500. Uses the OPENMATH system
    prompt which asks the model to output $\\boxed{answer}$.
    """
    from datasets import load_dataset
    amc = load_dataset("math-ai/amc23", split="test")

    messages_list = []
    prompts = []
    reference_answers = []

    for row in amc:
        problem = row["question"]
        ref_answer = str(row["answer"])

        prompt = COMPMATH_STRING.format(problem=problem)
        messages_list.append([
            {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
            {"role": "user", "content": prompt},
        ])
        prompts.append(prompt)
        reference_answers.append(ref_answer)

    dataset = Dataset.from_dict({
        "messages": messages_list,
        "prompt": prompts,
        "reference_answer": reference_answers,
    })

    if num_prompts is not None:
        dataset = dataset.select(range(min(num_prompts, len(dataset))))

    return dataset


# def get_amc_test_dataset(num_prompts=AMC_NUM_PROMPTS):
#     """Load AMC test set with messages, prompt, and reference_answer columns.
#
#     Source is AMC 10/12 competition problems with integer answers, scored with the
#     same math-verify path as MATH-500. The source is ordered AMC-10 first then
#     AMC-12, so it is shuffled with a fixed seed to give a level-balanced subset.
#     Uses the OPENMATH system prompt which asks the model to output $\\boxed{answer}$.
#     """
#     from datasets import load_dataset
#     amc = load_dataset("kaggle-aimo/amc_filtered", split="train").shuffle(seed=0)
#
#     messages_list = []
#     prompts = []
#     reference_answers = []
#
#     for row in amc:
#         problem = row["task"]
#         ref_answer = str(row["answer"])
#
#         prompt = COMPMATH_STRING.format(problem=problem)
#         messages_list.append([
#             {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
#             {"role": "user", "content": prompt},
#         ])
#         prompts.append(prompt)
#         reference_answers.append(ref_answer)
#
#     dataset = Dataset.from_dict({
#         "messages": messages_list,
#         "prompt": prompts,
#         "reference_answer": reference_answers,
#     })
#
#     if num_prompts is not None:
#         dataset = dataset.select(range(min(num_prompts, len(dataset))))
#
#     return dataset


def get_math500_test_dataset(num_prompts=None):
    """Load MATH-500 test set with messages, prompt, and reference_answer columns.

    Uses COMPMATH system prompt which asks the model to output $\\boxed{answer}$.
    """
    from datasets import load_dataset
    math500 = load_dataset("HuggingFaceH4/MATH-500", split="test")

    messages_list = []
    prompts = []
    reference_answers = []

    for row in math500:
        problem = row["problem"]
        ref_answer = row["answer"]

        prompt = COMPMATH_STRING.format(problem=problem)
        messages_list.append([
            {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
            {"role": "user", "content": prompt},
        ])
        prompts.append(prompt)
        reference_answers.append(ref_answer)

    dataset = Dataset.from_dict({
        "messages": messages_list,
        "prompt": prompts,
        "reference_answer": reference_answers,
    })

    if num_prompts is not None:
        dataset = dataset.select(range(min(num_prompts, len(dataset))))

    return dataset


def get_minervamath_test_dataset(num_prompts=None):
    """Load Minerva Math test set with messages, prompt, and reference_answer columns.

    Source is math-ai/minervamath (272 quantitative-reasoning problems, mostly
    physics/chemistry/math word problems from Lewkowycz et al. 2022), scored
    with the same math-verify path as MATH-500. Uses the OPENMATH system prompt
    which asks the model to output $\\boxed{answer}$.
    """
    from datasets import load_dataset
    minerva = load_dataset("math-ai/minervamath", split="test")

    messages_list = []
    prompts = []
    reference_answers = []

    for row in minerva:
        problem = row["question"]
        ref_answer = str(row["answer"])

        prompt = COMPMATH_STRING.format(problem=problem)
        messages_list.append([
            {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
            {"role": "user", "content": prompt},
        ])
        prompts.append(prompt)
        reference_answers.append(ref_answer)

    dataset = Dataset.from_dict({
        "messages": messages_list,
        "prompt": prompts,
        "reference_answer": reference_answers,
    })

    if num_prompts is not None:
        dataset = dataset.select(range(min(num_prompts, len(dataset))))

    return dataset


def get_olympiadbench_test_dataset(num_prompts=None):
    """Load OlympiadBench (open-ended text-only math, English competition subset)
    with messages, prompt, and reference_answer columns.

    Source is Hothan/OlympiadBench config OE_TO_maths_en_COMP, filtered to
    single-answer problems (the split is already text-only). The first
    final_answer entry is the reference, scored with the same math-verify path
    as MATH-500. Uses the OPENMATH system prompt which asks the model to output
    $\\boxed{answer}$.
    """
    from datasets import load_dataset
    olympiad = load_dataset("Hothan/OlympiadBench", "OE_TO_maths_en_COMP", split="train")

    messages_list = []
    prompts = []
    reference_answers = []

    for row in olympiad:
        if row["is_multiple_answer"]:
            continue
        if not row["final_answer"]:
            continue
        problem = row["question"]
        ref_answer = str(row["final_answer"][0])

        prompt = COMPMATH_STRING.format(problem=problem)
        messages_list.append([
            {"role": "system", "content": SYSTEM_MESSAGE_OPENMATH},
            {"role": "user", "content": prompt},
        ])
        prompts.append(prompt)
        reference_answers.append(ref_answer)

    dataset = Dataset.from_dict({
        "messages": messages_list,
        "prompt": prompts,
        "reference_answer": reference_answers,
    })

    if num_prompts is not None:
        dataset = dataset.select(range(min(num_prompts, len(dataset))))

    return dataset
