# ABOUTME: LLM-as-judge script to grade completions on a 1-10 scale.
# ABOUTME: Evaluates helpfulness, relevance, accuracy, depth, and creativity.

import re
import argparse
import pandas as pd
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/shougan/projects/aip-fredashi/shougan/balance-budget/tuning/models/llama3-3B-instruct_p@1-data-0_sft-0"

JUDGE_SYSTEM = "You are an impartial judge that scores responses in JSON format."

JUDGE_TEMPLATE = """\
Evaluate the response provided below to determine if it meets the specified constraints related to \
the following prompt. Provide an integer score from 1 to 10, taking into account its helpfulness, \
relevance, accuracy, depth, and creativity. Here are the criteria that you should score: \
1. Helpfulness: Does the response address the user's needs and questions effectively? \
2. Relevance: Is the response directly related to the context of the dialog? \
3. Accuracy: Are the facts and information presented in the response correct? \
4. Depth: Does the response cover the topic thoroughly, with sufficient detail? \
5. Creativity: Is the response original and engaging?

Prompt to Evaluate Against: {user_request}

Response to Evaluate: {completion}

The evaluation must be structured in the following JSON format: {{"Score": "<An integer score from 1 to 10.>"}}\
"""


def extract_user_request(prompt: str) -> str:
    """Pull the raw user message out of the chat-formatted prompt string."""
    # Format: "system\n\n<sys>user\n\n<user>assistant\n\n"
    m = re.search(r"user\n\n(.*?)assistant\n\n", prompt, re.DOTALL)
    if m:
        return m.group(1).strip()
    return prompt.strip()


def build_judge_messages(user_request: str, completion: str) -> list[dict]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_TEMPLATE.format(
            user_request=user_request,
            completion=completion,
        )},
    ]


def _parse_score(text: str) -> int:
    """Extract integer score from judge output, handling markdown fences and preamble."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).replace("```", "")
    text = text.strip("`").strip()

    # JSON object with "Score" key (case-insensitive value, quoted or unquoted)
    m = re.search(r'\{[^}]*["\']?[Ss]core["\']?\s*:\s*"?(\d+)"?[^}]*\}', text)
    if m:
        return max(1, min(10, int(m.group(1))))

    # Plain-text fallbacks: "Score: 8", "score of 8", "8/10", "8 out of 10"
    for pattern in [
        r'[Ss]core[^0-9]{0,20}(\d+)',
        r'(\d+)\s*/\s*10',
        r'(\d+)\s+out\s+of\s+10',
    ]:
        m = re.search(pattern, text)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 10:
                return val

    return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", help="Path to completions parquet file")
    parser.add_argument("--output", help="Output parquet path (default: input with _graded suffix)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    out_path = args.output or args.parquet.replace(".parquet", "_graded.parquet")

    print(f"Loading completions from {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"  {len(df)} rows")

    print(f"Loading judge model from {MODEL_PATH}")
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
    )

    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(max_tokens=128, temperature=0.0)

    print("Building judge prompts...")
    prompts = []
    for _, row in df.iterrows():
        user_req = extract_user_request(row["prompt"])
        messages = build_judge_messages(user_req, row["completion"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    print(f"Running judge on {len(prompts)} completions...")
    outputs = llm.generate(prompts, sampling)

    scores = []
    raw_outputs = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        raw_outputs.append(text)
        score = _parse_score(text)
        scores.append(score)

    df["judge_raw"] = raw_outputs
    df["judge_score"] = scores

    other_cols = [c for c in df.columns if c not in ("judge_raw", "judge_score")]
    df = df[["judge_raw", "judge_score"] + other_cols]

    df.to_parquet(out_path, index=False)
    print(f"Saved graded completions to {out_path}")

    valid = [s for s in scores if s != -1]
    n_fail = scores.count(-1)
    if valid:
        print(f"Mean score: {sum(valid)/len(valid):.2f}  (parse failures: {n_fail}/{len(scores)})")
    else:
        print(f"All {n_fail} outputs failed to parse — check judge_raw column")


if __name__ == "__main__":
    main()
