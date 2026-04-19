# IFBench Eval + IF-RLVR Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add IFBench as an OOD instruction-following eval benchmark and IF-RLVR as a GRPO training data source.

**Architecture:** Vendor two constraint packages (IFBench eval checkers from `allenai/IFBench`, IFEvalG training checkers from `allenai/open-instruct`). Refactor the existing `evaluation_lib.py` to accept an injectable constraint registry. Add `IFBenchStrategy`, `IfrlvrRLVR` dataset class, `ifrlvr_reward_func`, and wire everything into the pipeline.

**Tech Stack:** Python, HuggingFace datasets, NLTK, emoji, langdetect, syllapy

---

### Task 1: Install dependencies and download NLTK data

**Files:**
- Modify: `/project/6105902/shougan/balance-budget/instruction_following_eval/requirements.txt`

- [ ] **Step 1: Install pip dependencies**

```bash
pip install langdetect syllapy emoji
```

- [ ] **Step 2: Download NLTK data**

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "import langdetect; import syllapy; import emoji; import nltk; print('All imports OK')"
```
Expected: `All imports OK`

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "deps: add langdetect, syllapy, emoji for IFBench constraint checkers"
```

---

### Task 2: Vendor IFBench eval constraints

**Files:**
- Create: `/project/6105902/shougan/balance-budget/ifbench_eval/__init__.py`
- Create: `/project/6105902/shougan/balance-budget/ifbench_eval/instructions.py`
- Create: `/project/6105902/shougan/balance-budget/ifbench_eval/instructions_registry.py`
- Create: `/project/6105902/shougan/balance-budget/ifbench_eval/instructions_util.py`

- [ ] **Step 1: Clone IFBench repo to a temp directory**

```bash
cd /tmp && git clone https://github.com/allenai/IFBench.git ifbench_clone
```

- [ ] **Step 2: Create ifbench_eval package directory**

```bash
mkdir -p /project/6105902/shougan/balance-budget/ifbench_eval
```

- [ ] **Step 3: Copy constraint checker files**

```bash
cp /tmp/ifbench_clone/instructions.py /project/6105902/shougan/balance-budget/ifbench_eval/
cp /tmp/ifbench_clone/instructions_registry.py /project/6105902/shougan/balance-budget/ifbench_eval/
cp /tmp/ifbench_clone/instructions_util.py /project/6105902/shougan/balance-budget/ifbench_eval/
```

- [ ] **Step 4: Create `__init__.py`**

Write `/project/6105902/shougan/balance-budget/ifbench_eval/__init__.py`:
```python
# ABOUTME: IFBench OOD instruction-following constraint checkers (58 constraints).
# ABOUTME: Vendored from https://github.com/allenai/IFBench
```

- [ ] **Step 5: Fix imports to use package-relative paths**

In `ifbench_eval/instructions.py`, find all bare imports and prefix with `ifbench_eval.`:
```python
# Change:  import instructions_util
# To:      from ifbench_eval import instructions_util
```

In `ifbench_eval/instructions_registry.py`:
```python
# Change:  from instructions import ...
#          or: import instructions
# To:      from ifbench_eval import instructions
#          or: from ifbench_eval.instructions import ...
```

- [ ] **Step 6: Verify the registry loads**

```bash
python -c "from ifbench_eval.instructions_registry import INSTRUCTION_DICT; print(f'{len(INSTRUCTION_DICT)} constraints loaded')"
```
Expected: `58 constraints loaded` (approximately)

- [ ] **Step 7: Clean up temp clone**

```bash
rm -rf /tmp/ifbench_clone
```

- [ ] **Step 8: Commit**

```bash
git add ifbench_eval/ && git commit -m "vendor: add IFBench eval constraint checkers (58 OOD constraints)"
```

---

### Task 3: Vendor IFEvalG training constraints

**Files:**
- Create: `/project/6105902/shougan/balance-budget/ifrlvr/__init__.py`
- Create: `/project/6105902/shougan/balance-budget/ifrlvr/instructions.py`
- Create: `/project/6105902/shougan/balance-budget/ifrlvr/instructions_registry.py`

- [ ] **Step 1: Clone open-instruct repo to a temp directory**

```bash
cd /tmp && git clone --depth 1 https://github.com/allenai/open-instruct.git open_instruct_clone
```

- [ ] **Step 2: Create ifrlvr package directory**

```bash
mkdir -p /project/6105902/shougan/balance-budget/ifrlvr
```

- [ ] **Step 3: Copy IFEvalG constraint checker files**

```bash
cp /tmp/open_instruct_clone/open_instruct/IFEvalG/instructions.py /project/6105902/shougan/balance-budget/ifrlvr/
cp /tmp/open_instruct_clone/open_instruct/IFEvalG/instructions_registry.py /project/6105902/shougan/balance-budget/ifrlvr/
```

Note: IFEvalG's `instructions.py` may import from `instructions_util.py` — check and also copy if needed:
```bash
grep -n "import instructions_util" /tmp/open_instruct_clone/open_instruct/IFEvalG/instructions.py && \
  cp /tmp/open_instruct_clone/open_instruct/IFEvalG/instructions_util.py /project/6105902/shougan/balance-budget/ifrlvr/ || \
  echo "No instructions_util import found"
```

- [ ] **Step 4: Create `__init__.py`**

Write `/project/6105902/shougan/balance-budget/ifrlvr/__init__.py`:
```python
# ABOUTME: IFEvalG training constraint checkers for IF-RLVR (54 constraints).
# ABOUTME: Vendored from https://github.com/allenai/open-instruct (open_instruct/IFEvalG/)
```

- [ ] **Step 5: Fix imports to use package-relative paths**

In `ifrlvr/instructions.py`, find all bare imports and prefix with `ifrlvr.`:
```python
# Change:  import instructions_util
#          or: from open_instruct.IFEvalG import instructions_util
# To:      from ifrlvr import instructions_util
```

In `ifrlvr/instructions_registry.py`:
```python
# Change:  from open_instruct.IFEvalG.instructions import ...
#          or: from open_instruct.IFEvalG import instructions
# To:      from ifrlvr import instructions
#          or: from ifrlvr.instructions import ...
```

- [ ] **Step 6: Verify the registry loads**

```bash
python -c "from ifrlvr.instructions_registry import INSTRUCTION_DICT; print(f'{len(INSTRUCTION_DICT)} constraints loaded')"
```
Expected: `54 constraints loaded` (approximately — 25 original IFEval + 29 new)

- [ ] **Step 7: Clean up temp clone**

```bash
rm -rf /tmp/open_instruct_clone
```

- [ ] **Step 8: Commit**

```bash
git add ifrlvr/ && git commit -m "vendor: add IFEvalG training constraint checkers (54 constraints)"
```

---

### Task 4: Refactor evaluation_lib.py — injectable registry + null-filtering

**Files:**
- Modify: `/project/6105902/shougan/balance-budget/instruction_following_eval/evaluation_lib.py`
- Test: `/project/6105902/shougan/balance-budget/tests/test_evaluation_lib_refactor.py`

- [ ] **Step 1: Write failing test for injectable registry**

Write `/project/6105902/shougan/balance-budget/tests/test_evaluation_lib_refactor.py`:
```python
# ABOUTME: Tests for evaluation_lib refactoring (injectable registry + null-filtering).
# ABOUTME: Ensures backward compatibility and new IFBench support.

from instruction_following_eval.evaluation_lib import (
    test_instruction_following_strict,
    test_instruction_following_loose,
    InputExample,
)


class FakeInstruction:
    """Minimal constraint checker for testing."""
    def __init__(self, instruction_id):
        self.id = instruction_id
        self._keyword = None

    def build_description(self, keyword=None):
        self._keyword = keyword

    def get_instruction_args(self):
        return {"keyword": self._keyword}

    def get_instruction_args_keys(self):
        return ["keyword"]

    def check_following(self, value):
        return self._keyword is not None and self._keyword in value


FAKE_REGISTRY = {"test:keyword_check": FakeInstruction}


def test_strict_accepts_custom_instruction_dict():
    inp = InputExample(
        key=0,
        instruction_id_list=["test:keyword_check"],
        prompt="Say hello with the keyword 'banana'.",
        kwargs=[{"keyword": "banana"}],
    )
    result = test_instruction_following_strict(
        inp,
        {inp.prompt: "Hello banana world!"},
        instruction_dict=FAKE_REGISTRY,
    )
    assert result.follow_all_instructions is True
    assert result.follow_instruction_list == [True]


def test_loose_accepts_custom_instruction_dict():
    inp = InputExample(
        key=0,
        instruction_id_list=["test:keyword_check"],
        prompt="Say hello with the keyword 'banana'.",
        kwargs=[{"keyword": "banana"}],
    )
    result = test_instruction_following_loose(
        inp,
        {inp.prompt: "Hello banana world!"},
        instruction_dict=FAKE_REGISTRY,
    )
    assert result.follow_all_instructions is True
    assert result.follow_instruction_list == [True]


def test_strict_defaults_to_builtin_registry():
    """When no instruction_dict is passed, the existing IFEval registry is used."""
    inp = InputExample(
        key=0,
        instruction_id_list=["keywords:existence"],
        prompt="Include the word 'hello' in your response.",
        kwargs=[{"keywords": ["hello"]}],
    )
    result = test_instruction_following_strict(
        inp,
        {inp.prompt: "hello world"},
    )
    assert result.follow_all_instructions is True


def test_strict_filters_none_kwargs():
    """None values in kwargs should be filtered out before build_description."""
    inp = InputExample(
        key=0,
        instruction_id_list=["test:keyword_check"],
        prompt="test",
        kwargs=[{"keyword": "banana", "irrelevant_param": None}],
    )
    result = test_instruction_following_strict(
        inp,
        {inp.prompt: "banana"},
        instruction_dict=FAKE_REGISTRY,
    )
    assert result.follow_all_instructions is True


def test_loose_filters_none_kwargs():
    """None values in kwargs should be filtered out before build_description."""
    inp = InputExample(
        key=0,
        instruction_id_list=["test:keyword_check"],
        prompt="test",
        kwargs=[{"keyword": "banana", "extra": None, "another": None}],
    )
    result = test_instruction_following_loose(
        inp,
        {inp.prompt: "banana"},
        instruction_dict=FAKE_REGISTRY,
    )
    assert result.follow_all_instructions is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_evaluation_lib_refactor.py -v
```
Expected: FAIL — `test_instruction_following_strict()` doesn't accept `instruction_dict`

- [ ] **Step 3: Modify `test_instruction_following_strict`**

In `/project/6105902/shougan/balance-budget/instruction_following_eval/evaluation_lib.py`, change the function signature and body:

```python
def test_instruction_following_strict(
    inp,
    prompt_to_response,
    instruction_dict=None,
):
  """Tests response to see if instrutions are followed."""
  if instruction_dict is None:
    instruction_dict = instructions_registry.INSTRUCTION_DICT

  response = prompt_to_response[inp.prompt]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instruction_dict[instruction_id]
    instruction = instruction_cls(instruction_id)

    kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
    instruction.build_description(**kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    if response.strip() and instruction.check_following(response):
      is_following_list.append(True)
    else:
      is_following_list.append(False)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )
```

- [ ] **Step 4: Modify `test_instruction_following_loose`**

Same changes: add `instruction_dict=None` parameter, default to `instructions_registry.INSTRUCTION_DICT`, use it for lookup, and add null-filtering:

```python
def test_instruction_following_loose(
    inp,
    prompt_to_response,
    instruction_dict=None,
):
  """Tests response for an upper bound for following instructions."""
  if instruction_dict is None:
    instruction_dict = instructions_registry.INSTRUCTION_DICT

  response = prompt_to_response[inp.prompt]
  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
  revised_response_remove_first = response_remove_first.replace("*", "")
  revised_response_remove_last = response_remove_last.replace("*", "")
  revised_response_remove_both = response_remove_both.replace("*", "")
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instruction_dict[instruction_id]
    instruction = instruction_cls(instruction_id)

    kwargs = {k: v for k, v in inp.kwargs[index].items() if v is not None}
    instruction.build_description(**kwargs)
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list,
  )
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_evaluation_lib_refactor.py -v
```
Expected: ALL PASS

- [ ] **Step 6: Run existing tests to verify backward compatibility**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_eval_strategy.py tests/test_reward_functions.py tests/test_ifeval_evaluation_harness.py -v
```
Expected: ALL PASS (no regressions)

- [ ] **Step 7: Commit**

```bash
git add instruction_following_eval/evaluation_lib.py tests/test_evaluation_lib_refactor.py && git commit -m "refactor: make evaluation_lib instruction_dict injectable + add kwargs null-filtering"
```

---

### Task 5: Add IFBench test data loader

**Files:**
- Modify: `/project/6105902/shougan/balance-budget/tuning/data/test_dataset.py`
- Test: `/project/6105902/shougan/balance-budget/tests/test_ifbench_test_dataset.py`

- [ ] **Step 1: Write failing test**

Write `/project/6105902/shougan/balance-budget/tests/test_ifbench_test_dataset.py`:
```python
# ABOUTME: Tests for IFBench test dataset loader.
# ABOUTME: Validates format, columns, and constraint metadata.

from tuning.data.test_dataset import get_ifbench_test_dataset


def test_ifbench_test_dataset_has_required_columns():
    dataset = get_ifbench_test_dataset(num_prompts=5)
    assert "messages" in dataset.column_names
    assert "prompt" in dataset.column_names
    assert "instruction_id_list" in dataset.column_names
    assert "kwargs" in dataset.column_names


def test_ifbench_test_dataset_messages_format():
    dataset = get_ifbench_test_dataset(num_prompts=5)
    row = dataset[0]
    messages = row["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert len(messages[1]["content"]) > 0


def test_ifbench_test_dataset_prompt_matches_message():
    dataset = get_ifbench_test_dataset(num_prompts=5)
    row = dataset[0]
    assert row["prompt"] == row["messages"][1]["content"]


def test_ifbench_test_dataset_has_constraint_metadata():
    dataset = get_ifbench_test_dataset(num_prompts=5)
    row = dataset[0]
    assert isinstance(row["instruction_id_list"], list)
    assert len(row["instruction_id_list"]) >= 1
    assert isinstance(row["kwargs"], list)
    assert len(row["kwargs"]) == len(row["instruction_id_list"])


def test_ifbench_test_dataset_num_prompts_limits():
    dataset = get_ifbench_test_dataset(num_prompts=3)
    assert len(dataset) == 3


def test_ifbench_test_dataset_full_size():
    dataset = get_ifbench_test_dataset()
    assert len(dataset) >= 72  # at least 72 from GitHub, up to 300 from HF
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_ifbench_test_dataset.py -v
```
Expected: FAIL — `get_ifbench_test_dataset` not found

- [ ] **Step 3: Implement `get_ifbench_test_dataset` in test_dataset.py**

Add to `/project/6105902/shougan/balance-budget/tuning/data/test_dataset.py`:

```python
def get_ifbench_test_dataset(num_prompts=None):
    """Load IFBench test set with messages, prompt, instruction_id_list, and kwargs columns.

    Source: allenai/IFBench_test on HuggingFace (300 OOD instruction-following prompts).
    """
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_ifbench_test_dataset.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tuning/data/test_dataset.py tests/test_ifbench_test_dataset.py && git commit -m "feat: add IFBench test dataset loader (allenai/IFBench_test, 300 prompts)"
```

---

### Task 6: Add IFBenchStrategy eval strategy

**Files:**
- Modify: `/project/6105902/shougan/balance-budget/tuning/training/eval_strategy.py`
- Test: `/project/6105902/shougan/balance-budget/tests/test_ifbench_eval_strategy.py`

- [ ] **Step 1: Write failing test**

Write `/project/6105902/shougan/balance-budget/tests/test_ifbench_eval_strategy.py`:
```python
# ABOUTME: Tests for IFBenchStrategy eval implementation.
# ABOUTME: Validates interface conformance, scoring, and W&B metrics.

from unittest.mock import patch, MagicMock
from datasets import Dataset


def test_ifbench_strategy_implements_interface():
    from tuning.training.eval_strategy import IFBenchStrategy, EvalStrategy
    assert issubclass(IFBenchStrategy, EvalStrategy)

    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert hasattr(strategy, "get_test_messages")
        assert hasattr(strategy, "score_responses")
        assert hasattr(strategy, "stopping_metric")
        assert hasattr(strategy, "wandb_metrics")
        assert hasattr(strategy, "label_prefix")


def test_ifbench_strategy_id():
    from tuning.training.eval_strategy import IFBenchStrategy
    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert strategy.id == "ifbench"


def test_ifbench_stopping_metric():
    from tuning.training.eval_strategy import IFBenchStrategy
    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1, 5], n_samples=5, num_prompts=1)
        assert strategy.stopping_metric() == "pass_at_1"


def test_ifbench_label_prefix():
    from tuning.training.eval_strategy import IFBenchStrategy
    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1], n_samples=1, num_prompts=1)
        assert strategy.label_prefix == "ifbench-p@1"


def test_ifbench_wandb_metrics():
    from tuning.training.eval_strategy import IFBenchStrategy
    with patch("tuning.training.eval_strategy.get_ifbench_test_dataset") as mock_dataset:
        mock_dataset.return_value = Dataset.from_dict({
            "messages": [[{"role": "user", "content": "test"}]],
            "prompt": ["test"],
            "instruction_id_list": [["count:word_count_range"]],
            "kwargs": [[{"min_words": 5, "max_words": 10}]],
        })
        strategy = IFBenchStrategy(k_values=[1], n_samples=1, num_prompts=1)
        scores = {"pass_at_1": 0.42, "pass_at_1_prompt": 0.35, "avg_response_length_tokens": 100.0, "num_prompts_evaluated": 10}
        wandb_dict = strategy.wandb_metrics(scores)
        assert "eval/ifbench_pass_at_1" in wandb_dict
        assert "eval/ifbench_pass_at_1_prompt" in wandb_dict
        assert "eval/ifbench_avg_response_length_tokens" in wandb_dict
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_ifbench_eval_strategy.py -v
```
Expected: FAIL — `IFBenchStrategy` not found

- [ ] **Step 3: Add import for `get_ifbench_test_dataset` at top of eval_strategy.py**

In `/project/6105902/shougan/balance-budget/tuning/training/eval_strategy.py`, update the import line:

```python
from tuning.data.test_dataset import get_ifeval_test_dataset, get_gsm8k_test_dataset, get_math500_test_dataset, get_ifbench_test_dataset
```

- [ ] **Step 4: Implement `IFBenchStrategy` at end of eval_strategy.py**

Add to `/project/6105902/shougan/balance-budget/tuning/training/eval_strategy.py` after `MATH500EvalStrategy`:

```python
class IFBenchStrategy(EvalStrategy):
    """IFBench OOD instruction-following evaluation using pass@k scoring.

    Uses the shared evaluation_lib harness with the IFBench constraint registry.
    """

    def __init__(self, k_values=None, n_samples=1, num_prompts=None, strict=True):
        k_values = k_values or [1]
        self.k_values = k_values
        self.stopping_k = k_values[0]
        self._n_samples = n_samples
        self.strict = strict

        self.test_dataset = get_ifbench_test_dataset(num_prompts=num_prompts)

        self.inputs_map = {}
        for i in range(len(self.test_dataset)):
            row = self.test_dataset[i]
            self.inputs_map[row["prompt"]] = evaluation_lib.InputExample(
                key=i,
                instruction_id_list=row["instruction_id_list"],
                prompt=row["prompt"],
                kwargs=row["kwargs"],
            )

        print(f"[IFBenchStrategy] k_values={k_values}, n_samples={n_samples}, "
              f"strict={strict}, num_prompts={len(self.test_dataset)}")

    @property
    def n_samples(self) -> int:
        return self._n_samples

    @property
    def id(self) -> str:
        return "ifbench"

    def get_test_messages(self) -> List[List[dict]]:
        return list(self.test_dataset["messages"])

    def get_test_prompts(self) -> List[str]:
        return list(self.test_dataset["prompt"])

    def score_responses(self, results: List[Dict], tokenizer) -> Dict[str, float]:
        from ifbench_eval.instructions_registry import INSTRUCTION_DICT as IFBENCH_DICT

        all_prompt_results = []
        all_instruction_scores = {k: [] for k in self.k_values}
        response_token_lengths = []

        for item in results:
            prompt = item["prompt"]
            responses = item["responses"]

            encoded_batch = tokenizer(
                responses, add_special_tokens=False, padding=False, truncation=False,
            )
            response_token_lengths.extend(len(ids) for ids in encoded_batch["input_ids"])

            eval_input = self.inputs_map[prompt]

            eval_fn = evaluation_lib.test_instruction_following_strict if self.strict else evaluation_lib.test_instruction_following_loose
            eval_results = [eval_fn(eval_input, {prompt: r}, instruction_dict=IFBENCH_DICT) for r in responses]
            prompt_results = [er.follow_all_instructions for er in eval_results]
            all_prompt_results.append(prompt_results)

            for k in self.k_values:
                n = len(responses)
                num_instructions = len(eval_results[0].follow_instruction_list)
                pk_list = []
                for c_idx in range(num_instructions):
                    correct_count = sum(eval_results[r_idx].follow_instruction_list[c_idx] for r_idx in range(n))
                    pk_list.append(pass_at_k(n, correct_count, k))
                all_instruction_scores[k].append(sum(pk_list) / len(pk_list))

            item["per_response_correct"] = prompt_results
            item["per_response_instructions"] = [list(er.follow_instruction_list) for er in eval_results]

        scores = {}

        for k in self.k_values:
            scores[f"pass_at_{k}"] = np.mean(all_instruction_scores[k])

        for k in self.k_values:
            prompt_scores = [pass_at_k(len(r), sum(r), k) for r in all_prompt_results]
            scores[f"pass_at_{k}_prompt"] = np.mean(prompt_scores)

        scores["num_prompts_evaluated"] = len(results)
        scores["avg_response_length_tokens"] = (
            float(np.mean(response_token_lengths)) if response_token_lengths else 0.0
        )
        return scores

    def stopping_metric(self) -> str:
        return f"pass_at_{self.stopping_k}"

    @property
    def label_prefix(self) -> str:
        return f"ifbench-p@{self.stopping_k}"

    def wandb_metrics(self, scores: Dict[str, float]) -> Dict[str, float]:
        metrics = {}
        for k in self.k_values:
            metrics[f"eval/ifbench_pass_at_{k}"] = scores[f"pass_at_{k}"]
            metrics[f"eval/ifbench_pass_at_{k}_prompt"] = scores[f"pass_at_{k}_prompt"]
        metrics["eval/ifbench_avg_response_length_tokens"] = scores.get("avg_response_length_tokens", 0.0)
        return metrics
```

- [ ] **Step 5: Update ABOUTME comment at top of eval_strategy.py**

```python
# ABOUTME: ABC for eval strategies injected into the generation eval callback.
# ABOUTME: Includes IFEval, GSM8K, MATH-500, and IFBench pass@k implementations.
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_ifbench_eval_strategy.py tests/test_eval_strategy.py -v
```
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add tuning/training/eval_strategy.py tests/test_ifbench_eval_strategy.py && git commit -m "feat: add IFBenchStrategy for OOD instruction-following evaluation"
```

---

### Task 7: Add IF-RLVR dataset loader

**Files:**
- Create: `/project/6105902/shougan/balance-budget/tuning/data/ifrlvr_rlvr.py`
- Test: `/project/6105902/shougan/balance-budget/tests/test_ifrlvr_dataset.py`

- [ ] **Step 1: Write failing test**

Write `/project/6105902/shougan/balance-budget/tests/test_ifrlvr_dataset.py`:
```python
# ABOUTME: Tests for IF-RLVR dataset loader.
# ABOUTME: Validates format, columns, and ground_truth preservation.

from tuning.data.ifrlvr_rlvr import IfrlvrRLVR


def test_ifrlvr_format_produces_prompt_and_ground_truth():
    ds = IfrlvrRLVR()
    ds.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()

    train = dataset["train"]
    assert "prompt" in train.column_names
    assert "ground_truth" in train.column_names

    row = train[0]
    assert isinstance(row["prompt"], list)
    assert len(row["prompt"]) == 2
    assert row["prompt"][0]["role"] == "system"
    assert row["prompt"][1]["role"] == "user"
    assert isinstance(row["ground_truth"], str)
    assert len(row["ground_truth"]) > 0


def test_ifrlvr_ground_truth_is_parseable():
    import ast
    ds = IfrlvrRLVR()
    ds.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()

    row = dataset["train"][0]
    parsed = ast.literal_eval(row["ground_truth"])
    assert isinstance(parsed, list)
    assert len(parsed) >= 1
    assert "instruction_id" in parsed[0]
    assert "kwargs" in parsed[0]


def test_ifrlvr_deduplicates_prompts():
    ds = IfrlvrRLVR()
    ds.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ds.format_dataset()
    dataset = ds.get_dataset()
    train = dataset["train"]

    prompt_texts = [p[1]["content"] for p in train["prompt"]]
    assert len(prompt_texts) == len(set(prompt_texts)), "IF-RLVR dataset should have unique prompts"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_ifrlvr_dataset.py -v
```
Expected: FAIL — `IfrlvrRLVR` not found

- [ ] **Step 3: Implement IfrlvrRLVR**

Write `/project/6105902/shougan/balance-budget/tuning/data/ifrlvr_rlvr.py`:
```python
# ABOUTME: IF-RLVR dataset for GRPO training with verifiable instruction-following rewards.
# ABOUTME: Loads allenai/IF_multi_constraints_upto5 (95k prompts, up to 5 constraints each).

from datasets import Dataset
from tuning.data.hf_dataset import HFDataset
from tuning.data.config import SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING


class IfrlvrRLVR(HFDataset):
    def __init__(self):
        super().__init__(dataset_name="ifrlvr")

    def _get_rows(self, dataset):
        seen = set()
        rows = []
        for i in range(len(dataset)):
            row = dataset[i]
            prompt_text = row["messages"][0]["content"]
            if prompt_text in seen:
                continue
            seen.add(prompt_text)
            rows.append({
                "prompt": [
                    {"role": "system", "content": SYSTEM_MESSAGE_INSTRUCTION_FOLLOWING},
                    {"role": "user", "content": prompt_text},
                ],
                "ground_truth": row["ground_truth"],
            })
        return rows

    def format_dataset(self):
        rows = self._get_rows(self._dataset)
        formatted_dataset = Dataset.from_list(rows).train_test_split(test_size=100, shuffle=False)
        print(f"IF-RLVR Dataset - {formatted_dataset}")
        print(f"Example row - {formatted_dataset['train'][0]}")
        self._dataset = formatted_dataset


if __name__ == "__main__":
    ifrlvr = IfrlvrRLVR()
    ifrlvr.load_from_huggingface("allenai/IF_multi_constraints_upto5", split="train")
    ifrlvr.format_dataset()
    ifrlvr.clear_old_datasets(prefix="rlvr-ifrlvr")
    ifrlvr.save_dataset_to_disk(save_name="rlvr-ifrlvr")
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_ifrlvr_dataset.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Download and save dataset to disk**

```bash
cd /project/6105902/shougan/balance-budget && python tuning/data/ifrlvr_rlvr.py
```
Expected: Prints dataset info showing ~95k train rows, saves to `{DATASETS_DIR}/rlvr-ifrlvr`

- [ ] **Step 6: Commit**

```bash
git add tuning/data/ifrlvr_rlvr.py tests/test_ifrlvr_dataset.py && git commit -m "feat: add IF-RLVR dataset loader (allenai/IF_multi_constraints_upto5, 95k prompts)"
```

---

### Task 8: Add IF-RLVR reward function

**Files:**
- Modify: `/project/6105902/shougan/balance-budget/tuning/training/reward_functions.py`
- Test: `/project/6105902/shougan/balance-budget/tests/test_ifrlvr_reward.py`

- [ ] **Step 1: Write failing test**

Write `/project/6105902/shougan/balance-budget/tests/test_ifrlvr_reward.py`:
```python
# ABOUTME: Tests for IF-RLVR reward function using IFEvalG constraint checkers.
# ABOUTME: Validates ground_truth parsing, null-filtering, and fractional scoring.

from tuning.training.reward_functions import ifrlvr_reward_func, _remove_thinking_section


class TestRemoveThinkingSection:
    def test_strips_think_tags(self):
        text = "<think>some reasoning</think>The actual answer."
        assert _remove_thinking_section(text) == "The actual answer."

    def test_strips_answer_tags(self):
        text = "<answer>42</answer>"
        assert _remove_thinking_section(text) == "42"

    def test_strips_assistant_prefix(self):
        text = "<|assistant|>Hello world"
        assert _remove_thinking_section(text) == "Hello world"

    def test_no_tags_returns_unchanged(self):
        text = "Plain response with no tags."
        assert _remove_thinking_section(text) == "Plain response with no tags."

    def test_empty_string(self):
        assert _remove_thinking_section("") == ""


class TestIfrlvrReward:
    def test_returns_float_between_0_and_1(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['hello']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["Include the word hello."],
            completions=["hello world"],
            ground_truth=[gt],
        )
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0

    def test_keyword_present_gets_reward_1(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['banana']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["Include banana."],
            completions=["I like banana."],
            ground_truth=[gt],
        )
        assert rewards == [1.0]

    def test_keyword_missing_gets_reward_0(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['banana']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["Include banana."],
            completions=["I like apples."],
            ground_truth=[gt],
        )
        assert rewards == [0.0]

    def test_none_kwargs_handled(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [None]}]"
        rewards = ifrlvr_reward_func(
            prompts=["test"],
            completions=["test response"],
            ground_truth=[gt],
        )
        assert len(rewards) == 1
        assert 0.0 <= rewards[0] <= 1.0

    def test_batch_returns_correct_length(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['hello']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["p1", "p2"],
            completions=["hello", "goodbye"],
            ground_truth=[gt, gt],
        )
        assert len(rewards) == 2

    def test_conversational_format_completions(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['banana']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=[{"role": "user", "content": "test"}],
            completions=[[{"role": "assistant", "content": "I have a banana."}]],
            ground_truth=[gt],
        )
        assert rewards == [1.0]

    def test_thinking_section_stripped_before_eval(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['banana']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["Include banana."],
            completions=["<think>banana is not the answer</think>I have a banana."],
            ground_truth=[gt],
        )
        assert rewards == [1.0]

    def test_empty_completion_gets_reward_0(self):
        gt = "[{'instruction_id': ['keywords:existence'], 'kwargs': [{'keywords': ['hello']}]}]"
        rewards = ifrlvr_reward_func(
            prompts=["test"],
            completions=[""],
            ground_truth=[gt],
        )
        assert rewards == [0.0]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_ifrlvr_reward.py -v
```
Expected: FAIL — `ifrlvr_reward_func` not found

- [ ] **Step 3: Implement `_remove_thinking_section` and `ifrlvr_reward_func`**

Add to `/project/6105902/shougan/balance-budget/tuning/training/reward_functions.py`:

```python
import ast
import re


def _remove_thinking_section(text):
    """Strip chain-of-thought markup before constraint checking."""
    text = text.replace("<|assistant|>", "")
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<answer>", "").replace("</answer>", "")
    return text.strip()


def ifrlvr_reward_func(prompts, completions, ground_truth, **kwargs):
    """Fractional reward: fraction of IF-RLVR constraints satisfied per completion.

    Parses ground_truth (string-encoded Python list of constraint dicts) and
    uses the IFEvalG constraint registry for verification.
    """
    if not hasattr(ifrlvr_reward_func, "_instruction_dict"):
        from ifrlvr.instructions_registry import INSTRUCTION_DICT
        ifrlvr_reward_func._instruction_dict = INSTRUCTION_DICT

    instruction_dict = ifrlvr_reward_func._instruction_dict

    rewards = []
    for completion, gt_str in zip(completions, ground_truth):
        text = _extract_text(completion)
        answer = _remove_thinking_section(text)

        if not answer.strip():
            rewards.append(0.0)
            continue

        constraint_list = ast.literal_eval(gt_str)
        constraint_dict = constraint_list[0]
        if isinstance(constraint_dict, str):
            import json
            constraint_dict = json.loads(constraint_dict)

        instruction_keys = constraint_dict["instruction_id"]
        args_list = constraint_dict["kwargs"]

        passed = []
        for inst_key, inst_args in zip(instruction_keys, args_list):
            args = {} if inst_args is None else {k: v for k, v in inst_args.items() if v is not None}
            instruction_cls = instruction_dict[inst_key]
            instruction = instruction_cls(inst_key)
            instruction.build_description(**args)
            if instruction.check_following(answer):
                passed.append(1.0)
            else:
                passed.append(0.0)

        rewards.append(sum(passed) / len(passed) if passed else 0.0)

    return rewards
```

Also update the ABOUTME at the top:
```python
# ABOUTME: Reward functions for GRPO/RLVR training matching TRL's GRPOTrainer interface.
# ABOUTME: GSM8K and MATH-500 use binary correctness; IFEval and IF-RLVR use fractional instruction compliance.
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_ifrlvr_reward.py -v
```
Expected: ALL PASS

- [ ] **Step 5: Run existing reward function tests for regression**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_reward_functions.py -v
```
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add tuning/training/reward_functions.py tests/test_ifrlvr_reward.py && git commit -m "feat: add ifrlvr_reward_func using IFEvalG constraint verification"
```

---

### Task 9: Wire into unified_early_pipeline.py

**Files:**
- Modify: `/project/6105902/shougan/balance-budget/tuning/training/unified_early_pipeline.py`

- [ ] **Step 1: Add `"ifrlvr"` to `--dataset` choices**

At line 130, change:
```python
parser.add_argument("--dataset", default="gsm8k", choices=["tuluif", "gsm8k", "openmath"],)
```
To:
```python
parser.add_argument("--dataset", default="gsm8k", choices=["tuluif", "gsm8k", "openmath", "ifrlvr"],)
```

- [ ] **Step 2: Add `"ifbench"` to `--task-name` choices**

At line 138, change:
```python
parser.add_argument("--task-name", default="gsm8k", choices=["ifeval", "gsm8k", "math500"])
```
To:
```python
parser.add_argument("--task-name", default="gsm8k", choices=["ifeval", "gsm8k", "math500", "ifbench"])
```

- [ ] **Step 3: Add `"ifbench"` to `--monitor-evals` choices**

At lines 139-141, change:
```python
parser.add_argument("--monitor-evals", nargs="*", default=[],
                    choices=["ifeval", "gsm8k", "math500"],
                    help="Additional eval benchmarks to monitor (logged to W&B, no stopping)")
```
To:
```python
parser.add_argument("--monitor-evals", nargs="*", default=[],
                    choices=["ifeval", "gsm8k", "math500", "ifbench"],
                    help="Additional eval benchmarks to monitor (logged to W&B, no stopping)")
```

- [ ] **Step 4: Add IFBench to `_build_eval_components`**

In the `_build_eval_components` function, before the `else: raise ValueError` clause (around line 334), add:

```python
    elif args.task_name == "ifbench":
        from tuning.training.eval_strategy import IFBenchStrategy
        strict = getattr(args, f"{prefix}_passk_strict", True)
        primary_eval = IFBenchStrategy(
            k_values=k_values, n_samples=n_samples,
            num_prompts=num_prompts, strict=strict,
        )
```

- [ ] **Step 5: Add IFBench to `_build_monitor_evals`**

In the `_build_monitor_evals` function, after the IFEval elif block (around line 356), add:

```python
        elif name == "ifbench":
            from tuning.training.eval_strategy import IFBenchStrategy
            monitor_evals.append(IFBenchStrategy(k_values=k_values, n_samples=n_samples))
```

- [ ] **Step 6: Add IF-RLVR to `_build_reward_funcs`**

In the `_build_reward_funcs` function (around line 689), before the `else: raise ValueError` clause, add:

```python
    elif args.dataset == "ifrlvr":
        from tuning.training.reward_functions import ifrlvr_reward_func
        return [ifrlvr_reward_func]
```

- [ ] **Step 7: Verify argparse accepts new values**

```bash
cd /project/6105902/shougan/balance-budget && python tuning/training/unified_early_pipeline.py --help | grep -A2 "dataset\|task-name\|monitor-evals"
```
Expected: `ifrlvr` appears in `--dataset` choices, `ifbench` appears in `--task-name` and `--monitor-evals` choices.

- [ ] **Step 8: Run all existing tests for regression**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/ -v --ignore=tests/test_ifrlvr_dataset.py
```
Expected: ALL PASS (the ifrlvr dataset test is slow — ignore for now, it was already verified in Task 7)

- [ ] **Step 9: Commit**

```bash
git add tuning/training/unified_early_pipeline.py && git commit -m "feat: wire IFBench eval + IF-RLVR training into pipeline CLI"
```

---

### Task 10: Full integration smoke test

- [ ] **Step 1: Run all tests together**

```bash
cd /project/6105902/shougan/balance-budget && python -m pytest tests/test_evaluation_lib_refactor.py tests/test_ifbench_test_dataset.py tests/test_ifbench_eval_strategy.py tests/test_ifrlvr_reward.py tests/test_eval_strategy.py tests/test_reward_functions.py -v
```
Expected: ALL PASS

- [ ] **Step 2: Verify IFBench constraint registry loads and has expected count**

```bash
cd /project/6105902/shougan/balance-budget && python -c "
from ifbench_eval.instructions_registry import INSTRUCTION_DICT
print(f'IFBench: {len(INSTRUCTION_DICT)} constraints')
for k in sorted(INSTRUCTION_DICT.keys())[:5]:
    print(f'  {k}')
print('  ...')
"
```
Expected: ~58 constraints, IDs like `count:word_count_range`, `format:emoji`, etc.

- [ ] **Step 3: Verify IFEvalG constraint registry loads and has expected count**

```bash
cd /project/6105902/shougan/balance-budget && python -c "
from ifrlvr.instructions_registry import INSTRUCTION_DICT
print(f'IFEvalG: {len(INSTRUCTION_DICT)} constraints')
for k in sorted(INSTRUCTION_DICT.keys())[:5]:
    print(f'  {k}')
print('  ...')
"
```
Expected: ~54 constraints, IDs like `keywords:existence`, `copy:copying_multiple`, etc.

- [ ] **Step 4: Verify IFBench test dataset loads from HuggingFace**

```bash
cd /project/6105902/shougan/balance-budget && python -c "
from tuning.data.test_dataset import get_ifbench_test_dataset
ds = get_ifbench_test_dataset()
print(f'IFBench test: {len(ds)} prompts')
print(f'Columns: {ds.column_names}')
print(f'First prompt: {ds[0][\"prompt\"][:80]}...')
print(f'First constraint: {ds[0][\"instruction_id_list\"]}')
"
```
Expected: 300 prompts with proper columns

- [ ] **Step 5: Verify IF-RLVR dataset is saved to disk**

```bash
cd /project/6105902/shougan/balance-budget && python -c "
from datasets import load_from_disk
from tuning.config import DATASETS_DIR
ds = load_from_disk(f'{DATASETS_DIR}/rlvr-ifrlvr')
print(f'IF-RLVR train: {len(ds[\"train\"])} prompts')
print(f'IF-RLVR test: {len(ds[\"test\"])} prompts')
print(f'Columns: {ds[\"train\"].column_names}')
"
```
Expected: ~95k train, 100 test, columns include `prompt` and `ground_truth`

- [ ] **Step 6: Commit final state**

```bash
git status && git add -A && git commit -m "feat: IFBench eval + IF-RLVR training integration complete"
```
