# ABOUTME: Regression tests for reconstructible per-epoch GRPO prompt ordering.
# ABOUTME: A resumed run must iterate the epoch it stopped in, not replay epoch 0.

import sys

# Some older test modules install process-global MagicMock vLLM modules during
# collection. Remove only that incomplete package stub before importing the
# real TRL production path exercised below.
_vllm_module = sys.modules.get("vllm")
if _vllm_module is not None and getattr(_vllm_module, "__spec__", None) is None:
    for _module_name in tuple(sys.modules):
        if _module_name == "vllm" or _module_name.startswith("vllm."):
            sys.modules.pop(_module_name)
    del _module_name
del _vllm_module

from accelerate.data_loader import prepare_data_loader, skip_first_batches
from torch.utils.data import DataLoader
from trl.trainer.utils import RepeatSampler

from tuning.training.grpo_training import _EpochSeededRepeatSampler


def _sampler(cls, seed=42, shuffle=True, num_prompts=64):
    return cls(
        data_source=list(range(num_prompts)),
        mini_repeat_count=8,
        batch_size=4,
        repeat_count=1,
        shuffle=shuffle,
        seed=seed,
    )


def _prompt_order(sampler):
    """The distinct prompt indices in the order the sampler first yields them."""
    seen, order = set(), []
    for index in sampler:
        if index not in seen:
            seen.add(index)
            order.append(index)
    return order


def _orders_through(sampler, epochs):
    orders = {}
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        orders[epoch] = _prompt_order(sampler)
    return orders


def test_resumed_epoch_reproduces_the_historical_uninterrupted_order():
    upstream = _sampler(RepeatSampler)
    uninterrupted = [_prompt_order(upstream) for _ in range(3)]

    resumed = _sampler(_EpochSeededRepeatSampler)
    resumed.set_epoch(2)

    assert _prompt_order(resumed) == uninterrupted[2]


def test_upstream_sampler_replays_epoch_zero_when_resumed():
    """Pins the TRL behaviour our subclass exists to correct."""
    upstream = _sampler(RepeatSampler)
    uninterrupted = [_prompt_order(upstream) for _ in range(3)]

    resumed = _prompt_order(_sampler(RepeatSampler))

    assert resumed != uninterrupted[2]
    assert resumed == uninterrupted[0]


def test_uninterrupted_orders_are_unchanged_from_upstream():
    upstream = _sampler(RepeatSampler)
    seeded = _sampler(_EpochSeededRepeatSampler)

    assert [_prompt_order(seeded) for _ in range(3)] == [
        _prompt_order(upstream) for _ in range(3)
    ]


def test_mid_epoch_resume_survives_accelerate_skip_wrapper():
    """The skip loader must not reset a restored single-process sampler to epoch 0."""
    epoch = 2
    skip_batches = 3
    dataloader_batch_size = 4

    upstream = _sampler(RepeatSampler)
    uninterrupted = [list(upstream) for _ in range(epoch + 1)]
    expected = uninterrupted[epoch][skip_batches * dataloader_batch_size :]

    resumed_sampler = _sampler(_EpochSeededRepeatSampler)
    dataloader = prepare_data_loader(
        DataLoader(
            resumed_sampler.data_source,
            batch_size=dataloader_batch_size,
            sampler=resumed_sampler,
        ),
        num_processes=1,
        process_index=0,
    )
    dataloader.set_epoch(epoch)
    resumed_dataloader = skip_first_batches(dataloader, skip_batches)
    actual = [int(index) for batch in resumed_dataloader for index in batch]

    assert actual == expected


def test_each_epoch_draws_a_distinct_order():
    orders = _orders_through(_sampler(_EpochSeededRepeatSampler), epochs=3)

    assert orders[0] != orders[1]
    assert orders[1] != orders[2]
    assert sorted(orders[1]) == sorted(orders[0])


def test_unshuffled_sampler_keeps_dataset_order_every_epoch():
    sampler = _sampler(_EpochSeededRepeatSampler, shuffle=False)
    orders = _orders_through(sampler, epochs=2)

    assert orders[0] == list(range(64))
    assert orders[1] == list(range(64))
