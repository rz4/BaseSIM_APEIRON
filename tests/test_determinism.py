"""Tests for seeds and samplers (src/apeiron/experiment/determinism.py)."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import DataLoader, TensorDataset

from apeiron.experiment import EpochSeededSampler, stable_seed, window_generator


class TestStableSeed:
    def test_same_parts_same_seed(self):
        assert stable_seed(7, "train", 2) == stable_seed(7, "train", 2)

    def test_different_parts_differ(self):
        assert stable_seed(7, "train", 2) != stable_seed(7, "train", 3)
        assert stable_seed(7, "train", 2) != stable_seed(7, "stream", 2)

    def test_no_arithmetic_collisions(self):
        """seed+window schemes collide; hashing must not."""
        assert stable_seed(40, 2) != stable_seed(41, 1)

    def test_parts_cannot_run_together(self):
        """('ab', 'c') and ('a', 'bc') are different keys."""
        assert stable_seed("ab", "c") != stable_seed("a", "bc")

    def test_fits_a_torch_seed(self):
        value = stable_seed("anything", 1)
        assert 0 <= value < 2**63
        torch.Generator().manual_seed(value)  # must not raise


class TestWindowGenerator:
    def test_independent_of_what_came_before(self):
        torch.manual_seed(0)
        first = torch.randn(3, generator=window_generator(11, 2, "train"))
        # burn a lot of global randomness
        torch.manual_seed(99)
        torch.randn(1000)
        second = torch.randn(3, generator=window_generator(11, 2, "train"))
        assert torch.equal(first, second)

    def test_roles_and_windows_differ(self):
        a = torch.randn(3, generator=window_generator(11, 2, "train"))
        b = torch.randn(3, generator=window_generator(11, 2, "stream"))
        c = torch.randn(3, generator=window_generator(11, 3, "train"))
        assert not torch.equal(a, b) and not torch.equal(a, c)


class TestEpochSeededSampler:
    def test_covers_every_index_once(self):
        sampler = EpochSeededSampler(10, seed=1, window=0, role="train")
        assert sorted(sampler) == list(range(10))
        assert len(sampler) == 10

    def test_each_epoch_is_a_different_order(self):
        sampler = EpochSeededSampler(20, seed=1, window=0, role="train")
        first, second = list(sampler), list(sampler)
        assert first != second

    def test_an_epoch_is_a_function_of_its_index(self):
        """Two samplers agree epoch for epoch, whatever else has happened."""
        a = EpochSeededSampler(20, seed=1, window=0, role="train")
        b = EpochSeededSampler(20, seed=1, window=0, role="train")
        orders = [list(a) for _ in range(3)]

        torch.manual_seed(1234)
        torch.randn(500)  # unrelated randomness in between
        assert [list(b) for _ in range(3)] == orders

    def test_landing_back_on_an_epoch(self):
        """Resuming needs one integer, not a replay of every draw."""
        original = EpochSeededSampler(20, seed=1, window=0, role="train")
        for _ in range(4):
            list(original)
        saved = original.epochs_started

        # a fresh sampler told only the epoch count produces the next order
        resumed = EpochSeededSampler(20, seed=1, window=0, role="train")
        resumed.epochs_started = saved
        assert list(resumed) == list(original)

    def test_seed_window_and_role_all_matter(self):
        def order(**kw):
            base = dict(length=12, seed=1, window=0, role="train")
            return list(EpochSeededSampler(**{**base, **kw}))

        assert order() != order(seed=2)
        assert order() != order(window=1)
        assert order() != order(role="stream")

    def test_drives_a_dataloader(self):
        data = TensorDataset(torch.arange(12).float().unsqueeze(1))
        sampler = EpochSeededSampler(12, seed=3, window=0, role="train")
        loader = DataLoader(data, batch_size=4, sampler=sampler)
        seen = [int(v) for batch in loader for v in batch[0].flatten()]
        assert sorted(seen) == list(range(12))
        assert sampler.epochs_started == 1


class TestHarnessSamplers:
    def test_registered_by_role(self, default_cfg, dummy_harness):
        dummy_harness.make_sampler("train", 10, window=0)
        dummy_harness.make_sampler("stream", 20, window=0)
        assert dummy_harness.sampler_state() == {"train": 0, "stream": 0}

    def test_asking_twice_gives_the_same_sampler(self, dummy_harness):
        """eval() rebuilds loaders mid-window; that must not reset the count."""
        first = dummy_harness.make_sampler("train", 10, window=0)
        list(first)
        second = dummy_harness.make_sampler("train", 10, window=0)
        assert second is first
        assert dummy_harness.sampler_state()["train"] == 1

    def test_a_new_window_gets_a_new_sampler(self, dummy_harness):
        first = dummy_harness.make_sampler("train", 10, window=0)
        list(first)
        second = dummy_harness.make_sampler("train", 10, window=1)
        assert second is not first
        assert dummy_harness.sampler_state()["train"] == 0

    def test_a_changed_length_gets_a_new_sampler(self, dummy_harness):
        first = dummy_harness.make_sampler("train", 10, window=0)
        assert dummy_harness.make_sampler("train", 11, window=0) is not first

    def test_state_round_trips(self, dummy_harness):
        dummy_harness.make_sampler("train", 10, window=0)
        dummy_harness.make_sampler("stream", 10, window=0)
        dummy_harness.load_sampler_state({"train": 5, "stream": 2, "absent": 9})
        assert dummy_harness.sampler_state() == {"train": 5, "stream": 2}

    def test_default_harness_has_none(self, default_cfg, make_harness):
        harness = make_harness(replace(default_cfg))
        assert harness.sampler_state() == {}
        harness.load_sampler_state({"train": 3})  # must not raise
