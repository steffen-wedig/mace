"""Tests for LevelQuotaBatchSampler: guaranteed batch composition.

Pins what the sampler promises fused multi-level training: every batch
carries exactly the effective quota of the sparse level and no structure
twice, the quota is raised to a well-covered level's natural share instead
of capping it, one epoch is one pass over the structures outside the pool,
the pool cycles evenly across epochs, and the whole thing is reproducible
from the seed.
"""

import pytest
import torch

from mace.data import LevelQuotaBatchSampler


def make_sampler(
    dataset_size=100,
    pool_size=10,
    batch_size=10,
    quota=1,
    seed=0,
    pool_indices=None,
):
    if pool_indices is None:
        # Spread the pool through the dataset rather than over a prefix.
        stride = dataset_size // pool_size
        pool_indices = [index * stride for index in range(pool_size)]
    return LevelQuotaBatchSampler(
        dataset_size=dataset_size,
        quota_pool_indices=pool_indices,
        batch_size=batch_size,
        quota=quota,
        seed=seed,
    )


def test_every_batch_carries_exactly_the_quota_and_no_duplicates():
    sampler = make_sampler(dataset_size=1000, pool_size=10, batch_size=32, quota=4)
    assert sampler.effective_quota == 4
    pool = set(sampler.pool_indices)
    for batch in sampler:
        assert len(batch) == 32
        assert len(set(batch)) == 32
        assert len([index for index in batch if index in pool]) == 4
        assert len([index for index in batch if index not in pool]) == 28


def test_quota_is_raised_to_the_natural_share_of_a_well_covered_level():
    """A hard partition at quota 4 would UNDERsample a 50 %-covered level."""
    sampler = make_sampler(dataset_size=1000, pool_size=500, batch_size=32, quota=4)
    assert sampler.effective_quota == 16
    for batch in sampler:
        assert len(batch) == 32
        assert sum(index in set(sampler.pool_indices) for index in batch) == 16


def test_natural_share_at_the_coverages_of_the_scaling_study():
    """Batch 32 over 11892 structures, the fractions the study sweeps."""
    effective_quotas = {}
    for coverage in (0.01, 0.05, 0.10, 0.25, 0.50):
        pool_size = round(0.01 * 11892) if coverage == 0.01 else round(coverage * 11892)
        sampler = LevelQuotaBatchSampler(
            dataset_size=11892,
            quota_pool_indices=range(pool_size),
            batch_size=32,
            quota=4,
            seed=0,
        )
        effective_quotas[coverage] = sampler.effective_quota
    assert effective_quotas == {0.01: 4, 0.05: 4, 0.10: 4, 0.25: 8, 0.50: 16}


def test_the_quota_never_fills_the_whole_batch():
    """Even a level covering (almost) everything leaves room for the rest."""
    sampler = make_sampler(dataset_size=100, pool_size=98, batch_size=10, quota=9)
    assert sampler.effective_quota == 9
    assert sampler.rest_per_batch == 1
    sampler = LevelQuotaBatchSampler(
        dataset_size=100,
        quota_pool_indices=range(99),
        batch_size=10,
        quota=5,
        seed=0,
    )
    assert sampler.effective_quota == 9  # natural share 9.9, capped at batch - 1


def test_one_epoch_is_one_pass_over_the_structures_outside_the_pool():
    sampler = make_sampler(dataset_size=100, pool_size=10, batch_size=10, quota=1)
    assert sampler.rest_per_batch == 9
    assert len(sampler) == 10  # 90 rest structures, 9 per batch, nothing dropped
    pool = set(sampler.pool_indices)
    seen = [index for batch in sampler for index in batch if index not in pool]
    assert sorted(seen) == sorted(set(range(100)) - pool)


def test_the_incomplete_final_batch_is_dropped():
    sampler = make_sampler(dataset_size=100, pool_size=10, batch_size=12, quota=2)
    assert sampler.rest_per_batch == 10
    assert len(sampler) == 9  # 90 rest structures, 10 per batch, 0 left over
    sampler = make_sampler(dataset_size=97, pool_size=7, batch_size=12, quota=2)
    assert sampler.rest_per_batch == 10
    assert len(sampler) == 9  # 90 rest structures again
    batches = list(sampler)
    assert len(batches) == 9
    assert all(len(batch) == 12 for batch in batches)


def test_len_matches_the_number_of_yielded_batches():
    sampler = make_sampler(dataset_size=253, pool_size=11, batch_size=16, quota=3)
    for _ in range(3):
        assert len(list(sampler)) == len(sampler)


def test_the_pool_cycles_evenly_across_epochs():
    sampler = make_sampler(dataset_size=100, pool_size=7, batch_size=10, quota=3)
    assert sampler.effective_quota == 3
    visits = {index: 0 for index in sampler.pool_indices}
    for _ in range(7):
        for batch in sampler:
            for index in batch:
                if index in visits:
                    visits[index] += 1
        # The cycle carries across epochs, so the counts stay balanced at
        # every epoch boundary, not just after a whole number of cycles.
        assert max(visits.values()) - min(visits.values()) <= 1
    assert sum(visits.values()) == 7 * len(sampler) * sampler.effective_quota


def test_pool_passes_per_epoch_reports_the_oversampling_factor():
    sampler = make_sampler(dataset_size=11892, pool_size=119, batch_size=32, quota=4)
    assert sampler.effective_quota == 4
    assert len(sampler) == (11892 - 119) // 28
    assert sampler.pool_passes_per_epoch == pytest.approx(
        4 * len(sampler) / 119, rel=1e-12
    )
    # 150 epochs of this is comparable to the converged baseline's 2500 passes.
    assert 1500 < 150 * sampler.pool_passes_per_epoch < 2500


def test_the_same_seed_reproduces_the_batches_and_a_different_seed_does_not():
    first = make_sampler(dataset_size=200, pool_size=20, batch_size=10, quota=2, seed=7)
    second = make_sampler(dataset_size=200, pool_size=20, batch_size=10, quota=2, seed=7)
    other = make_sampler(dataset_size=200, pool_size=20, batch_size=10, quota=2, seed=8)

    first_epochs = [list(first), list(first)]
    second_epochs = [list(second), list(second)]
    other_epochs = [list(other), list(other)]

    assert first_epochs == second_epochs
    assert first_epochs != other_epochs
    # Successive epochs of one run reshuffle.
    assert first_epochs[0] != first_epochs[1]


def test_an_empty_pool_is_rejected():
    with pytest.raises(ValueError, match="quota pool is empty"):
        LevelQuotaBatchSampler(
            dataset_size=100,
            quota_pool_indices=[],
            batch_size=10,
            quota=2,
            seed=0,
        )


def test_a_quota_filling_the_batch_is_rejected():
    with pytest.raises(ValueError, match="must leave room"):
        make_sampler(batch_size=4, quota=4)
    with pytest.raises(ValueError, match="must leave room"):
        make_sampler(batch_size=4, quota=5)
    with pytest.raises(ValueError, match="at least one structure"):
        make_sampler(quota=0)


def test_out_of_range_pool_indices_are_rejected():
    with pytest.raises(ValueError, match="outside the dataset"):
        make_sampler(dataset_size=50, pool_indices=[1, 2, 60])


def test_a_dataset_too_small_for_one_batch_is_rejected():
    with pytest.raises(ValueError, match="cannot fill a single batch"):
        LevelQuotaBatchSampler(
            dataset_size=8,
            quota_pool_indices=[0, 1],
            batch_size=20,
            quota=1,
            seed=0,
        )


def test_it_drives_a_dataloader_as_a_batch_sampler():
    """torch rejects batch_size/shuffle/sampler/drop_last alongside this."""
    dataset = [torch.tensor([float(index)]) for index in range(100)]
    sampler = make_sampler(dataset_size=100, pool_size=10, batch_size=10, quota=2)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_sampler=sampler)
    batches = list(loader)
    assert len(batches) == len(sampler)
    assert all(batch.shape == (10, 1) for batch in batches)


def test_parse_multilevel_level_quota():
    from mace.tools.scripts_utils import parse_multilevel_level_quota

    heads = ["revpbe", "delta_cc"]
    assert parse_multilevel_level_quota(None, heads) is None
    assert parse_multilevel_level_quota("delta_cc:4", heads) == {"delta_cc": 4}
    assert parse_multilevel_level_quota("delta_cc:4, revpbe:2", heads) == {
        "delta_cc": 4,
        "revpbe": 2,
    }
    with pytest.raises(ValueError, match="but the run has heads"):
        parse_multilevel_level_quota("ccsd_t:4", heads)
    with pytest.raises(ValueError, match="malformed"):
        parse_multilevel_level_quota("delta_cc=4", heads)
    with pytest.raises(ValueError, match="duplicate"):
        parse_multilevel_level_quota("delta_cc:4,delta_cc:2", heads)
    with pytest.raises(ValueError, match="at least one structure"):
        parse_multilevel_level_quota("delta_cc:0", heads)


def test_the_command_line_argument_defaults_to_off():
    from mace.tools.arg_parser import build_default_arg_parser

    arguments = build_default_arg_parser().parse_args(
        ["--name", "test", "--train_file", "train.xyz"]
    )
    assert arguments.multilevel_level_quota is None
    arguments = build_default_arg_parser().parse_args(
        ["--name", "test", "--train_file", "train.xyz", "--multilevel_level_quota", "ccsd_t:4"]
    )
    assert arguments.multilevel_level_quota == "ccsd_t:4"
