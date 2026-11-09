import json
from pathlib import Path
import numpy as np
import pytest
from aps import APSConfig, select_frames

FIXTURES = json.loads((Path(__file__).parent/'fixtures/reference_selector.json').read_text())


@pytest.mark.parametrize('row', FIXTURES)
def test_matches_reference_selections(row):
    assert select_frames(row['scores'], row['frames'], APSConfig(**row['config'])) == row['expected']


@pytest.mark.parametrize('budget', [8, 16, 32, 64])
@pytest.mark.parametrize('n', [7, 8, 16, 32, 64, 65, 100])
def test_budget_and_fallback_boundary(n, budget):
    frames = list(range(0, n*9, 9))
    selected, trace = select_frames(np.random.default_rng(1).random(n), frames,
                                    APSConfig(budget=budget), return_details=True)
    assert len(selected) == min(n, budget)
    assert selected == sorted(set(selected))
    assert set(selected) <= set(frames)
    assert trace['uniform_fallback'] == (n < budget)
    if trace['partition_used']:
        assert sum(trace['allocation']) == budget
        assert trace['intervals'][0][0] == 0
        assert trace['intervals'][-1][1] == n


def test_equal_length_preserves_full_partition_count():
    x = np.random.default_rng(7).random(130)
    _, full = select_frames(x, list(range(130)), APSConfig(), return_details=True)
    _, equal = select_frames(x, list(range(130)), APSConfig(variant='equal_length'), return_details=True)
    assert len(full['intervals']) == len(equal['intervals'])
    lengths = [r-l for l,r in equal['intervals']]
    assert max(lengths)-min(lengths) <= 1


@pytest.mark.parametrize('scores,frames', [([],[]), ([float('nan')],[0]), ([1],[0,1]), ([1,2],[2,1]), ([1,2],[1,1]), ([1],[.5])])
def test_reject_bad_inputs(scores, frames):
    with pytest.raises(ValueError):
        select_frames(scores, frames)


def test_zero_penalty_and_full_grid_parameters():
    scores = np.random.default_rng(9).random(97)
    for alpha in [.5, 1., 1.5]:
        for beta in [.4, .5, .6, .7]:
            result = select_frames(scores, list(range(97)), APSConfig(alpha=alpha,beta=beta,penalty=0))
            assert len(result) == 32
