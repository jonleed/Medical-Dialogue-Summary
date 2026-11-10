"""Seeded human-annotation sample and dialogue-level splits (spec §5.2)."""
import random

SEED = 20261109
N_RANDOM, N_ENRICH, N_CALIBRATION = 40, 10, 15
# Dialogues whose outputs the owner already saw during the 2026-11-09 audit.
SEEN = frozenset({'val:9', 'val:14', 'val:24', 'val:37', 'val:39', 'val:43', 'val:59', 'val:81'})


def enrichment_eligible(rows, runs):
    """Dialogues with a contradiction, any race suggestion, or a cross-dimension flag in the given runs."""
    hit = lambda s: (s['proposed_label'] == 'contradicted' or s['fact_type'] == 'race_ethnicity'
                     or s['issue'] == 'cross_dimension_inconsistency')
    return {r['case_id'] for r in rows if r['run'] in runs and any(hit(s) for s in r['suggestions'])}


def select_sample(dialogue_ids, enrichment_pool, seed=SEED, seen=SEEN, heldout_target=None):
    rng = random.Random(seed)
    ids = sorted(set(dialogue_ids))
    rand = rng.sample(ids, N_RANDOM)
    pool = sorted((set(ids) - set(rand)) & set(enrichment_pool))
    enrich = rng.sample(pool, N_ENRICH) if len(pool) > N_ENRICH else pool
    sampled = rand + enrich
    seen_in = sorted(set(sampled) & seen)
    if len(seen_in) > N_CALIBRATION:
        raise ValueError(f'{len(seen_in)} already-seen dialogues sampled; calibration holds only {N_CALIBRATION}')
    rest = sorted(set(sampled) - set(seen_in))
    calibration = sorted(seen_in + rng.sample(rest, N_CALIBRATION - len(seen_in)))
    heldout = sorted(set(sampled) - set(calibration))
    probability = {i: N_RANDOM / len(ids) for i in rand}
    probability.update({i: min(1.0, N_ENRICH / len(pool)) for i in enrich})
    # Optional top-up of held-out from unsampled, unseen dialogues. A separate seeded draw, so the
    # original random/enrichment/calibration selection is unchanged. Top-ups count as random draws.
    top_up = []
    if heldout_target and len(heldout) < heldout_target:
        remaining = sorted(set(ids) - set(sampled) - seen)
        top_up = random.Random(seed + 1).sample(remaining, min(heldout_target - len(heldout), len(remaining)))
        heldout = sorted(heldout + top_up)
        probability.update({i: len(top_up) / len(remaining) for i in top_up})
    return {'seed': seed, 'random': rand, 'enrichment': enrich, 'top_up': top_up, 'calibration': calibration,
            'heldout': heldout, 'heldout_target': heldout_target,
            'inclusion_probability': probability, 'enrichment_pool_size': len(pool),
            'seen_excluded_from_heldout': sorted(seen)}
