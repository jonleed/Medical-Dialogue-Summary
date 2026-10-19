"""Separate overlap score: direct mean F1; no evaluate/nltk network downloads."""
from .io import case_hash


def compute_rouge(cases):
    import numpy as np
    from rouge_score.rouge_scorer import RougeScorer
    if not cases or any(c.get('prediction') is None for c in cases):
        raise ValueError('ROUGE requires generated predictions for every case')
    if len({c['protocol'] for c in cases}) != 1: raise ValueError('Do not combine generation protocols')
    scorer = RougeScorer(['rouge1','rouge2','rougeL','rougeLsum'], use_stemmer=True, split_summaries=False)
    per_case = [{'case_id':c['case_id'], 'case_hash':case_hash(c),
      **{k:v.fmeasure for k,v in scorer.score(c['reference'], c['prediction']).items()}} for c in cases]
    return {'protocol':cases[0]['protocol'], 'cases':len(cases), 'aggregation':'arithmetic mean F1',
      'reference':'full notebook-mapped reference, no target truncation',
      'sentence_splitting':'existing newlines only; no nltk sentence tokenizer',
      'comparable_to_historical_scores':False, 'reason':'Historical target truncation and scorer aggregation differ.',
      'metrics':{k:float(np.mean([row[k] for row in per_case])) for k in ['rouge1','rouge2','rougeL','rougeLsum']},
      'per_case':per_case}
