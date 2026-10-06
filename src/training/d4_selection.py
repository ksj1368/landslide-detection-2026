"""D103 inner-only checkpoint selection with the fixed D4/.2/contour.4 rule."""
import math

import torch

from evaluation.scorer import score_predictions
from inference.unet import predict_probabilities
from postprocessing.boundary_processing import boundary_prediction


DENOMINATORS = ('input_count', 'labeled_count', 'positive_count', 'negative_count',
                'shape_eligible_count', 'shape_site_count', 'invalid_gt_count')


def reference_total(summary):
    values = [summary['presence'], summary['shape_diagnostic']]
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError('Undefined inner D4 selection metric')
    return .6 * values[0] + .4 * values[1]


def selection_population(samples):
    empty = [{'key': s['key'], 'polygons': []} for s in samples]
    summary = score_predictions(samples, empty, presence_threshold=.2)['summary']
    reference_total(summary)
    if not summary['positive_count'] or not summary['negative_count']:
        raise ValueError('Inner selection requires both label classes')
    return {k: summary[k] for k in DENOMINATORS}


def evaluate_selection(model, samples, device):
    """No model construction, optimizer calls or training RNG consumption."""
    was_training = model.training
    predictions = []
    try:
        model.eval()
        with torch.inference_mode():
            for sample in samples:
                x = torch.from_numpy(sample['input']).unsqueeze(0).to(device)
                mask, head, _ = predict_probabilities(model, x, tta='d4')
                predictions.append({'key': sample['key'], **boundary_prediction(
                    sample, mask[0], float(head[0]), method='probability_contour',
                    presence_threshold=.2, mask_threshold=.4)})
        summary = score_predictions(samples, predictions, presence_threshold=.2)['summary']
        summary['reference_total'] = reference_total(summary)
        return summary, predictions
    finally:
        model.train(was_training)


def selected_row(history, checkpoint_name):
    if checkpoint_name == 'last':
        return history[-1]
    if checkpoint_name == 'best':
        return min(history, key=lambda row: row['loss'])
    if checkpoint_name == 'best_d4':
        return max(history, key=lambda row: row['selection_d4']['reference_total'])
    raise ValueError('Unknown checkpoint selection policy')
