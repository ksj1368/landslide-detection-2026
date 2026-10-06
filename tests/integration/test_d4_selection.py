"""D103 selection, fixed metric, and unchanged optimization trajectory."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from shapely.geometry import Polygon, box

from evaluation.preprocessing_boundary_diagnostics import score_condition
from inference.unet import predict_probabilities
from training.checkpoint_selection_experiment import D103_POLICIES, train_trajectory, evaluate_policy
from training.d4_selection import selected_row, selection_population, reference_total, evaluate_selection
from training.unet import load_checkpoint


def fixture():
    from preprocessing.model_input import prepare_sample, normalize_sample, index_configuration
    from training.unet import CONFIG, BCE_DICE, DICE_PARAMETERS
    raw = np.random.default_rng(42).integers(100, 6000, (64, 64, 4)).astype(np.float32)
    raw[0, 0, 0] = 0
    raw[1, 1, 1] = np.nan
    template = prepare_sample(raw, box(25,25,34,34), {'visible': True, 'polygon': box(4,4,10,10)},
                              origin_policy='shift_inside', input_index='NDVI')
    stats = {'bands': ['R','G','B','NIR'], 'mean': [.25,.3,.2,.4], 'std': [.15,.2,.1,.25]}
    template = normalize_sample(template, 'zscore', stats)
    partitions = []
    for group in ('train', 'selection', 'outer'):
        records = []
        for i in range(4):
            s = copy.deepcopy(template)
            s.update(key=(group, '20200101', str(i)), group=group)
            if i % 2:
                s.update(state='negative', presence=0., gt_geometry=Polygon())
                s['target'].fill(0)
            else:
                s['gt_geometry'] = box(4,4,10,10)
            records.append(s)
        partitions.append(records)
    config = {**CONFIG, **index_configuration('NDVI'), 'origin_policy':'shift_inside',
              'normalization':'zscore', 'normalization_statistics':stats, 'device':'cpu',
              'loss':BCE_DICE, 'loss_parameters':DICE_PARAMETERS, 'diagnostics':True,
              'scheduler':{'name':'constant'}, 'augmentation':'d4', 'split_id':'synthetic',
              'experiment_id':'E086', 'updates':3, 'validation_every':1, 'batch_size':3, 'scope':'holdout'}
    return (*partitions, config)


class D4SelectionTests(unittest.TestCase):
    def test_first_extrema_and_last(self):
        rows = [{'update': i+1, 'loss': loss, 'selection_d4': {'reference_total': score}}
                for i, (loss, score) in enumerate([(3,.2),(1,.3),(1,.5),(2,.5)])]
        self.assertEqual(selected_row(rows,'last')['update'],4)
        self.assertEqual(selected_row(rows,'best')['update'],2)
        self.assertEqual(selected_row(rows,'best_d4')['update'],3)
        with self.assertRaises(ValueError):
            selected_row(rows,'other')

    def test_population_na_and_invalid_contract(self):
        _, samples, _, _ = fixture()
        selection_population(samples)
        with self.assertRaises(ValueError):
            selection_population([s for s in samples if s['state']=='negative'])
        with self.assertRaises(ValueError):
            selection_population([s for s in samples if s['state']=='positive'])
        tiny=copy.deepcopy(samples)
        for s in tiny:
            if s['state']=='positive': s['gt_geometry']=box(0,0,1,1)
        with self.assertRaisesRegex(ValueError,'Undefined'):
            selection_population(tiny)
        from evaluation.scorer import score_predictions
        samples[0]['gt_geometry']=Polygon([(0,0),(5,5),(0,5),(5,0),(0,0)])
        scores=score_predictions(samples,[{'key':s['key'],'polygons':[]} for s in samples])['summary']
        self.assertIsNone(scores['shape']); self.assertIsNone(scores['total'])
        self.assertTrue(np.isfinite(reference_total(scores)))
        self.assertEqual(selection_population(samples)['invalid_gt_count'],1)

    def test_selector_preserves_training_and_restores_three_policies(self):
        train, selection, outer, config = fixture()
        torch.set_num_threads(2)
        def same(a,b):
            if torch.is_tensor(a): self.assertTrue(torch.equal(a,b))
            elif isinstance(a,dict):
                self.assertEqual(a.keys(),b.keys())
                for k in a: same(a[k],b[k])
            elif isinstance(a,(list,tuple)):
                self.assertEqual(len(a),len(b))
                for x,y in zip(a,b): same(x,y)
            else: self.assertEqual(a,b)
        with tempfile.TemporaryDirectory() as d:
            off,on=Path(d)/'off',Path(d)/'on'
            a=train_trajectory(train,selection,config,off)
            config={**config,'checkpoint_selection':'bce_and_d4_reference'}
            b=train_trajectory(train,selection,config,on)
            for name in ('last','best'):
                _,saved_a=load_checkpoint(off/f'{name}.pt','cpu')
                _,saved_b=load_checkpoint(on/f'{name}.pt','cpu')
                for field in ('model','optimizer','updates'): same(saved_a[field],saved_b[field])
            def history(root):
                return [{k:v for k,v in json.loads(r).items() if k!='seconds'}
                        for r in (root/'updates.jsonl').read_text().splitlines()]
            self.assertEqual(history(off),history(on))
            ha=json.loads((off/'validation_history.json').read_text())
            hb=json.loads((on/'validation_history.json').read_text())
            self.assertEqual(ha,[{k:v for k,v in row.items() if k!='selection_d4'} for row in hb])
            self.assertEqual(a['sampling_d4_trace_sha256'],b['sampling_d4_trace_sha256'])
            self.assertEqual(b['d4_validation_count'],3)
            model,_=load_checkpoint(on/'best_d4.pt','cpu')
            model.train(); rng=torch.get_rng_state().clone()
            summary,predictions=evaluate_selection(model,selection,'cpu')
            self.assertTrue(model.training); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            masks,heads=[],[]
            for s in selection:
                mask,head,_=predict_probabilities(model,torch.from_numpy(s['input']).unsqueeze(0),tta='d4')
                masks.append(mask[0]); heads.append(head[0])
            condition=score_condition(selection,masks,heads,condition_id='test',method='probability_contour')
            for field in ('presence','shape_diagnostic','confusion'):
                self.assertEqual(summary[field],condition['summary'][field])
            self.assertEqual([p['polygons'] for p in predictions],[p['polygons'] for p in condition['predictions']])
            for policy in D103_POLICIES:
                result=evaluate_policy(on,selection,outer,config,policy,policies=D103_POLICIES)
                self.assertEqual({r['key'][0] for r in result['rows']},{'outer'})
                expected=selected_row(hb,D103_POLICIES[policy])
                self.assertEqual(result['checkpoint']['update'],expected['update'])
                self.assertEqual(result['checkpoint']['inner_d4'],expected['selection_d4'])
            with patch('training.checkpoint_selection_experiment.fit_holdout') as fit:
                train_trajectory(train,selection,config,on); fit.assert_not_called()


if __name__=='__main__': unittest.main()
