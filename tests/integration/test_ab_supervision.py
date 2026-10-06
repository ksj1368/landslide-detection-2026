import copy
import unittest
import numpy as np
import torch
from shapely.geometry import box, Polygon, MultiPolygon
from training.ab_supervision import exact_coverage, attach_site_weights, attach_coverage
from training.holdout import augment_sample
from training.unet import make_batch, supervised_loss, BCE_DICE, COVERAGE_BCE_DICE, SITE_BCE_DICE
from tests.integration.test_d4_selection import fixture

class ABSupervisionTests(unittest.TestCase):
    def test_exact_area_fraction_holes_components_clipping(self):
        shapes=[box(.2,.3,8.7,9.1), box(2.3,1,2.65,31),
                MultiPolygon([box(1.2,1.3,3.7,4.1),box(5.1,7,8.4,9)]),
                Polygon([(1,1),(12,1),(12,12),(1,12)], [[(3.2,3.1),(7.7,3.1),(7.7,8.4),(3.2,8.4)]])]
        for geom in shapes:
            c=exact_coverage(geom,(0,0),(32,32))
            self.assertEqual(c.dtype,np.float32)
            self.assertAlmostEqual(c.sum(dtype=np.float64),geom.area,places=5)
            self.assertTrue(((c>=0)&(c<=1)).all())
        self.assertLess(shapes[1].area,11)
        self.assertLess(exact_coverage(shapes[1],(0,0),(32,32)).max(),.4)
        self.assertAlmostEqual(exact_coverage(box(-1,-1,4.5,4.5),(2,2),(2,2)).sum(),4)

    def test_d4_optional_alignment_and_exclusion(self):
        train,_,_,_=fixture()
        train[0]['gt_geometry']=box(4.2,4.3,10.7,10.1)
        samples,_=attach_coverage(train)
        samples[0]['segmentation_valid'][0,0,:]=False
        for rotation in range(4):
            for flip in (False,True):
                s=augment_sample(samples[0],rotation,flip)
                from preprocessing.transforms import transform_aligned_arrays
                expected=transform_aligned_arrays({'coverage':samples[0]['coverage_target'].transpose(1,2,0)},rotation,flip)['coverage']
                np.testing.assert_array_equal(s['coverage_target'],expected.transpose(2,0,1))
                self.assertEqual(s['coverage_target'].shape,s['target'].shape)
                self.assertAlmostEqual(float(s['coverage_target'].sum()),float(samples[0]['coverage_target'].sum()),places=4)
        self.assertEqual(samples[1]['coverage_target'].sum(),0)
        invalid=copy.deepcopy(train[0]);invalid['segmentation_valid'].fill(False)
        invalid['gt_geometry']=Polygon([(0,0),(5,5),(0,5),(5,0)])
        result,rows=attach_coverage([invalid]);self.assertEqual(result[0]['coverage_target'].sum(),0);self.assertEqual(rows,[])

    def test_site_population_collision_and_weights(self):
        train,_,_,_=fixture()
        a=copy.deepcopy(train[0]);a['key']=('a','d1','same')
        b=copy.deepcopy(a);b['key']=('a','d2','same')
        c=copy.deepcopy(a);c['key']=('b','d1','same')
        ignored=copy.deepcopy(c);ignored['segmentation_valid'].fill(False)
        samples,rows=attach_site_weights([a,b,c,ignored,train[1]])
        self.assertEqual(len(rows),2)
        self.assertEqual([float(s['dice_site_weight']) for s in samples],[.75,.75,1.5,1,1])
        self.assertEqual(sum(float(s['dice_site_weight']) for s in samples[:3])/3,1)

    def test_both_auxiliary_paths_train_and_restore(self):
        import tempfile
        from pathlib import Path
        from training.checkpoint_selection_experiment import train_trajectory, evaluate_policy
        torch.set_num_threads(2)
        traces=[]
        with tempfile.TemporaryDirectory() as directory:
            for name,loss,attach in [('A',COVERAGE_BCE_DICE,attach_coverage),
                                     ('B',SITE_BCE_DICE,attach_site_weights)]:
                train,selection,outer,config=fixture()
                config.update(loss=loss,checkpoint_selection='bce_and_d4_reference')
                train,_=attach(train)
                run=Path(directory)/name
                fit=train_trajectory(train,selection,config,run)
                traces.append(fit['sampling_d4_trace_sha256'])
                for policy in ('best_d4','last'):
                    evaluate_policy(run,selection,outer,config,policy,
                                    policies={'best_d4':'best_d4','last':'last'})
        self.assertEqual(traces[0],traces[1])

    def test_bce_only_coverage_and_fixed_weight_math(self):
        train,_,_,config=fixture()
        batch=make_batch(train)
        outputs={'segmentation':torch.full_like(batch['target'],.7), 'presence':torch.zeros(4)}
        base={**config,'loss':BCE_DICE};a={**config,'loss':COVERAGE_BCE_DICE};b={**config,'loss':SITE_BCE_DICE}
        batch['coverage_target']=batch['target']*.6
        batch['dice_site_weight']=torch.ones(4)
        db,da,dw={},{},{}
        lb=supervised_loss(outputs,batch,base,diagnostics=db)
        la=supervised_loss(outputs,batch,a,diagnostics=da)
        lw=supervised_loss(outputs,batch,b,diagnostics=dw)
        self.assertEqual(db['dice_loss'],da['dice_loss']);self.assertEqual(lb[2],la[2])
        expected=torch.nn.functional.binary_cross_entropy_with_logits(outputs['segmentation'][batch['segmentation_valid']],batch['coverage_target'][batch['segmentation_valid']])
        self.assertAlmostEqual(da['segmentation_bce'],expected.item(),places=6)
        for x,y in zip(lb,lw):self.assertTrue(torch.equal(x,y))
        one={k:v[:1] for k,v in batch.items()};out={k:v[:1] for k,v in outputs.items()}
        one['dice_site_weight'].fill_(2)
        d={};supervised_loss(out,one,b,diagnostics=d)
        self.assertAlmostEqual(d['dice_loss'],2*d['dice_loss_unweighted'],places=6)
        negative={k:v[1:2] for k,v in batch.items()};out={k:v[1:2] for k,v in outputs.items()}
        for x,y in zip(supervised_loss(out,negative,b),supervised_loss(out,negative,base)):self.assertTrue(torch.equal(x,y))

if __name__=='__main__':unittest.main()
