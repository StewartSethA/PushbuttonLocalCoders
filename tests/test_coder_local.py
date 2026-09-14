import json, pathlib, sys, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
import coder_local_plan as p

class CoderPlanTests(unittest.TestCase):
 def reqs(self,*models): return [p.parse_model_spec(x,{}) for x in models]

 def test_two_flashnext_replicas_use_4x4(self):
  x=p.build_plan(self.reqs('qwen3.8-flash-next'),2,p.synthetic_v100(8),262144)
  self.assertEqual([len(w['gpus']) for w in x['workers']],[4,4])
  self.assertTrue(all(w['profile']['quant']=='UD-IQ4_XS' for w in x['workers']))
  self.assertEqual(len(x['unused_gpus']),0)

 def test_mixed_four_agent_layout_leaves_one_spare(self):
  x=p.build_plan(self.reqs('qwen3.8-flash-next','qwen3.8:27b','qwen3.6:35b','nemotron-3.5-lightning'),4,p.synthetic_v100(8),262144)
  self.assertEqual([len(w['gpus']) for w in x['workers']],[4,1,1,1])
  self.assertEqual(len(x['unused_gpus']),1)
  self.assertEqual(len({g['index'] for w in x['workers'] for g in w['gpus']}),7)

 def test_dual_3090_pins_and_16g_caps_choose_fitting_quants(self):
  reqs=self.reqs('qwen3.8:27b@gpu=0,vram=16G','qwen3.6:35b@gpu=1,vram=16G')
  x=p.build_plan(reqs,2,p.synthetic_3090(2),262144)
  self.assertEqual([w['cuda_visible_devices'] for w in x['workers']],['0','1'])
  self.assertEqual([w['profile']['quant'] for w in x['workers']],['IQ3_XXS','UD-IQ3_XXS'])
  self.assertTrue(all(w['required_mib']<=16384 for w in x['workers']))
  self.assertTrue(all(w['vram_limit_mib_per_gpu']==16384 for w in x['workers']))

 def test_exact_multi_gpu_pin(self):
  r=p.parse_model_spec('qwen3.8-flash-next@gpu=2+3+4+5,vram=32G',{})
  x=p.build_plan([r],1,p.synthetic_v100(8),262144)
  self.assertEqual({g['index'] for g in x['workers'][0]['gpus']},{2,3,4,5})
  self.assertEqual(x['workers'][0]['profile']['quant'],'UD-IQ4_XS')

 def test_too_small_cap_fails_instead_of_overcommitting(self):
  with self.assertRaisesRegex(ValueError,'no fitting quant'):
   p.build_plan(self.reqs('qwen3.8:27b@gpu=0,vram=12G'),1,p.synthetic_3090(1),262144)

 def test_persistent_config_and_inline_override(self):
  with tempfile.TemporaryDirectory() as td:
   path=pathlib.Path(td)/'placement.json'
   path.write_text(json.dumps({'models':{'qwen3.8:27b':{'gpus':[1],'vram_limit':'16G'}}}))
   defaults=p.load_placement_config(str(path))
   r=p.parse_model_spec('qwen3.8:27b',defaults)
   self.assertEqual(r.gpu_indices,(1,)); self.assertEqual(r.vram_limit_mib,16384)
   r2=p.parse_model_spec('qwen3.8:27b@gpu=0,vram=15G',defaults)
   self.assertEqual(r2.gpu_indices,(0,)); self.assertEqual(r2.vram_limit_mib,15360)

if __name__=='__main__': unittest.main()
