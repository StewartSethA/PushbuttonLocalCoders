import pathlib, sys, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
import coder_local_plan as p
class CoderPlanTests(unittest.TestCase):
 def test_two_flashnext_replicas_use_4x4(self):
  x=p.build_plan(['qwen3.8-flash-next'],2,p.synthetic_v100(8),262144)
  self.assertEqual([len(w['gpus']) for w in x['workers']],[4,4])
  self.assertTrue(all(w['profile']['quant']=='UD-IQ4_XS' for w in x['workers']))
  self.assertEqual(len(x['unused_gpus']),0)
 def test_mixed_four_agent_layout_leaves_one_spare(self):
  x=p.build_plan(['qwen3.8-flash-next','qwen3.8:27b','qwen3.6:35b','nemotron-3.5-lightning'],4,p.synthetic_v100(8),262144)
  self.assertEqual([len(w['gpus']) for w in x['workers']],[4,1,1,1])
  self.assertEqual(len(x['unused_gpus']),1)
  self.assertEqual(len({g['index'] for w in x['workers'] for g in w['gpus']}),7)
if __name__=='__main__':unittest.main()
