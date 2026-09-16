import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
import pushbutton_policy as policy


class PolicyTests(unittest.TestCase):
    def test_decode_floor(self):
        self.assertEqual(policy.speed_label(24.9,measured=True)[0],'SLOW')
        self.assertEqual(policy.speed_label(25.0,measured=True)[0],'OK')
        self.assertEqual(policy.speed_label(14.9,measured=False)[0],'VERY_SLOW')

    def test_context_clamps_to_model_and_backend(self):
        ctx,msg=policy.clamp_context(300000,model_context=200000,backend='llama.cpp')
        self.assertEqual(ctx,200000); self.assertIn('clamped',msg)

    def test_unknown_backend_stays_c1(self):
        self.assertEqual(policy.proven_concurrency(backend='mystery',requested_context=8192),1)

    def test_proof_at_larger_context_applies_downward(self):
        env=[{'concurrency':4,'max_context':16384,'safe':True,'tg_per_client_p10':31.0}]
        self.assertEqual(policy.proven_concurrency(backend='llama.cpp',requested_context=8192,measured_envelopes=env),4)
        self.assertEqual(policy.proven_concurrency(backend='llama.cpp',requested_context=32768,measured_envelopes=env),1)

    def test_framework_cap_is_hard(self):
        env=[{'concurrency':8,'max_context':65536,'safe':True,'tg_per_client_p10':40.0}]
        self.assertEqual(policy.proven_concurrency(backend='ninfer-3090',requested_context=8192,framework_max=2,measured_envelopes=env),2)

    def test_aggregate_speed_cannot_hide_slow_clients(self):
        env=[{'concurrency':4,'max_context':65536,'safe':True,'aggregate_tg':80.0}]
        self.assertEqual(policy.envelope_client_tg(env[0]),20.0)
        self.assertEqual(policy.proven_concurrency(backend='llama.cpp',requested_context=8192,measured_envelopes=env),1)

    def test_per_client_floor_promotes_at_25(self):
        env=[{'concurrency':3,'max_context':65536,'safe':True,'tg_per_client_p10':25.0}]
        self.assertEqual(policy.proven_concurrency(backend='llama.cpp',requested_context=8192,measured_envelopes=env),3)

    def test_missing_concurrent_speed_stays_c1(self):
        env=[{'concurrency':4,'max_context':65536,'safe':True}]
        self.assertEqual(policy.proven_concurrency(backend='llama.cpp',requested_context=8192,measured_envelopes=env),1)

    def test_admission(self):
        self.assertEqual(policy.admission_decision(active=0,capacity=1,queue_depth=0),'ADMIT')
        self.assertEqual(policy.admission_decision(active=1,capacity=1,queue_depth=0),'QUEUE')
        self.assertEqual(policy.admission_decision(active=1,capacity=1,queue_depth=64),'REJECT_BUSY')


if __name__=='__main__':unittest.main()
