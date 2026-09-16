import pathlib
import sys
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
import pushbutton_pool as pool


class PoolTests(unittest.TestCase):
    def inst(self,**kw):
        base=dict(id='a',model='qwen3.8:27b',backend='llama.cpp',endpoint='http://127.0.0.1:1/v1',max_context=65536,tg=50.0,tg_measured=True)
        base.update(kw);return pool.Instance(**base)

    def test_slow_instance_loses_to_fast(self):
        slow=self.inst(id='slow',tg=20.0);fast=self.inst(id='fast',tg=45.0)
        self.assertEqual(pool.choose_instance([slow,fast],'qwen3.8:27b',8192).id,'fast')

    def test_available_replica_beats_busy(self):
        busy=self.inst(id='busy',active=1,measured_envelopes=[{'concurrency':1,'max_context':65536,'safe':True}])
        free=self.inst(id='free',active=0)
        self.assertEqual(pool.choose_instance([busy,free],'qwen3.8:27b',8192).id,'free')

    def test_context_limit_blocks_route(self):
        x=self.inst(max_context=8192)
        self.assertIsNone(pool.choose_instance([x],'qwen3.8:27b',12000))

    def test_measured_concurrency_expands_capacity(self):
        x=self.inst(measured_envelopes=[{'concurrency':4,'max_context':16384,'safe':True,'tg_per_client_p10':30.0}])
        self.assertEqual(x.capacity(8192),4)
        self.assertEqual(x.capacity(32768),1)

    def test_slow_concurrent_envelope_does_not_expand(self):
        x=self.inst(measured_envelopes=[{'concurrency':4,'max_context':65536,'safe':True,'tg_per_client_p10':19.0}])
        self.assertEqual(x.capacity(8192),1)

    def test_auto_prefers_non_slow(self):
        slow=self.inst(id='slow',tg=12.0);unknown=self.inst(id='unknown',tg=None,tg_measured=False)
        self.assertEqual(pool.auto_candidates([slow,unknown],8192)[0].id,'unknown')


if __name__=='__main__':unittest.main()
