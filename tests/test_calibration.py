import importlib.machinery, importlib.util, pathlib, sys, tempfile, unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
loader=importlib.machinery.SourceFileLoader('pushbutton_calibrate',str(ROOT/'pushbutton-calibrate'))
spec=importlib.util.spec_from_loader(loader.name,loader)
cal=importlib.util.module_from_spec(spec);loader.exec_module(cal)


class CalibrationTests(unittest.TestCase):
    def base(self,**kw):
        x={'id':'i1','model':'m','backend':'llama.cpp','endpoint':'http://127.0.0.1:1/v1',
           'max_context':65536,'framework_max_concurrency':4,'provisioned_slots':4,
           'calibration_safe':True,'measured_envelopes':[{'concurrency':1,'max_context':65536,'safe':True,'evidence':'PROVEN'}]}
        x.update(kw);return x

    def test_depths_and_levels_are_bounded(self):
        self.assertEqual(cal.calibration_depths(20000),[4096,16384,20000])
        self.assertEqual(cal.concurrency_levels(6),[2,4,6])

    def test_next_plan_starts_c2_at_smallest_depth(self):
        self.assertEqual(cal.next_plan(self.base()),(4096,2))

    def test_proven_envelope_skips_redundant_pair(self):
        x=self.base(measured_envelopes=[
            {'concurrency':1,'max_context':65536,'safe':True,'evidence':'PROVEN'},
            {'concurrency':2,'max_context':4096,'safe':True,'evidence':'MEASURED','tg_per_client_p10':30},
        ])
        self.assertEqual(cal.next_plan(x),(4096,4))

    def test_slow_failure_blocks_higher_concurrency_at_same_depth(self):
        x=self.base(calibration_attempts=[{'target_context':4096,'concurrency':2,'safe':False,'reason':'SLOW','epoch':1}])
        self.assertEqual(cal.next_plan(x),(16384,2))

    def test_evaluation_promotes_only_at_floor(self):
        good=[{'ok':True,'tg':25+i,'prompt_tokens':3500,'completion_tokens':100} for i in range(4)]
        out=cal.evaluate_results(good,4096,2)
        self.assertTrue(out['safe']);self.assertGreaterEqual(out['tg_per_client_p10'],25)
        bad=[{'ok':True,'tg':24,'prompt_tokens':3500,'completion_tokens':100} for _ in range(4)]
        self.assertFalse(cal.evaluate_results(bad,4096,2)['safe'])

    def test_record_result_persists_measured_envelope(self):
        with tempfile.TemporaryDirectory() as td:
            import json
            p=pathlib.Path(td)/'instances.json';p.write_text(json.dumps({'instances':[self.base()]}))
            r={'safe':True,'reason':'PASS','target_context':4096,'max_context':3900,'concurrency':2,'samples':4,'tg_per_client_p10':28.0,'tg_per_client_median':30.0}
            cal.record_result('i1',r,p)
            obj=json.loads(p.read_text());inst=obj['instances'][0]
            env=[e for e in inst['measured_envelopes'] if e.get('concurrency')==2][0]
            self.assertEqual(env['max_context'],3900);self.assertEqual(env['tg_per_client_p10'],28.0)


if __name__=='__main__':unittest.main()
