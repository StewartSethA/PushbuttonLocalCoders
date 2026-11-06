import json, os, pathlib, sys, tempfile, unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
import pushbutton_metrics as m

class MetricsTests(unittest.TestCase):
    def test_profile_specific_observations_do_not_mix_speculation_lanes(self):
        common = {'model': 'qwen3.6:35b', 'backend': 'ninfer-3090'}
        rows = [{**common, 'profile': 'mtp', 'tg': 100},
                {**common, 'profile': 'mtp-k8v4', 'tg': 200},
                {**common, 'tg': 300}]
        with mock.patch.object(m, 'local_rows', return_value=rows), mock.patch.object(m, '_remote_rows', return_value=[]):
            observations = m.observations('qwen3.6:35b', 'ninfer-3090', profile='mtp-k8v4')
        self.assertEqual([r['tg'] for r in observations], [200])

    def test_local_benchmark_rows_preserve_profile_and_unlock_state(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(m, 'LOCAL_RESULTS', pathlib.Path(td)):
            report = {'model': 'qwen3.6:35b',
                      'backend': {'id': 'ninfer-3090', 'profile': 'mtp-k8v4'},
                      'gpu_unlock_state': 'stock',
                      'cases': [{'name': '1k-c1', 'median_decode_tok_s': 100}]}
            (pathlib.Path(td) / 'report.json').write_text(json.dumps(report))
            rows = m._benchmark_rows()
        self.assertEqual(rows[0]['profile'], 'mtp-k8v4')
        self.assertEqual(rows[0]['gpu_unlock_state'], 'stock')

    def test_remote_source_local_is_still_reference_evidence(self):
        row = {'source':'local', 'model':'qwen3.8:27b', 'backend':'vllm-qwen38-3090', 'tg':42}
        with mock.patch.object(m, 'local_rows', return_value=[row]), \
             mock.patch.object(m, '_remote_rows', return_value=[row]):
            rows = m.observations('qwen3.8:27b', 'vllm-qwen38-3090')
        self.assertEqual([r['evidence_origin'] for r in rows], ['local', 'reference'])
        self.assertNotIn('evidence_origin', row)

    def test_bundled_3090_measurements_exist(self):
        rows=m.observations('qwen3.8:27b','vllm-qwen38-3090',None,'NVIDIA GeForce RTX 3090',262144)
        self.assertTrue(any(r.get('tg')==127.0 for r in rows))

    def test_telemetry_payload_excludes_text(self):
        with tempfile.TemporaryDirectory() as td:
            old_cfg,old_queue=m.CONFIG,m.QUEUE
            try:
                m.CONFIG=pathlib.Path(td)/'cfg'; m.QUEUE=pathlib.Path(td)/'queue'
                m.set_telemetry(True)
                report={'timestamp_utc':'2026-09-14T00:00:00Z','model':'qwen3.8:27b','backend':{'id':'llama.cpp','quant':'IQ3_XXS'},'context':262144,'hardware':{'gpus':[{'name':'RTX 3090','memory_total_mib':24576}]},'cases':[{'name':'1k-c1','median_prefill_tok_s':1000,'median_decode_tok_s':50}], 'prompt':'SECRET','generated_text':'SECRET2'}
                p=m.queue_report(report); obj=json.loads(p.read_text())
                raw=json.dumps(obj)
                self.assertNotIn('SECRET',raw); self.assertNotIn('generated_text',raw)
                self.assertEqual(obj['artifact'],'IQ3_XXS')
            finally:
                m.CONFIG, m.QUEUE=old_cfg,old_queue

if __name__=='__main__': unittest.main()
