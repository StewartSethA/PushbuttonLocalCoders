import json, os, pathlib, subprocess, sys, tempfile, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
SEL=ROOT/'pushbutton-select'

class SelectorTests(unittest.TestCase):
    def run_sel(self,*args,env=None,check=True):
        e=os.environ.copy(); e['PUSHBUTTON_METRICS_URL']='http://127.0.0.1:9/offline'
        if env:e.update(env)
        return subprocess.run([sys.executable,str(SEL),*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=check,env=e)

    def test_16g_qwen38_blocks_ninfer_and_fits_iq3(self):
        cp=self.run_sel('qwen3.8:27b','--gpu','0','--vram-limit','16G','--no-interactive')
        self.assertIn('IQ3_XXS',cp.stdout)
        self.assertRegex(cp.stdout,r'BLOCK\s+\S+\s+ninfer-3090')
        self.assertRegex(cp.stdout,r'UNVERIFIED\s+\S+\s+(llamampere|vllm-qwen38-3090)')
        self.assertIn('Recommended launch:',cp.stdout)

    def test_24g_table_exposes_measured_upstream_speed(self):
        cp=self.run_sel('qwen3.8:27b','--gpu','0','--vram-limit','24G','--no-interactive')
        self.assertIn('vllm-qwen38-3090',cp.stdout)
        self.assertIn('127',cp.stdout)
        self.assertIn('llamampere',cp.stdout)
        self.assertIn('99',cp.stdout)
        self.assertIn('MEASURED',cp.stdout)
        self.assertIn('25 TG/s',cp.stdout)

    def test_model_optional_noninteractive_defaults_safely(self):
        cp=self.run_sel('--vram-limit','16G','--ram-limit','32G','--no-interactive','--json')
        obj=json.loads(cp.stdout)
        self.assertEqual(obj['model'],'qwen3.8:27b')
        self.assertEqual(obj['vram_limit_mib'],16384)
        self.assertEqual(obj['ram_limit_mib'],32768)
        self.assertEqual(obj['speed_floor_tg'],25.0)
        self.assertTrue(all('speed_status' in r for r in obj['rows']))

    def test_telemetry_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            cp=self.run_sel('qwen3.8:27b','--vram-limit','16G','--no-interactive','--telemetry-opt-in',env={'PUSHBUTTON_CONFIG_DIR':td})
            cfg=json.loads((pathlib.Path(td)/'telemetry.json').read_text())
            self.assertTrue(cfg['enabled'])
            self.assertIn('Opt-in telemetry enabled',cp.stdout)

if __name__=='__main__': unittest.main()
