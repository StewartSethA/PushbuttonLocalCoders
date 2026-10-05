import contextlib, io, json, os, pathlib, runpy, subprocess, sys, tempfile, unittest
from unittest import mock
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
        self.assertRegex(cp.stdout,r'BLOCK\s+ninfer-3090')
        self.assertRegex(cp.stdout,r'UNVERIFIED\s+llamampere|UNVERIFIED\s+vllm-qwen38-3090')
        self.assertIn('Recommended launch:',cp.stdout)

    def test_24g_table_exposes_measured_upstream_speed(self):
        cp=self.run_sel('qwen3.8:27b','--gpu','0','--vram-limit','24G','--no-interactive')
        self.assertIn('vllm-qwen38-3090',cp.stdout)
        self.assertIn('127',cp.stdout)
        self.assertIn('llamampere',cp.stdout)
        self.assertIn('99',cp.stdout)
        self.assertIn('MEASURED',cp.stdout)

    def test_model_optional_noninteractive_defaults_safely(self):
        cp=self.run_sel('--vram-limit','16G','--ram-limit','32G','--no-interactive','--json')
        obj=json.loads(cp.stdout)
        self.assertEqual(obj['model'],'qwen3.8:27b')
        self.assertEqual(obj['vram_limit_mib'],16384)
        self.assertEqual(obj['ram_limit_mib'],32768)

    def test_calibration_only_explicit_requests_fail_cleanly(self):
        for model in ('qwen3-4b-instruct-2507', 'q3-4b', 'qwen3:4b-instruct-2507'):
            for output in ((), ('--json',)):
                with self.subTest(model=model, output=output):
                    cp=self.run_sel(model,'--vram-limit','128G','--ram-limit','256G',
                                    '--no-interactive',*output,check=False)
                    self.assertNotEqual(cp.returncode,0)
                    self.assertIn('qwen3-4b-instruct-2507 requires calibrated memory metadata',cp.stderr)
                    self.assertNotIn('Traceback',cp.stderr)
                    self.assertNotIn('Recommended launch:',cp.stdout)
                    self.assertEqual(cp.stdout,'')

    def test_interactive_default_excludes_calibration_only_model(self):
        selector=runpy.run_path(str(SEL))
        out=io.StringIO()
        with mock.patch('builtins.input',return_value=''), contextlib.redirect_stdout(out):
            self.assertEqual(selector['model_menu'](),'qwen3.8:27b')
        self.assertNotIn('qwen3-4b-instruct-2507',out.getvalue())
        self.assertIn('  1. qwen3.8:27b',out.getvalue())

    def test_interactive_explicit_calibration_alias_is_preserved(self):
        selector=runpy.run_path(str(SEL))
        with mock.patch('builtins.input',return_value='q3-4b'), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(selector['model_menu'](),'qwen3-4b-instruct-2507')

    def test_rows_skip_uncalibrated_profiles_in_mixed_catalogue(self):
        selector=runpy.run_path(str(SEL))
        plan=selector['plan']
        calibrated=plan.PROFILES['qwen3.8:27b'][-1]
        uncalibrated=plan.PROFILES['qwen3-4b-instruct-2507'][0]
        with mock.patch.dict(plan.PROFILES,{'mixed':(uncalibrated,calibrated)}), \
             mock.patch.object(plan,'scaled_required_mib',wraps=plan.scaled_required_mib) as scale:
            rows=selector['rows_for']('mixed',None,16384,32768,262144)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['artifact'],calibrated.quant)
        self.assertEqual(rows[0]['status'],'FIT')
        scale.assert_called_once_with(calibrated,262144)

    def test_telemetry_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            cp=self.run_sel('qwen3.8:27b','--vram-limit','16G','--no-interactive','--telemetry-opt-in',env={'PUSHBUTTON_CONFIG_DIR':td})
            cfg=json.loads((pathlib.Path(td)/'telemetry.json').read_text())
            self.assertTrue(cfg['enabled'])
            self.assertIn('Opt-in telemetry enabled',cp.stdout)

if __name__=='__main__': unittest.main()
