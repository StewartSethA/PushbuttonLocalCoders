import pathlib, subprocess, sys, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
SEL=ROOT/'pushbutton-select'

class SelectorTests(unittest.TestCase):
    def run_sel(self,*args):
        return subprocess.run([sys.executable,str(SEL),*args],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)

    def test_16g_qwen38_blocks_ninfer_and_fits_iq3(self):
        cp=self.run_sel('qwen3.8:27b','--gpu','0','--vram-limit','16G','--no-interactive')
        self.assertIn('IQ3_XXS',cp.stdout)
        self.assertRegex(cp.stdout,r'BLOCK\s+ninfer-3090')
        self.assertRegex(cp.stdout,r'UNVERIFIED\s+llamampere|UNVERIFIED\s+vllm-qwen38-3090')
        self.assertIn("qwen-local 'qwen3.8:27b@gpu=0,vram=16G'",cp.stdout)

    def test_24g_table_exposes_upstream_speed(self):
        cp=self.run_sel('qwen3.8:27b','--gpu','0','--vram-limit','24G','--no-interactive')
        self.assertIn('vllm-qwen38-3090',cp.stdout)
        self.assertIn('127',cp.stdout)
        self.assertIn('llamampere',cp.stdout)
        self.assertIn('99',cp.stdout)

if __name__=='__main__': unittest.main()
