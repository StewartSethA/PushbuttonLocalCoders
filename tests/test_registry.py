import json, os, pathlib, subprocess, sys, tempfile, time, unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lib'))
import pushbutton_registry as reg


class RegistryTests(unittest.TestCase):
    def test_append_is_duplicate_safe(self):
        with tempfile.TemporaryDirectory() as td:
            p=pathlib.Path(td)/'instances.json'
            self.assertTrue(reg.append_instance({'id':'a','model':'x'},p))
            self.assertFalse(reg.append_instance({'id':'a','model':'x'},p))
            self.assertEqual(len(reg.load(p)['instances']),1)

    def test_two_process_appends_are_not_lost(self):
        with tempfile.TemporaryDirectory() as td:
            env=os.environ.copy();env['PUSHBUTTON_RUNTIME_STATE']=td
            code=("import sys;sys.path.insert(0,%r);import pushbutton_registry as r;"
                  "r.append_instance({'id':sys.argv[1]},r.DEFAULT_REGISTRY)" % str(ROOT/'lib'))
            ps=[subprocess.Popen([sys.executable,'-c',code,str(i)],env=env) for i in range(8)]
            self.assertTrue(all(p.wait()==0 for p in ps))
            obj=json.loads((pathlib.Path(td)/'instances.json').read_text())
            self.assertEqual({x['id'] for x in obj['instances']},{str(i) for i in range(8)})

    def test_named_lock_serializes_processes(self):
        with tempfile.TemporaryDirectory() as td:
            env=os.environ.copy();env['PUSHBUTTON_RUNTIME_STATE']=td
            code=("import sys,time;sys.path.insert(0,%r);import pushbutton_registry as r;"
                  "\nwith r.named_lock('cold',timeout_s=3):\n time.sleep(float(sys.argv[1]));print('done')" % str(ROOT/'lib'))
            p1=subprocess.Popen([sys.executable,'-c',code,'0.6'],env=env,stdout=subprocess.PIPE,text=True)
            time.sleep(.1);start=time.monotonic()
            p2=subprocess.run([sys.executable,'-c',code,'0'],env=env,stdout=subprocess.PIPE,text=True,check=True)
            elapsed=time.monotonic()-start;p1.wait()
            self.assertGreater(elapsed,.35)
            self.assertIn('done',p2.stdout)


if __name__=='__main__':unittest.main()
