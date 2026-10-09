import json, os, pathlib, socket, subprocess, sys, tempfile, threading, time, unittest, urllib.error, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT=pathlib.Path(__file__).resolve().parents[1]


def free_port():
    s=socket.socket();s.bind(('127.0.0.1',0));p=s.getsockname()[1];s.close();return p


class FakeState:
    lock=threading.Lock();active=0;max_active=0;calls=0


class FakeBackend(BaseHTTPRequestHandler):
    def log_message(self,*a):pass
    def do_GET(self):
        if self.path.endswith('/models'):
            b=json.dumps({'data':[{'id':'backend-real-id'}]}).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b);return
        self.send_error(404)
    def do_POST(self):
        n=int(self.headers.get('Content-Length') or 0);obj=json.loads(self.rfile.read(n) or b'{}')
        with FakeState.lock:
            FakeState.active+=1;FakeState.calls+=1;FakeState.max_active=max(FakeState.max_active,FakeState.active)
        time.sleep(.18)
        with FakeState.lock:FakeState.active-=1
        content='seen:'+str(obj.get('model'))
        out={'id':'x','model':'backend-real-id','choices':[{'message':{'role':'assistant','content':content}}],'usage':{'prompt_tokens':10,'completion_tokens':4}}
        b=json.dumps(out).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)


class BrokerTests(unittest.TestCase):
    def setUp(self):
        FakeState.active=FakeState.max_active=FakeState.calls=0
        self.backend_port=free_port();self.backend=ThreadingHTTPServer(('127.0.0.1',self.backend_port),FakeBackend);self.bt=threading.Thread(target=self.backend.serve_forever,daemon=True);self.bt.start()
        self.td=tempfile.TemporaryDirectory();state=pathlib.Path(self.td.name);self.broker_port=free_port()
        reg={'instances':[{'id':'i1','model':'logical-model','backend':'llama.cpp','endpoint':f'http://127.0.0.1:{self.backend_port}/v1','max_context':128,'framework_max_concurrency':1,'measured_envelopes':[{'concurrency':1,'max_context':128,'safe':True,'evidence':'PROVEN'}],'tg':50.0,'tg_measured':True,'healthy':True}]}
        (state/'instances.json').write_text(json.dumps(reg))
        env=os.environ.copy();env['PUSHBUTTON_RUNTIME_STATE']=self.td.name;env['PUSHBUTTON_CONFIG_DIR']=str(state/'cfg');env['PUSHBUTTON_CACHE_DIR']=str(state/'cache')
        self.proc=subprocess.Popen([sys.executable,str(ROOT/'pushbutton-broker'),'--port',str(self.broker_port),'--registry',str(state/'instances.json'),'--no-scaler','--no-calibrator'],env=env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        deadline=time.time()+5
        while time.time()<deadline:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{self.broker_port}/v1/models',timeout=.2):break
            except Exception:time.sleep(.05)
        else:self.fail('broker did not start')
    def tearDown(self):
        self.proc.terminate()
        try:self.proc.wait(timeout=2)
        except Exception:self.proc.kill()
        self.backend.shutdown();self.backend.server_close();self.td.cleanup()
    def request(self,model='pushbutton/auto',text='hi',max_tokens=4):
        body=json.dumps({'model':model,'messages':[{'role':'user','content':text}],'max_tokens':max_tokens}).encode();req=urllib.request.Request(f'http://127.0.0.1:{self.broker_port}/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=5) as r:return json.load(r)
    def control(self,path,obj):
        body=json.dumps(obj).encode();req=urllib.request.Request(f'http://127.0.0.1:{self.broker_port}{path}',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=5) as r:return json.load(r)
    def test_logical_auto_model_is_translated(self):
        obj=self.request();self.assertEqual(obj['choices'][0]['message']['content'],'seen:backend-real-id')
    def test_c1_serializes_two_clients(self):
        out=[]
        ts=[threading.Thread(target=lambda:out.append(self.request())) for _ in range(2)]
        for t in ts:t.start()
        for t in ts:t.join(5)
        self.assertEqual(len(out),2);self.assertEqual(FakeState.max_active,1)
    def test_context_over_limit_is_rejected_before_backend(self):
        before=FakeState.calls
        with self.assertRaises(urllib.error.HTTPError) as cm:self.request(text='x'*1600,max_tokens=32)
        self.assertEqual(cm.exception.code,503);self.assertEqual(FakeState.calls,before)
    def test_maintenance_lease_queues_new_work_until_release(self):
        lease=self.control('/pushbutton/maintenance/acquire',{'id':'i1'});self.assertTrue(lease['ok'])
        out=[]
        t=threading.Thread(target=lambda:out.append(self.request()));t.start();time.sleep(.12)
        self.assertEqual(FakeState.calls,0);self.assertTrue(t.is_alive())
        rel=self.control('/pushbutton/maintenance/release',{'id':'i1'});self.assertTrue(rel['ok'])
        t.join(5);self.assertEqual(len(out),1);self.assertEqual(FakeState.calls,1)


if __name__=='__main__':unittest.main()
