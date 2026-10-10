import gzip, http.client, re, importlib.machinery, importlib.util, json, pathlib, sys, tempfile, threading, time, unittest
from http.server import ThreadingHTTPServer
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib'))
import pushbutton_metrics as m  # noqa: E402


def load_script(name):
    loader = importlib.machinery.SourceFileLoader(name.replace('-', '_'), str(ROOT / name))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


collector_mod = load_script('pushbutton-telemetry-collector')


class TelemetryClientTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.old = m.CONFIG, m.QUEUE
        m.CONFIG = pathlib.Path(self.td.name) / 'cfg'
        m.QUEUE = pathlib.Path(self.td.name) / 'queue'

    def tearDown(self):
        m.CONFIG, m.QUEUE = self.old
        self.td.cleanup()

    def test_nothing_queued_without_consent(self):
        self.assertIsNone(m.queue_performance('ornith-1.5:9b', 'llama.cpp', 'Q8_0', load_time_s=3.0))
        m.set_telemetry(False)
        self.assertIsNone(m.queue_performance('ornith-1.5:9b', 'llama.cpp', 'Q8_0', load_time_s=3.0))

    def test_performance_record_has_load_time_and_depths_only(self):
        m.set_telemetry(True, 'http://127.0.0.1:1/v1/telemetry')
        dev = m.device_descriptor({'model_name': 'Intel Xeon Phi 7250', 'family': 'knl', 'tier': 'knl',
                                   'sockets': 1, 'physical_cores': 68, 'fast_mem_kind': 'mcdram',
                                   'fast_mem_mode': 'flat', 'ram_total_mib': 196608, 'hostname': 'secret-host'},
                                  {'name': 'mcdram-bind', 'threads': 68, 'threads_batch': 136, 'memory_tier': 'mcdram'})
        p = m.queue_performance('ornith-1.5:9b', 'llama.cpp', 'Q8_0', load_time_s=12.3456,
                                depths=[{'depth': 0, 'pp_tps': 50.0, 'tg_tps': 12.0, 'prompt': 'SECRET'},
                                        {'depth': 4096, 'pp_tps': 45.0, 'tg_tps': 11.5}],
                                context=32768, device=dev)
        rec = json.loads(p.read_text().splitlines()[-1])
        self.assertEqual(rec['event'], 'performance')
        self.assertEqual(rec['load_time_s'], 12.346)
        self.assertEqual([d['depth'] for d in rec['depths']], [0, 4096])
        self.assertEqual((rec['device'], rec['cpu_family'], rec['strategy']), ('cpu', 'knl', 'mcdram-bind'))
        self.assertNotIn('SECRET', p.read_text())
        self.assertNotIn('secret-host', p.read_text())

    def test_uploads_at_most_once_per_five_minutes(self):
        m.set_telemetry(True, 'http://collector.invalid/v1/telemetry')
        sent = []

        class Resp:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def fake_urlopen(req, timeout=0):
            sent.append(gzip.decompress(req.data).decode())
            return Resp()

        with mock.patch.object(m.urllib.request, 'urlopen', side_effect=fake_urlopen):
            m.queue_performance('ornith-1.5:9b', 'llama.cpp', 'Q8_0', load_time_s=1.0)
            n, _ = m.maybe_upload()
            self.assertEqual(n, 1)
            for _ in range(50):  # large queues no longer bypass the interval
                m.queue_performance('ornith-1.5:9b', 'llama.cpp', 'Q8_0', load_time_s=1.0, depths=[{'depth': 0, 'tg_tps': 1.0}] * 30)
            self.assertEqual(m.maybe_upload()[0], 0)
            self.assertEqual(m.maybe_upload(force=True)[0], 0)
            self.assertEqual(m.upload_pending()[0], 0)
            self.assertEqual(len(sent), 1)
            cfg = m.telemetry_config()
            cfg['last_upload_epoch'] = cfg['last_attempt_epoch'] = int(time.time()) - m.MIN_UPLOAD_INTERVAL_S - 1
            m._write_json(m.CONFIG / 'telemetry.json', cfg)
            self.assertEqual(m.maybe_upload()[0], 50)
            self.assertEqual(len(sent), 2)

    def test_failed_upload_also_backs_off(self):
        m.set_telemetry(True, 'http://collector.invalid/v1/telemetry')
        m.queue_performance('ornith-1.5:9b', 'llama.cpp', 'Q8_0', load_time_s=1.0)
        with mock.patch.object(m.urllib.request, 'urlopen', side_effect=OSError('down')) as up:
            self.assertIn('deferred', m.maybe_upload()[1])
            self.assertIn('next upload', m.maybe_upload()[1])
            self.assertEqual(up.call_count, 1)

    def test_first_run_prompt_defaults_to_enabled_and_discloses_ip(self):
        src = (ROOT / 'pushbutton').read_text()
        self.assertIn("Enable telemetry? [Y/n]", src)
        self.assertIn("raw not in {'n','no'}", src)
        self.assertIn('print(metrics.CONSENT_TEXT', src)
        for phrase in ('keyed hash of your IP address', 'private repository', 'anonymized',
                       f'at least {m.PUBLIC_MIN_CONTRIBUTORS} different contributors', 'Disclaimer',
                       'never sold', 'once every 5 minutes', '--telemetry-off'):
            self.assertIn(phrase, m.CONSENT_TEXT)

    def test_older_consent_must_be_renewed_before_queueing(self):
        m._write_json(m.CONFIG / 'telemetry.json', {'enabled': True, 'consent_version': m.CONSENT_VERSION - 1,
                                                    'upload_url': 'http://collector.invalid'})
        self.assertFalse(m.telemetry_enabled())
        self.assertIsNone(m.queue_performance('ornith-1.5:9b', 'llama.cpp', 'Q8_0', load_time_s=1.0))
        self.assertIn('consent', m.maybe_upload()[1])

    def test_opt_out_clears_queue(self):
        m.set_telemetry(True)
        m.queue_performance('ornith-1.5:9b', 'llama.cpp', 'Q8_0', load_time_s=1.0)
        self.assertGreater(m.queued_bytes(), 0)
        m.set_telemetry(False)
        self.assertEqual(m.queued_bytes(), 0)

    def test_worker_allowlist_matches_python(self):
        js = (ROOT / 'telemetry/worker/src/index.js').read_text()
        def arr(name):
            body = re.search(rf'export const {name} = \[(.*?)\];', js, re.S).group(1)
            return tuple(re.findall(r'"([^"]+)"', body))
        self.assertEqual(arr('SANITIZED_FIELDS'), m.SANITIZED_FIELDS)
        self.assertEqual(arr('STRING_FIELDS'), m.STRING_FIELDS)
        self.assertEqual(arr('DEPTH_FIELDS'), m.DEPTH_FIELDS)
        self.assertEqual(collector_mod.sender_hash('203.0.113.7', b'salt'), '549fe5bec61ee86f')


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.records = pathlib.Path(self.td.name) / 'records'
        self.collector = collector_mod.Collector(self.records, salt=b'salt')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), collector_mod.make_handler(self.collector, False))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.td.cleanup()

    def post(self, body, headers=None):
        c = http.client.HTTPConnection('127.0.0.1', self.server.server_address[1], timeout=5)
        c.request('POST', '/v1/telemetry', body=body, headers=headers or {})
        r = c.getresponse()
        data = json.loads(r.read() or b'{}')
        return r.status, dict(r.getheaders()), data

    def test_appends_records_with_timestamp_and_hashed_sender_ip_then_rate_limits(self):
        lines = [json.dumps({'event': 'performance', 'model': 'ornith-1.5:9b', 'load_time_s': 4.2,
                             'depths': [{'depth': 0, 'pp_tps': 40, 'tg_tps': 10}], 'prompt': 'SECRET'}),
                 json.dumps({'model': 'qwen3.6:35b', 'tg': 20.0, 'prompt_tokens': 1000})]
        body = gzip.compress('\n'.join(lines).encode())
        status, _, data = self.post(body, {'Content-Encoding': 'gzip', 'Content-Type': 'application/x-ndjson'})
        self.assertEqual((status, data['accepted']), (200, 2))
        files = list(self.records.glob('*.ndjson'))
        self.assertEqual(len(files), 1)
        recs = [json.loads(x) for x in files[0].read_text().splitlines()]
        self.assertTrue(all(r['sender_hash'] == collector_mod.sender_hash('127.0.0.1', b'salt') for r in recs))
        self.assertNotIn('127.0.0.1', files[0].read_text())
        self.assertNotIn('sender_ip', files[0].read_text())
        self.assertTrue(all(r['received_utc'].endswith('Z') for r in recs))
        self.assertEqual(recs[0]['depths'][0]['tg_tps'], 10)
        self.assertNotIn('SECRET', files[0].read_text())
        status, headers, _ = self.post(b'{"model":"x"}\n')
        self.assertEqual(status, 429)
        self.assertGreater(int(headers['Retry-After']), 0)
        self.assertEqual(len(files[0].read_text().splitlines()), 2)

    def test_rejects_garbage_and_oversized(self):
        self.assertEqual(self.post(b'not json')[0], 400)
        self.assertEqual(self.post(b'x' * (collector_mod.MAX_BODY + 1))[0], 413)
        # rejected payloads must not consume the sender's 5-minute slot
        self.assertEqual(self.post(b'{"model":"x"}\n')[0], 200)

    def test_end_to_end_client_upload(self):
        old = m.CONFIG, m.QUEUE
        try:
            m.CONFIG = pathlib.Path(self.td.name) / 'cfg'
            m.QUEUE = pathlib.Path(self.td.name) / 'queue'
            m.set_telemetry(True, f'http://127.0.0.1:{self.server.server_address[1]}/v1/telemetry')
            m.queue_performance('ornith-1.5:9b', 'llama.cpp', 'Q8_0', load_time_s=2.5,
                                depths=[{'depth': 4096, 'pp_tps': 30.0, 'tg_tps': 9.0}])
            n, msg = m.maybe_upload()
            self.assertEqual(n, 1, msg)
            rec = json.loads(next(self.records.glob('*.ndjson')).read_text())
            self.assertEqual((rec['load_time_s'], rec['sender_hash']), (2.5, collector_mod.sender_hash('127.0.0.1', b'salt')))
        finally:
            m.CONFIG, m.QUEUE = old


aggregate_mod = load_script('telemetry/aggregate.py')


class AggregateTests(unittest.TestCase):
    def rec(self, sender, model='ornith-1.5:9b', depth=4096, tg=10.0, **kw):
        return {'event': 'performance', 'model': model, 'artifact': 'Q8_0', 'backend': 'llama.cpp',
                'device': 'cpu', 'cpu_model': 'Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz', 'cpu_sockets': 2,
                'strategy': 'numa-distribute', 'load_time_s': 5.0, 'received_utc': '2026-10-10T10:00:00Z',
                'sender_hash': sender, 'depths': [{'depth': depth, 'pp_tps': 50.0, 'tg_tps': tg,
                                                    'tg_tps_estimate': 999}], **kw}

    def test_k_anonymity_and_no_identifying_fields(self):
        recs = [self.rec(f's{i}', tg=10 + i) for i in range(3)]
        recs += [self.rec('lonely', model='rare-model:1b')]
        recs += [self.rec(None, sender_ip='198.51.100.9', model='legacy:7b')]
        s = aggregate_mod.aggregate(recs, 3)
        text = json.dumps(s)
        for leak in ('s0', 'lonely', 'rare-model', '198.51.100.9', 'sender', 'received_utc', '2026-10-10T10', 'legacy'):
            self.assertNotIn(leak, text)
        self.assertEqual(len(s['performance']), 1)
        row = s['performance'][0]
        self.assertEqual((row['hardware'], row['depth'], row['contributors']), ('2x Intel Xeon Gold 6248', '≤4K', 3))
        self.assertEqual(row['tg_tps']['median'], 11.0)
        self.assertEqual(s['load_time'][0]['load_time_s']['median'], 5.0)
        self.assertEqual(s['groups_suppressed'], 4)

    def test_records_without_sender_are_ignored_and_unsafe_strings_collapsed(self):
        recs = [self.rec(f's{i}', model='<script>alert(1)</script>') for i in range(3)]
        recs.append(self.rec(None))
        s = aggregate_mod.aggregate(recs, 3)
        self.assertEqual(s['performance'][0]['model'], 'other')
        self.assertEqual(s['performance'][0]['contributors'], 3)

    def test_cli_writes_dashboard_with_escaped_data(self):
        with tempfile.TemporaryDirectory() as td:
            d = pathlib.Path(td) / 'records' / '2026' / '10'
            d.mkdir(parents=True)
            (d / 'a.ndjson').write_text('\n'.join(json.dumps(self.rec(f's{i}')) for i in range(3)) + '\nnot json\n')
            out = pathlib.Path(td) / 'site'
            self.assertEqual(aggregate_mod.main([str(pathlib.Path(td) / 'records'), '--out', str(out)]), 0)
            html = (out / 'index.html').read_text()
            self.assertNotIn('/*__SUMMARY__*/null', html)
            self.assertIn('ornith-1.5:9b', html)
            self.assertNotIn('sender_hash', html + (out / 'summary.json').read_text())
            with self.assertRaises(SystemExit):
                aggregate_mod.main([str(d), '--out', str(out), '--min-contributors', '1'])


if __name__ == '__main__':
    unittest.main()
