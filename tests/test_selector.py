import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
SEL = ROOT / 'pushbutton-select'
loader = importlib.machinery.SourceFileLoader('selector', str(SEL))
spec = importlib.util.spec_from_loader(loader.name, loader)
selector = importlib.util.module_from_spec(spec)
loader.exec_module(selector)
runtime_loader = importlib.machinery.SourceFileLoader('runtime', str(ROOT / 'pushbutton'))
runtime_spec = importlib.util.spec_from_loader(runtime_loader.name, runtime_loader)
runtime = importlib.util.module_from_spec(runtime_spec)
runtime_loader.exec_module(runtime)

class SelectorTests(unittest.TestCase):
    def run_sel(self, *args, gpus=None, tty=False, inputs=()):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, 'argv', [str(SEL), *args]))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            stack.enter_context(mock.patch.object(sys.stdin, 'isatty', return_value=tty))
            stack.enter_context(mock.patch.object(out, 'isatty', return_value=tty))
            stack.enter_context(mock.patch('builtins.input', side_effect=inputs))
            inventory = stack.enter_context(mock.patch.object(selector, 'gpu_inventory', return_value=gpus or []))
            stack.enter_context(mock.patch.object(selector.metrics, '_remote_rows', side_effect=selector.metrics._bundled_rows))
            stack.enter_context(mock.patch.object(selector.metrics, 'local_rows', return_value=[]))
            stack.enter_context(mock.patch.object(selector.workers, 'load_placement_config', return_value={}))
            launch = stack.enter_context(mock.patch.object(selector.subprocess, 'call', return_value=0))
            code = selector.run()
        return code, out.getvalue(), err.getvalue(), launch, inventory

    def test_16g_qwen38_blocks_ninfer_and_fits_iq3(self):
        code, out, _, launch, _ = self.run_sel(
            'qwen3.8:27b', '--gpu', '0', '--vram-limit', '16G',
            '--no-interactive', gpus=selector.workers.synthetic_3090(1))
        self.assertEqual(code, 0)
        self.assertIn('IQ3_XXS', out)
        self.assertRegex(out, r'BLOCK\s+\S+\s+ninfer-3090')
        self.assertRegex(out, r'UNVERIFIED\s+\S+\s+llamampere')
        self.assertIn('Recommended launch:', out)
        launch.assert_not_called()

    def test_24g_table_labels_upstream_as_reference_not_local(self):
        code, out, _, _, _ = self.run_sel(
            'qwen3.8:27b', '--gpu', '0', '--vram-limit', '24G',
            '--no-interactive', gpus=selector.workers.synthetic_3090(1))
        self.assertEqual(code, 0)
        self.assertIn('vllm-qwen38-3090', out)
        self.assertIn('127', out)
        self.assertIn('llamampere', out)
        self.assertRegex(out, r'llamampere.*UNKNOWN')
        self.assertRegex(out, r'127\s+UPSTREAM REFERENCE')
        self.assertNotIn('ESTIMATED', out)
        self.assertIn('25 TG/s', out)

    def test_noninteractive_no_model_does_not_inventory_or_read(self):
        code, out, _, launch, inventory = self.run_sel()
        self.assertEqual(code, 0)
        self.assertIn('--select', out)
        self.assertIn('No provisioning attempted', out)
        inventory.assert_not_called()
        launch.assert_not_called()
        cp = subprocess.run([sys.executable, str(SEL)], input='', text=True,
                            capture_output=True, timeout=5)
        self.assertEqual(cp.returncode, 0)
        self.assertIn('No model supplied', cp.stdout)

    def test_telemetry_is_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(selector.metrics, 'CONFIG', pathlib.Path(td)):
                code, _, err, _, _ = self.run_sel(
                    'qwen3.8:27b', '--vram-limit', '16G', '--no-interactive',
                    '--telemetry-opt-in', gpus=selector.workers.synthetic_3090(1))
            cfg = json.loads((pathlib.Path(td) / 'telemetry.json').read_text())
            self.assertEqual(code, 0)
            self.assertTrue(cfg['enabled'])
            self.assertIn('Opt-in telemetry enabled', err)

    def test_help_before_inventory(self):
        with self.assertRaises(SystemExit) as cm:
            with mock.patch.object(selector, 'gpu_inventory', side_effect=AssertionError('hardware accessed')):
                self.run_sel('--help')
        self.assertEqual(cm.exception.code, 0)

    def test_welcome_invalid_help_and_quit_before_inventory(self):
        code, out, _, launch, inventory = self.run_sel(tty=True, inputs=['invalid', '4', '5'])
        self.assertEqual(code, 0)
        self.assertIn('Welcome', out)
        self.assertIn('Please choose', out)
        self.assertIn('usage:', out)
        inventory.assert_not_called()
        launch.assert_not_called()

    def test_eof_and_ctrl_c_cancel_without_provisioning(self):
        for exception in (EOFError(), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                code, _, err, launch, inventory = self.run_sel(tty=True, inputs=[exception])
                self.assertEqual(code, 0)
                self.assertIn('Cancelled', err)
                launch.assert_not_called()
                inventory.assert_not_called()

    def test_multimodel_joint_plan_leases_and_spare_cards(self):
        code, out, _, launch, _ = self.run_sel(
            'qwen3.8-flash-next', 'qwen3.8:27b@gpu=4,vram=30G',
            '--agents', '2', '--json', gpus=selector.workers.synthetic_v100(6))
        obj = json.loads(out)
        self.assertEqual(code, 0, obj)
        self.assertEqual([len(w['gpus']) for w in obj['plan']['workers']], [4, 1])
        self.assertEqual(obj['plan']['workers'][0]['profile']['quant'], 'UD-IQ4_XS')
        self.assertEqual(obj['plan']['workers'][1]['vram_limit_mib_per_gpu'], 30720)
        self.assertEqual(len(obj['plan']['unused_gpus']), 1)
        self.assertIn('qwen3.8:27b@gpu=4,vram=30720MiB', obj['launch_argv'])
        flash = obj['rows']['qwen3.8-flash-next']
        self.assertEqual(len([r for r in flash if r['backend'] == 'llama.cpp']), 6)
        self.assertTrue(all(r['pp'] is None and r['tg'] is None for r in flash))
        adapter = next(r for r in flash if r['backend'] == 'sglang-v100')
        self.assertEqual(adapter['support'], 'experimental')
        self.assertFalse(adapter['frontend_integrated'])
        self.assertEqual(adapter['constraints']['gpu_count'], 4)
        self.assertIsInstance(adapter['validation_argv'], list)
        launch.assert_not_called()

    def test_independent_flash_replicas_preserved(self):
        code, out, _, _, _ = self.run_sel(
            'q38next', '--agents', '2', '--json', gpus=selector.workers.synthetic_v100(8))
        self.assertEqual(code, 0)
        ws = json.loads(out)['plan']['workers']
        self.assertEqual([len(w['gpus']) for w in ws], [4, 4])
        self.assertTrue(set(g['index'] for g in ws[0]['gpus']).isdisjoint(
            g['index'] for g in ws[1]['gpus']))

    def test_claude_and_hermes_share_roles_not_replicas(self):
        for frontend in ('claude-local', 'hermes-local'):
            with self.subTest(frontend=frontend):
                code, out, _, _, _ = self.run_sel(
                    'qwen3.8:27b', 'qwen3.8:27b', '--frontend', frontend,
                    '--json', gpus=selector.workers.synthetic_v100(2))
                obj = json.loads(out)
                self.assertEqual(code, 0, obj)
                self.assertEqual(len(obj['plan']['servers']), 1)
                self.assertIn('roles', obj['plan'])
                self.assertNotIn('--agents', obj['launch_argv'])
                self.assertEqual(obj['launch_env']['PUSHBUTTON_SELECTOR_GPU_INDICES'], '0')
                if frontend == 'claude-local':
                    self.assertEqual(obj['launch_argv'][:2], ['bash', str(ROOT / 'lib/claude_local_entry.sh')])

    def test_role_launch_rechecks_only_preview_gpu_pool(self):
        gpus = selector.workers.synthetic_3090(1) + [
            selector.plan.GPU(1, 'Unsupported GPU', 65536, 65536, '6.1')]
        for frontend in ('claude-local', 'hermes-local'):
            with self.subTest(frontend=frontend):
                code, _, _, launch, _ = self.run_sel(
                    'qwen3.8:27b', '--frontend', frontend, tty=True,
                    inputs=['yes'], gpus=gpus)
                self.assertEqual(code, 0)
                self.assertEqual(launch.call_args.kwargs['env']['PUSHBUTTON_SELECTOR_GPU_INDICES'], '0')
        out = io.StringIO()
        with mock.patch.object(sys, 'argv', ['planner', 'plan', 'qwen3.8:27b']), \
             mock.patch.object(selector.plan, 'inventory', return_value=gpus), \
             mock.patch.dict(os.environ, {'PUSHBUTTON_SELECTOR_GPU_INDICES': '0'}), \
             contextlib.redirect_stdout(out):
            self.assertEqual(selector.plan.main(), 0)
        self.assertEqual(json.loads(out.getvalue())['servers'][0]['cuda_visible_devices'], '0')

    def test_role_replan_rejects_missing_or_invalid_preview_pool(self):
        for pool in ('99', '', 'invalid'):
            with self.subTest(pool=pool):
                err = io.StringIO()
                with mock.patch.object(sys, 'argv', ['planner', 'plan', 'qwen3.8:27b']), \
                     mock.patch.object(selector.plan, 'inventory', return_value=selector.workers.synthetic_3090(1)), \
                     mock.patch.dict(os.environ, {'PUSHBUTTON_SELECTOR_GPU_INDICES': pool}), \
                     contextlib.redirect_stderr(err):
                    self.assertEqual(selector.plan.main(), 2)
                self.assertIn('selector GPU pool', err.getvalue())

    def test_insufficient_vram_overlap_and_no_gpu_do_not_launch(self):
        cases = [
            ([], ['qwen3.8:27b']),
            (selector.workers.synthetic_3090(1), ['qwen3.8:27b@gpu=0,vram=12G']),
            (selector.workers.synthetic_v100(2), ['qwen3.8:27b@gpu=0', 'qwen3.6:35b@gpu=0']),
            ([selector.plan.GPU(0, 'unsupported GPU', 32768, 32768, '6.1')], ['qwen3.8:27b']),
        ]
        for gpus, models in cases:
            with self.subTest(models=models, gpus=gpus):
                code, out, _, launch, _ = self.run_sel(*models, '--json', gpus=gpus)
                self.assertEqual(code, 2)
                self.assertIsNotNone(json.loads(out)['error'])
                launch.assert_not_called()

    def test_context_clamp_and_invalid_client_context(self):
        code, out, _, _, _ = self.run_sel(
            'qwen3.8:27b', '--context', '8192', '--json',
            gpus=selector.workers.synthetic_3090(1))
        self.assertEqual(code, 0)
        obj = json.loads(out)
        self.assertEqual(obj['client_context'], 8192)
        argv = obj['launch_argv']
        self.assertEqual(argv[argv.index('--local-client-context') + 1], '8192')
        code, _, err, launch, _ = self.run_sel(
            'qwen3.8:27b', '--context', '8192', '--client-context', '200000',
            gpus=selector.workers.synthetic_3090(1))
        self.assertEqual(code, 2)
        self.assertIn('no larger', err)
        launch.assert_not_called()

    def test_confirmation_preserves_frontend_cwd_and_argv_without_shell(self):
        cwd = os.getcwd()
        code, _, _, launch, _ = self.run_sel(
            'qwen3.8:27b', '--frontend', 'opencode-local',
            '--launch-arg=--resume', '--launch-arg=--',
            '--launch-arg=a prompt; $(not-a-command)',
            tty=True, inputs=['yes'], gpus=selector.workers.synthetic_3090(1))
        self.assertEqual(code, 0)
        argv = launch.call_args.args[0]
        self.assertIsInstance(argv, list)
        self.assertEqual(argv[0], str(ROOT / 'opencode-local'))
        self.assertEqual(argv[-3:], ['--resume', '--', 'a prompt; $(not-a-command)'])
        self.assertEqual(os.getcwd(), cwd)

    def test_reject_opaque_context_override(self):
        code, _, err, launch, inventory = self.run_sel(
            'qwen3.8:27b', '--launch-arg=--local-context', '--launch-arg=1')
        self.assertEqual(code, 2)
        self.assertIn('changes the plan', err)
        inventory.assert_not_called()
        launch.assert_not_called()

    def test_declined_confirmation_and_print_only_never_launch(self):
        for extra, tty, inputs in (([], True, ['no']), (['--plan-only'], True, []),
                                   (['--no-interactive'], False, [])):
            code, _, _, launch, _ = self.run_sel(
                'qwen3.8:27b', *extra, tty=tty, inputs=inputs,
                gpus=selector.workers.synthetic_3090(1))
            self.assertEqual(code, 0)
            launch.assert_not_called()

    def test_invalid_model_menu_input_retries(self):
        code, out, _, launch, _ = self.run_sel(
            tty=True, inputs=['3', '999 invalid', 'qwen3.8:27b', '8192', '1', '', '16G'],
            gpus=selector.workers.synthetic_3090(1))
        self.assertEqual(code, 0)
        self.assertIn('Please enter valid', out)
        self.assertIn('Joint placement preview', out)
        launch.assert_not_called()

    def test_experimental_flash_eligibility_is_not_fit(self):
        registry = json.loads((ROOT / 'configs/backend-registry.json').read_text())['backends']
        sglang = registry['sglang-v100']
        vllm = registry['vllm-flashnext-3090']
        check = selector.adapter_eligibility
        self.assertEqual(check(sglang, selector.workers.synthetic_v100(4), 262144, None)[0], 'UNKNOWN')
        self.assertEqual(check(sglang, selector.workers.synthetic_v100(3), 262144, None)[0], 'UNAVAILABLE')
        self.assertEqual(check(sglang, selector.workers.synthetic_3090(4), 262144, None)[0], 'UNAVAILABLE')
        self.assertEqual(check(vllm, selector.workers.synthetic_3090(4), 65536, None)[0], 'UNKNOWN')
        self.assertEqual(check(vllm, selector.workers.synthetic_3090(4), 262144, None)[0], 'UNAVAILABLE')
        self.assertEqual(check(vllm, [], 65536, None)[0], 'UNKNOWN')
        occupied = [selector.plan.GPU(g.index, g.name, g.total_mib, 1024, g.compute_cap)
                    for g in selector.workers.synthetic_3090(4)]
        self.assertEqual(check(vllm, occupied, 65536, None)[0], 'UNKNOWN')

    def test_local_evidence_is_separate_and_requires_quant(self):
        rows = [{'evidence_origin': 'reference', 'tg': 99}, {'evidence_origin': 'local', 'context': 8192, 'tg': 42}]
        with mock.patch.object(selector.metrics, 'observations', return_value=rows) as observations:
            obs, summary = selector.measured_summary('qwen3.8:27b', 'llama.cpp', 'RTX 3090', 8192, 'IQ3_XXS')
        self.assertEqual(summary['tg']['median'], 42)
        self.assertEqual(selector.evidence_label(obs, summary), 'LOCAL MEASURED')
        self.assertEqual(observations.call_args.kwargs['artifact'], 'IQ3_XXS')

    def test_mismatched_local_context_does_not_become_verified(self):
        rows = [{'evidence_origin': 'reference', 'tg': 99}, {'evidence_origin': 'local', 'context': 8192, 'tg': 42}]
        with mock.patch.object(selector.metrics, 'observations', return_value=rows):
            obs, summary = selector.measured_summary('qwen3.8:27b', 'llama.cpp', 'RTX 3090', 65536, 'IQ3_XXS')
        self.assertEqual(summary['tg']['median'], 99)
        self.assertEqual(selector.evidence_label(obs, summary), 'UPSTREAM REFERENCE')

    def test_telemetry_only_no_model_is_explicit_and_nonblocking(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(selector.metrics, 'CONFIG', pathlib.Path(td)):
                code, _, _, launch, inventory = self.run_sel('--telemetry-opt-in', '--json')
            self.assertTrue(json.loads((pathlib.Path(td) / 'telemetry.json').read_text())['enabled'])
        self.assertEqual(code, 0)
        launch.assert_not_called()
        inventory.assert_not_called()

    def test_occupied_memory_is_not_total_capacity(self):
        occupied = [selector.plan.GPU(0, 'RTX 3090', 24576, 4096, '8.6')]
        code, out, _, launch, _ = self.run_sel('qwen3.8:27b', '--json', gpus=occupied)
        obj = json.loads(out)
        self.assertEqual(code, 2)
        self.assertEqual(obj['inventory'][0]['free_mib'], 4096)
        self.assertTrue(all(r['status'] == 'BLOCK' for r in obj['rows']['qwen3.8:27b']
                            if r['backend'] == 'llama.cpp'))
        launch.assert_not_called()

    def test_no_model_json_is_guidance_not_default_selection(self):
        code, out, _, launch, inventory = self.run_sel('--json', '--no-interactive')
        self.assertEqual(code, 0)
        obj = json.loads(out)
        self.assertEqual(obj['models'], [])
        self.assertIsNone(obj['plan'])
        self.assertIsNone(obj['launch_argv'])
        launch.assert_not_called()
        inventory.assert_not_called()

    def test_catalog_aliases_capabilities_and_speed_policy_are_preserved(self):
        code, out, _, launch, _ = self.run_sel(
            'ornith-9b', '--json', gpus=selector.workers.synthetic_3090(1))
        obj = json.loads(out)
        self.assertEqual(code, 2)  # Catalog inspection is not a joint frontend profile.
        self.assertEqual(obj['speed_floor_tg'], 25.0)
        self.assertIn('SWE-bench Verified', obj['capability_benchmarks']['ornith-1.5:9b'])
        rows = obj['rows']['ornith-1.5:9b']
        self.assertTrue(any(r['status'] == 'FIT' for r in rows))
        self.assertTrue(all(r['tg'] is None and r['evidence'] == 'UNKNOWN' for r in rows))
        self.assertTrue(all('speed_status' in r for r in rows))
        self.assertTrue(all(not r['frontend_integrated'] for r in rows))
        launch.assert_not_called()

    def test_unified_runtime_consumes_joint_json_and_catalog_only_rows(self):
        row = {'model': 'ornith-1.5:9b', 'backend': 'llama.cpp',
               'artifact': 'Q4_K_M', 'status': 'FIT', 'tg': None, 'req': 9000}
        response = subprocess.CompletedProcess(
            [], 2, json.dumps({'rows': {'ornith-1.5:9b': [row]},
                              'error': 'no joint frontend profile'}))
        with mock.patch.object(runtime.plan, 'inventory', return_value=selector.workers.synthetic_3090(1)), \
             mock.patch.object(runtime.subprocess, 'run', return_value=response):
            rows = runtime.selector_views('ornith-9b', 65536)
        self.assertEqual(rows, [{**row, 'gpu': 0}])
        self.assertEqual(runtime.choose_candidate(rows), rows[0])

    def test_unified_runtime_does_not_treat_reference_rates_as_local(self):
        reference = {'status': 'FIT', 'tg': 100, 'evidence': 'UPSTREAM REFERENCE',
                     'quality': 95, 'req': 10000}
        local = {**reference, 'tg': 90, 'evidence': 'LOCAL MEASURED'}
        self.assertEqual(runtime.choose_candidate([reference, local]), local)

    def test_catalog_large_models_remain_guidance_not_automatic_adapter_fit(self):
        rows = selector.rows_for('deepseek-v4.1-flash', None, None, None, 65536,
                                 selector.workers.synthetic_v100(24))
        self.assertEqual(rows[0]['status'], 'UNVERIFIED')
        self.assertEqual(rows[0]['evidence'], 'UNKNOWN')
        self.assertFalse(rows[0]['frontend_integrated'])
        self.assertNotIn('validation_argv', rows[0])

if __name__ == '__main__':
    unittest.main()
