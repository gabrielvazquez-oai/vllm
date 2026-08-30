import json, os, pathlib, subprocess, tempfile, unittest

ROOT = pathlib.Path(__file__).resolve().parent
WRAPPER = ROOT.parent / 'codex-security-diagnostics.py'


class DiagnosticsTests(unittest.TestCase):
    def run_wrapper(self, script, input_text='private-request-body\n'):
        directory = pathlib.Path(tempfile.mkdtemp(prefix='diagnostics-test-', dir=ROOT))
        self.addCleanup(__import__('shutil').rmtree, directory)
        real_cli = directory / 'fake-codex'
        real_cli.write_text('#!/usr/bin/env python3\n' + script)
        real_cli.chmod(0o700)
        env = dict(os.environ, CODEX_HOME=str(directory), CODEX_CLI_PATH=str(WRAPPER),
                   CODEX_DIAGNOSTICS_REAL_CLI=str(real_cli))
        process = subprocess.run(['python3', str(WRAPPER), 'app-server', '--config', 'model="unchanged"'],
                                 input=input_text, text=True, capture_output=True, env=env, timeout=10)
        log = directory / 'app-server-diagnostics.jsonl'
        self.assertEqual(log.stat().st_mode & 0o777, 0o600)
        return process, log.read_text()

    def test_protocol_passthrough_and_selected_errors(self):
        process, diagnostic = self.run_wrapper('''
import json, os, sys
assert sys.argv[1:] == ['app-server','--config','model="unchanged"']
assert 'CODEX_CLI_PATH' not in os.environ
assert 'CODEX_DIAGNOSTICS_REAL_CLI' not in os.environ
assert sys.stdin.read() == 'private-request-body\\n'
events = [
  {'id':1,'result':{'sensitive_metadata':'DO_NOT_RECORD'}},
  {'method':'item/completed','params':{'item':{'type':'agentMessage','text':'DO_NOT_RECORD'}}},
  {'method':'error','params':{'error':{'message':'upstream rejected request','request_id':'req_123'},'willRetry':False}},
  {'method':'turn/completed','params':{'threadId':'thread_1','turn':{'id':'turn_1','status':'failed','error':{'message':'concrete cause'}}}},
  {'id':9,'error':{'code':400,'message':'rpc failed'}},
]
for event in events: print(json.dumps(event), flush=True)
sys.stderr.write('separate stderr evidence\\n')
sys.exit(7)
''')
        self.assertEqual(process.returncode, 7)
        self.assertEqual(len(process.stdout.splitlines()), 5)
        self.assertEqual(process.stderr, 'separate stderr evidence\n')
        self.assertNotIn('DO_NOT_RECORD', diagnostic)
        self.assertNotIn('private-request-body', diagnostic)
        for text in ['req_123','concrete cause','rpc failed','separate stderr evidence']:
            self.assertIn(text, diagnostic)
        self.assertEqual(json.loads(diagnostic.splitlines()[-1])['returncode'], 7)

    def test_fragmented_stderr_and_oversized_protocol_line(self):
        process, diagnostic = self.run_wrapper('''
import os, sys, time
sys.stdout.write('x' * (300 * 1024) + '\\n')
sys.stdout.flush()
os.write(2, b'Bearer sk-fragment-')
time.sleep(0.03)
os.write(2, b'complete\\n')
''')
        self.assertEqual(process.returncode, 0)
        self.assertEqual(len(process.stdout), 300 * 1024 + 1)
        self.assertIn('Bearer sk-fragment-complete', diagnostic)
        self.assertIn('Oversized stdout line omitted', diagnostic)
        self.assertNotIn('x' * 50, diagnostic)

    def test_log_size_cap_preserves_process_output(self):
        process, diagnostic = self.run_wrapper('''
import sys
for i in range(400): sys.stderr.write('evidence ' + 'x' * 10000 + '\\n')
''')
        self.assertEqual(process.returncode, 0)
        self.assertGreater(len(process.stderr), 3 * 1024 * 1024)
        self.assertLess(len(diagnostic), 2 * 1024 * 1024 + 200)
        self.assertIn('Log size limit reached', diagnostic)


if __name__ == '__main__':
    unittest.main(verbosity=2)
