import importlib.util
import pathlib
import unittest

spec = importlib.util.spec_from_file_location('guard', pathlib.Path(__file__).resolve().parents[1] / 'tools/check_secrets.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

class SecretGuardTests(unittest.TestCase):
    def test_provider_shapes(self):
        for token in (b'AQ.' + b'x' * 50, b'AIza' + b'x' * 35,
                      b'ghp_' + b'x' * 36, b'hf_' + b'x' * 34):
            with self.subTest(prefix=token[:3]):
                self.assertTrue(list(guard.findings(token)))

    def test_location_without_value(self):
        hits = list(guard.findings(b'first line\n' + b'AQ.' + b'x' * 50))
        self.assertEqual(hits, [('google-aq', 2)])

    def test_placeholders(self):
        self.assertEqual(list(guard.findings(b'API_KEY="your_api_key_here"')), [])

    def test_generic_assignment(self):
        self.assertTrue(list(guard.findings(b'api_key="' + b'a9' * 16 + b'"')))

if __name__ == '__main__':
    unittest.main()
