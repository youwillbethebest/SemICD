"""Focused regression checks for the public synthetic walkthrough."""
import json
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from pathlib import Path

from semicd.data.icd import normalize_code
from semicd.training.config import resolve_config
from semicd.training.tasks import prepare_training

ROOT = Path(__file__).resolve().parents[1]


class CodeNormalizationTests(unittest.TestCase):
    def test_declared_diagnosis_formats(self):
        cases = [(' 001.0 ', 'ICD-9', '0010'), ('v10.3', 'ICD-9', 'V103'),
                 ('E812.0', 'ICD-9', 'E8120'), (' j45.909 ', 'ICD-10-CM', 'J45909'),
                 ('I10', 'ICD-10-CM', 'I10')]
        for value, system, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(normalize_code(value, system), expected)
                self.assertEqual(normalize_code(expected, system), expected)

    def test_invalid_input_contract(self):
        for value in (None, 10, 1.0):
            with self.assertRaises(TypeError):
                normalize_code(value, 'ICD-9')
        for value in ('', ' ', '.'):
            with self.assertRaises(ValueError):
                normalize_code(value, 'ICD-9')
        with self.assertRaises(ValueError):
            normalize_code('I10', 'synthetic')


class SyntheticPreparationTests(unittest.TestCase):
    def test_readme_tasks_and_inferred_bundle(self):
        fixture = ROOT / 'examples/synthetic'
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'tasks'
            report = prepare_training(Namespace(train_file=fixture / 'train.jsonl',
                profiles_file=fixture / 'profiles.jsonl', sid_bundle=fixture / 'bundle', output_dir=output))
            rows = [json.loads(line) for line in (output / 'train.jsonl').read_text().splitlines()]
            self.assertEqual(Counter(row.get('task', 'note_to_sid') for row in rows),
                             {'note_to_sid': 6, 'profile_to_sid': 3, 'sid_to_profile': 3})
            self.assertEqual({c for row in rows for c in row.get('codes', [])}, {'SYN:A', 'SYN:B', 'SYN:C'})
            self.assertFalse(report['test_used'])
            config = resolve_config(Namespace(train_file=str(output / 'train.jsonl'),
                validation_file=str(fixture / 'validation.jsonl'), model_name_or_path='tiny',
                output_dir=str(Path(directory) / 'run')))
            self.assertEqual(config.sid_bundle, str((fixture / 'bundle').resolve()))
            self.assertEqual(config.representation, 'semantic')


if __name__ == '__main__':
    unittest.main()
