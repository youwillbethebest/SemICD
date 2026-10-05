"""Mixed ICD-9 fixtures are invented and contain no clinical/licensed records."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from semicd.data.icd import catalog_label_space, typed_icd9
from semicd.data.icd9_inputs import prepare_icd9_inputs
from semicd.training.codec import make_codec
from semicd.training.prompts import paper_messages


class ICD9InputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.codes = ['DIAG:0031', 'PROC:0031']
        (self.root / 'codes.json').write_text(json.dumps(self.codes))
        fields = ['code_norm', 'official_code', 'code_kind', 'chapter_text', 'block_text', 'category_text', 'description']
        with (self.root / 'catalog.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
            for code, official, kind in [('DIAG:0031', '003.1', 'diagnosis'), ('PROC:0031', '00.31', 'procedure')]:
                writer.writerow(dict(zip(fields, [code, official, kind, 'Chapter', 'Block', 'Category', 'Artificial ' + kind])))
        with (self.root / 'paths.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['code_norm', 'chapter_id', 'block_id', 'category_id', 'leaf_id']); writer.writeheader()
            for code, leaf in [('DIAG:0031', '003.1'), ('PROC:0031', '00.31')]:
                writer.writerow(dict(code_norm=code, chapter_id=code[0]+'1', block_id=code[0]+'2', category_id=code[0]+'3', leaf_id=leaf))
        (self.root / 'terms.json').write_text(json.dumps({'003.1': ['Invented diagnosis term'], '00.31': ['Invented procedure term']}))
        (self.root / 'contract.json').write_text('{}')

    def args(self, name='inputs'):
        return Namespace(codes_file=self.root / 'codes.json', catalog_file=self.root / 'catalog.csv',
            paths_file=self.root / 'paths.csv', umls_terms_file=self.root / 'terms.json',
            source_contract=self.root / 'contract.json', output_dir=self.root / name)

    def run_cli(self, *args):
        result = subprocess.run([sys.executable, '-m', 'semicd', *map(str, args)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_cli_chain_preserves_overlapping_codes_and_unknown_metadata(self):
        args = self.args()
        command = ['prepare-icd9-inputs']
        for flag in ('codes_file', 'catalog_file', 'paths_file', 'umls_terms_file', 'source_contract', 'output_dir'):
            command += ['--' + flag.replace('_', '-'), getattr(args, flag)]
        report = self.run_cli(*command)
        self.assertEqual(report['code_types'], {'DIAG': 1, 'PROC': 1})
        self.assertIsNone(report['source_metadata']['ontology_edition'])
        ontology = [json.loads(s) for s in (args.output_dir / 'ontology.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['billable'] is None for r in ontology))
        terms = json.loads((args.output_dir / 'umls_terms.json').read_text())
        self.assertNotEqual(terms[self.codes[0]], terms[self.codes[1]])
        report = self.run_cli('prepare-profiles', '--ontology-file', args.output_dir / 'ontology.jsonl',
            '--umls-terms-file', args.output_dir / 'umls_terms.json', '--codes-file', args.codes_file,
            '--output-dir', self.root / 'profiles')
        self.assertEqual(report['label_space'], 'mixed')
        self.assertIn('not verified', report['code_validation'])
        for representation in ('atomic', 'raw'):
            bundle = self.root / representation
            report = self.run_cli('build-codebook', '--representation', representation, '--codes-file', args.codes_file,
                                 '--code-system', 'ICD-9', '--output-dir', bundle)
            self.assertEqual(report['label_space'], 'mixed')
            if representation == 'raw':
                self.assertIn('procedure', report['raw_prompt'])
                from semicd.codebook.bundle import read_bundle
                class TextTokenizer:
                    eos_token_id = 1000
                    def encode(self, text, **kwargs): return list(map(ord, text))
                    def decode(self, ids, **kwargs): return ''.join(map(chr, ids))
                codec = make_codec(TextTokenizer(), read_bundle(bundle))
                self.assertEqual(codec.decode(codec.encode(self.codes)), (self.codes, True, True))
            (self.root / 'train.jsonl').write_text(json.dumps({'prompt': 'Synthetic note', 'codes': self.codes})+'\n')
            report = self.run_cli('prepare-training', '--train-file', self.root / 'train.jsonl',
                '--profiles-file', self.root / 'profiles/profiles.jsonl', '--sid-bundle', bundle,
                '--output-dir', self.root / (representation + '-tasks'))
            self.assertEqual(report['rows'], 5 if representation == 'atomic' else 1)

    def test_semantic_and_random_bundles_keep_typed_identity(self):
        import numpy as np
        from semicd.codebook.bundle import read_bundle
        root = self.root
        profiles = root / 'synthetic_profiles.jsonl'
        profiles.write_text(''.join(json.dumps({'code': c, 'code_system': 'ICD-9',
            'edition': 'SYNTHETIC', 'profile': 'Artificial ' + c}) + '\n' for c in self.codes))
        cache = root / 'embeddings'; cache.mkdir()
        (cache / 'codes.json').write_text(json.dumps(self.codes))
        (cache / 'manifest.json').write_text('{}')
        np.save(cache / 'embeddings.npy', np.array([[1., 0.], [0., 1.]], dtype=np.float32))
        self.run_cli('build-codebook', '--profiles-file', profiles, '--embedding-dir', cache,
                     '--output-dir', root / 'semantic')
        self.run_cli('build-codebook', '--representation', 'random', '--source-bundle', root / 'semantic',
                     '--output-dir', root / 'random')
        class Tokenizer:
            def __init__(self): self.tokens = {}
            def encode(self, token, **kwargs):
                return [self.tokens.setdefault(token, len(self.tokens))]
            def convert_ids_to_tokens(self, index):
                return next(token for token, value in self.tokens.items() if value == index)
        for name in ('semantic', 'random'):
            bundle = read_bundle(root / name)
            self.assertEqual(bundle['manifest']['label_space'], 'mixed')
            codec = make_codec(Tokenizer(), bundle)
            self.assertEqual(codec.decode(codec.encode(self.codes)), (self.codes, True, True))

    def test_bad_path_and_missing_raw_terms(self):
        self.root.joinpath('terms.json').write_text('{}')
        report = prepare_icd9_inputs(self.args())
        self.assertEqual(report['unmatched_umls_codes'], self.codes)
        text = (self.root / 'paths.csv').read_text().replace('00.31', '003.1')
        self.root.joinpath('paths.csv').write_text(text)
        with self.assertRaises(ValueError):
            prepare_icd9_inputs(self.args('bad'))
        self.assertFalse((self.root / 'bad/ontology.jsonl').exists())

    def test_normalization_and_mixed_prompt(self):
        self.assertEqual(typed_icd9('003.1', 'DIAG'), 'DIAG:0031')
        self.assertEqual(typed_icd9('00.31', 'PROC'), 'PROC:0031')
        self.assertEqual(typed_icd9('v10.3', 'DIAG'), 'DIAG:V103')
        self.assertEqual(typed_icd9('E812.0', 'DIAG'), 'DIAG:E8120')
        with self.assertRaises(ValueError): catalog_label_space(['DIAG:0031', '0031'], 'ICD-9')
        with self.assertRaises(ValueError): typed_icd9('V103', 'PROC')
        message = paper_messages('note_to_sid', 'Synthetic note', 'ICD-9', label_space='mixed')[0]['content']
        self.assertIn('diagnoses and procedures', message)


class MixedDatasetTests(unittest.TestCase):
    def test_splits_train_only_catalog_and_unknown_labels(self):
        import pandas as pd
        from semicd.data.dataset import convert_tables
        data = pd.DataFrame([
            {'_id': 'a', 'subject_id': 1, 'text': 'Synthetic A', 'icd9_diag': ['003.1'], 'icd9_proc': ['00.31']},
            {'_id': 'b', 'subject_id': 2, 'text': 'Synthetic B', 'icd9_diag': [], 'icd9_proc': ['01.01']},
            {'_id': 'c', 'subject_id': 3, 'text': 'Synthetic C', 'icd9_diag': ['V10.3'], 'icd9_proc': []}])
        splits = pd.DataFrame({'_id': ['c', 'a', 'b'], 'split': ['test', 'train', 'val']})
        outputs, codes, report = convert_tables(data, splits, 'ICD-9', 'icd9_diag', {'documents': 3, 'codes': 4}, 'icd9_proc')
        self.assertTrue(report['passed'])
        self.assertFalse(report['diagnosis_only'])
        self.assertEqual(codes, ['DIAG:0031', 'PROC:0031'])
        self.assertEqual(outputs['validation'][0]['codes'], ['PROC:0101'])
        self.assertEqual(report['splits']['test']['unknown_gold_assignments'], 1)
        with tempfile.TemporaryDirectory() as tmp:
            import contextlib
            import io
            from semicd.cli import main
            data.to_json(Path(tmp) / 'notes.jsonl', orient='records', lines=True)
            splits.to_json(Path(tmp) / 'splits.jsonl', orient='records', lines=True)
            with patch('semicd.data.dataset.ICD9_FULL_STATISTICS', {'documents': 3, 'codes': 4}), \
                 patch('semicd.data.dataset.ICD9_FULL_SPLITS', {'train': 1, 'validation': 1, 'test': 1}), \
                 patch('semicd.data.dataset.ICD9_FULL_TRAIN_CODES', {'DIAG': 1, 'PROC': 1}), \
                 contextlib.redirect_stdout(io.StringIO()):
                main(['prepare-dataset', '--upstream-file', str(Path(tmp) / 'notes.jsonl'),
                      '--splits-file', str(Path(tmp) / 'splits.jsonl'), '--output-dir', str(Path(tmp) / 'data')])
            exported = json.loads((Path(tmp) / 'data/manifest.json').read_text())
            self.assertEqual(exported['label_space'], 'mixed')
            self.assertTrue(exported['passed'])
            self.assertEqual(json.loads((Path(tmp) / 'data/train_codes.json').read_text()), codes)
            from semicd.training.tasks import validate_training_dataset
            p = Path(tmp) / 'manifest.json'
            report.update(code_system='ICD-9', label_space='mixed')
            p.write_text(json.dumps(report))
            bundle = {'manifest': {'code_system': 'ICD-9', 'label_space': 'mixed'}, 'assignments': {c: [i] for i, c in enumerate(codes)}}
            with patch('semicd.training.tasks.ICD9_FULL_STATISTICS', {'documents': 3, 'codes': 4}), \
                 patch('semicd.training.tasks.ICD9_FULL_SPLITS', {'train': 1, 'validation': 1, 'test': 1}), \
                 patch('semicd.training.tasks.ICD9_FULL_TRAIN_CODES', {'DIAG': 1, 'PROC': 1}):
                self.assertIsNotNone(validate_training_dataset(Namespace(dataset_manifest=str(p)), bundle, outputs['train'], outputs['validation']))
                report['label_space'] = 'diagnosis'; p.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    validate_training_dataset(Namespace(dataset_manifest=str(p)), bundle, outputs['train'], outputs['validation'])
        data.loc[1, 'subject_id'] = 1
        with self.assertRaises(ValueError):
            convert_tables(data, splits, 'ICD-9', 'icd9_diag', {}, 'icd9_proc')


if __name__ == '__main__':
    unittest.main()
