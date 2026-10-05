"""Artificial official-format fixtures; no licensed or clinical data."""
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from semicd.data.ontology import FILES, prepare_ontology
from semicd.data.umls import prepare_umls


def release(root, edition, records, body):
    root.mkdir()
    order, xml = FILES[edition]
    (root / order).write_text(''.join(
        f'{i:05d} {code:<8}{flag} {desc:<61}{desc}\n'
        for i, (code, flag, desc) in enumerate(records, 1)))
    (root / xml).write_text('<ICD10CM.tabular><chapter><desc>Chapter</desc>'
        '<section><desc>Block</desc>' + body + '</section></chapter></ICD10CM.tabular>')


def diag(code):
    return f'<diag><name>{code}</name><desc>Category {code}</desc></diag>'


def atom(cui, code, term, sab='ICD10CM', language='ENG', suppress='N'):
    fields = [cui, language, 'P', 'L1', 'PF', 'S1', 'Y', 'A1', '', '', '',
              sab, 'PT', code, term, '0', suppress, '']
    return '|'.join(fields) + '|\n'


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.codes = self.root / 'codes.json'
        self.codes.write_text(json.dumps(['A001', 'B001', 'S000XXA']))
        self.primary = self.root / '2020'
        self.supplement = self.root / '2015'
        release(self.primary, 'FY2020', [('A001', 1, 'Alpha'), ('S000XXA', 1, 'Injury')],
            diag('A00.1') + '<diag><name>S00</name><desc>Injury category</desc>'
            '<sevenChrDef><extension char="A">Initial</extension></sevenChrDef>'
            '<diag><name>S00.0</name><desc>Injury leaf</desc></diag></diag>')
        release(self.supplement, 'FY2015', [('A001', 1, 'Old alpha'), ('B001', 1, 'Beta')],
                diag('A00.1') + diag('B00.1'))
        self.rrf = self.root / 'MRCONSO.RRF'
        self.rrf.write_text(atom('C1', 'A00.1', 'Alpha') + atom('C2', 'A00.1', 'Other alpha')
            + atom('C1', 'unused', 'Cross vocabulary', 'SYNTHETIC', suppress='Y')
            + atom('C1', 'unused', 'Cross vocabulary', 'SYNTHETIC')
            + atom('C1', 'unused', 'Non English', 'SYNTHETIC', 'SPA'))

    def ontology(self, output='ontology'):
        return prepare_ontology(Namespace(codes_file=self.codes, primary_dir=self.primary,
            supplement_dir=self.supplement, output_dir=self.root / output))

    def test_cli_chain_and_provenance(self):
        commands = [
            ['prepare-ontology', '--codes-file', self.codes, '--primary-dir', self.primary,
             '--supplement-dir', self.supplement, '--output-dir', self.root / 'ontology'],
            ['prepare-umls', '--codes-file', self.codes, '--umls-file', self.rrf,
             '--release', 'SYNTHETIC', '--output-dir', self.root / 'umls'],
            ['prepare-profiles', '--codes-file', self.codes, '--ontology-file', self.root / 'ontology/ontology.jsonl',
             '--umls-terms-file', self.root / 'umls/umls_terms.json', '--output-dir', self.root / 'profiles']]
        for args in commands:
            subprocess.run([sys.executable, '-m', 'semicd', *map(str, args)], check=True, capture_output=True)
        rows = [json.loads(line) for line in (self.root / 'ontology/ontology.jsonl').read_text().splitlines()]
        self.assertEqual([row['edition'] for row in rows], ['FY2020', 'FY2015', 'FY2020'])
        self.assertEqual(rows[0]['description'], 'Alpha')
        self.assertEqual(rows[2]['category'], 'Injury category')
        terms = json.loads((self.root / 'umls/umls_terms.json').read_text())
        self.assertEqual(terms['A001'], ['Alpha', 'Other alpha', 'Cross vocabulary', 'Cross vocabulary'])
        self.assertEqual(terms['B001'], [])
        manifest = json.loads((self.root / 'umls/manifest.json').read_text())
        self.assertEqual(manifest['cuis_by_code']['A001'], ['C1', 'C2'])
        profiles = [json.loads(line) for line in (self.root / 'profiles/profiles.jsonl').read_text().splitlines()]
        self.assertEqual(profiles[1]['fields']['UMLS Terms'], 'Beta')
        self.assertEqual(len(profiles), 3)

    def test_ontology_failures_do_not_export_success(self):
        order_name, xml_name = FILES['FY2020']
        order_path, xml_path = self.primary / order_name, self.primary / xml_name
        original_order, original_xml = order_path.read_text(), xml_path.read_text()
        cases = [
            ('duplicate_order', original_order + original_order.splitlines()[0] + '\n', original_xml),
            ('nonbillable', original_order.replace('A001    1', 'A001    0'), original_xml),
            ('missing_path', original_order, original_xml.replace(diag('A00.1'), '')),
            ('duplicate_or_multi_parent', original_order, original_xml.replace(diag('A00.1'), diag('A00.1') * 2)),
            ('unmatched', original_order, original_xml)]
        for reason, order, xml in cases:
            with self.subTest(reason=reason):
                order_path.write_text(order)
                xml_path.write_text(xml)
                self.codes.write_text(json.dumps(['Z999'] if reason == 'unmatched' else ['A001']))
                with self.assertRaises(ValueError):
                    self.ontology(reason)
                output = self.root / reason
                self.assertFalse((output / 'ontology.jsonl').exists())
                self.assertEqual(json.loads((output / 'issues.json').read_text())[0]['reason'], reason)
                self.assertFalse(json.loads((output / 'manifest.json').read_text())['passed'])

    def test_catalog_and_rrf_reject_invalid_inputs(self):
        self.codes.write_text(json.dumps(['A00.1']))
        with self.assertRaises(ValueError):
            self.ontology()
        self.codes.write_text(json.dumps(['A001']))
        self.rrf.write_text('bad|row\n')
        with self.assertRaises(ValueError):
            prepare_umls(Namespace(codes_file=self.codes, umls_file=self.rrf,
                release='SYNTHETIC', output_dir=self.root / 'umls'))


if __name__ == '__main__':
    unittest.main()
