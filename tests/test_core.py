"""Tests unitaires (python -m unittest discover tests  ou  pytest)."""
import random
import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from receiptfraud.data import Document, Line  # noqa: E402
from receiptfraud.extract import Extraction, RuleExtractor  # noqa: E402
from receiptfraud.fraud import parse_financials, run_checks  # noqa: E402
from receiptfraud.fraud_synth import date_future, date_impossible, total_inflated  # noqa: E402
from receiptfraud.labeling import line_labels  # noqa: E402
from receiptfraud.metrics import field_scores, is_correct  # noqa: E402
from receiptfraud.textutils import clean_company, find_amounts, find_dates  # noqa: E402


def receipt(total="12.60", cash="20.00", change="7.40", date_txt="25/12/2018 8:13 PM"):
    rows = [
        ("KEDAI MAJU SDN BHD (123456-X)", 60, 20, 300, 40),
        ("NO 5, JALAN MERAH, 81100 JOHOR BAHRU", 40, 50, 330, 70),
        ("DATE:", 20, 100, 70, 118), (date_txt, 120, 100, 300, 118),
        ("NASI LEMAK", 20, 150, 150, 168), ("8.00", 300, 150, 340, 168),
        ("TEH TARIK", 20, 175, 150, 193), ("4.00", 300, 175, 340, 193),
        ("SUBTOTAL", 20, 210, 150, 228), ("12.00", 300, 210, 340, 228),
        ("GST 6%", 20, 235, 150, 253), ("0.60", 300, 235, 340, 253),
        ("TOTAL", 20, 260, 150, 278), (total, 300, 260, 340, 278),
        ("CASH", 20, 285, 150, 303), (cash, 300, 285, 340, 303),
        ("CHANGE", 20, 310, 150, 328), (change, 300, 310, 340, 328),
    ]
    lines = [Line(t, x0, y0, x1, y1) for t, x0, y0, x1, y1 in rows]
    labels = {"company": "KEDAI MAJU SDN BHD", "date": "25/12/2018",
              "address": "NO 5, JALAN MERAH, 81100 JOHOR BAHRU", "total": "12.60"}
    return Document("t", lines, 360, 350, labels)


def gold_extraction(doc, total="12.60", date_s="25/12/2018"):
    return Extraction(values={"company": "KEDAI MAJU SDN BHD", "date": date_s, "address": "x", "total": total},
                      confidence={"company": 1, "date": 1, "total": 1, "address": 1}, lines={"total": [13]})


class TextUtils(unittest.TestCase):
    def test_amounts(self):
        self.assertEqual([v for _, v in find_amounts("TOTAL RM 1,234.50")], [1234.5])
        self.assertEqual(find_amounts("GST @6.00%"), [])
        self.assertEqual([v for _, v in find_amounts("-0.02")], [-0.02])

    def test_dates(self):
        self.assertEqual(find_dates("25/12/2018 8:13")[0][1], (2018, 12, 25))
        self.assertEqual(find_dates("2018-03-04")[0][1], (2018, 3, 4))
        self.assertEqual(find_dates("05 JAN 2019")[0][1], (2019, 1, 5))
        self.assertIsNone(find_dates("31/02/2018")[0][1])  # date impossible détectée

    def test_clean_company(self):
        self.assertEqual(clean_company("99 SPEED MART S/B (519537-X)"), "99 SPEED MART S/B")


class LabelingAndMetrics(unittest.TestCase):
    def test_line_labels(self):
        lab = line_labels(receipt())
        self.assertEqual(lab["company"][0], 1)
        self.assertEqual(lab["date"][3], 1)
        self.assertEqual(lab["total"][13], 1)

    def test_metrics(self):
        self.assertTrue(is_correct("total", "9.00", "RM9.00"))
        self.assertTrue(is_correct("date", "25-12-2018", "25/12/2018"))
        s = field_scores([{"company": "A", "date": "", "address": "B", "total": "1.00"}],
                         [{"company": "A", "date": "01/01/2019", "address": "C", "total": "1.00"}])
        self.assertEqual(s["company"]["f1"], 1.0)
        self.assertEqual(s["date"]["recall"], 0.0)
        self.assertEqual(s["address"]["f1"], 0.0)

    def test_rule_extractor(self):
        ex = RuleExtractor().extract(receipt())
        self.assertEqual(ex.values["total"], "12.60")
        self.assertEqual(ex.values["date"], "25/12/2018")


class FraudChecks(unittest.TestCase):
    today = date(2026, 10, 25)

    def test_clean_receipt_passes(self):
        d = receipt()
        chk = run_checks(d, gold_extraction(d), self.today)
        fin = chk.financials
        self.assertEqual((fin.subtotal, fin.tax, fin.cash, fin.change), (12.0, 0.6, 20.0, 7.4))
        self.assertEqual(fin.items, [8.0, 4.0])
        hard = [r for r, h in chk.hits.items() if h]
        self.assertEqual(hard, [])

    def test_inflated_total_detected(self):
        d = receipt(total="45.60")
        chk = run_checks(d, gold_extraction(d, total="45.60"), self.today)
        self.assertTrue(chk.hits["subtotal_tax_mismatch"])
        self.assertTrue(chk.hits["cash_change_mismatch"])

    def test_dates_detected(self):
        d = receipt(date_txt="31/02/2018")
        self.assertTrue(run_checks(d, gold_extraction(d, date_s="31/02/2018"), self.today).hits["date_invalid"])
        d = receipt(date_txt="25/12/2031")
        self.assertTrue(run_checks(d, gold_extraction(d, date_s="25/12/2031"), self.today).hits["date_future"])

    def test_generators(self):
        rng = random.Random(0)
        d = receipt()
        f = total_inflated(d, rng)
        self.assertNotEqual(f.lines[13].text, "12.60")
        self.assertEqual(f.lines[9].text, "12.00")  # le sous-total n'est pas touché
        self.assertIsNone(find_dates(date_impossible(d, rng).lines[3].text)[0][1])
        self.assertGreater(find_dates(date_future(d, rng).lines[3].text)[0][1][0], 2027)


class ParseFinancialsMisc(unittest.TestCase):
    def test_negative_cash(self):
        d = receipt(cash="-20.00")
        fin = parse_financials(d, gold_extraction(d))
        self.assertEqual(fin.cash, 20.0)


if __name__ == "__main__":
    unittest.main()
