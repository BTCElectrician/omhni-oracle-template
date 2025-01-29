import unittest
from utils.pdf_processor import extract_text_and_tables_from_pdf
from utils.pdf_utils import structure_panel_data

class TestPDFProcessing(unittest.IsolatedAsyncioTestCase):
    async def test_panel_schedule_extraction(self):
        test_file = "samples/panel_schedule.pdf"
        content = await extract_text_and_tables_from_pdf(test_file)
        
        self.assertIn("Main Panel", content)
        self.assertRegex(content, r"Circuit\s+\d+")
        
        structured = await structure_panel_data(client, content)
        self.assertIn("circuits", structured)
        self.assertTrue(len(structured["circuits"]) > 5) 