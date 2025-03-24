import unittest
from services.ai_service import detect_drawing_subtype, DRAWING_INSTRUCTIONS, optimize_model_parameters, ModelType
import os
from unittest.mock import patch

class TestDrawingSubtypeDetection(unittest.TestCase):
    
    def test_detect_electrical_subtypes(self):
        """Test electrical drawing subtype detection"""
        # Panel schedules
        self.assertEqual(detect_drawing_subtype("Electrical", "E-601_PANEL_SCHEDULES.pdf"), "Electrical_PanelSchedule")
        self.assertEqual(detect_drawing_subtype("Electrical", "PANEL-SCHEDULES.pdf"), "Electrical_PanelSchedule")
        self.assertEqual(detect_drawing_subtype("Electrical", "E4.1 panelboard.pdf"), "Electrical_PanelSchedule")
        
        # Lighting
        self.assertEqual(detect_drawing_subtype("Electrical", "E-201_LIGHTING.pdf"), "Electrical_Lighting")
        self.assertEqual(detect_drawing_subtype("Electrical", "LIGHTING FIXTURE SCHEDULE.pdf"), "Electrical_PanelSchedule")
        
        # Power
        self.assertEqual(detect_drawing_subtype("Electrical", "E-301_POWER.pdf"), "Electrical_Power")
        self.assertEqual(detect_drawing_subtype("Electrical", "POWER PLAN - FIRST FLOOR.pdf"), "Electrical_Power")
        self.assertEqual(detect_drawing_subtype("Electrical", "RECEPTACLE LAYOUT.pdf"), "Electrical_Power")
        
        # Fire Alarm
        self.assertEqual(detect_drawing_subtype("Electrical", "E-701_FIRE_ALARM.pdf"), "Electrical_FireAlarm")
        self.assertEqual(detect_drawing_subtype("Electrical", "FA FLOOR PLAN.pdf"), "Electrical_FireAlarm")
        
        # Technology
        self.assertEqual(detect_drawing_subtype("Electrical", "E-501_TECHNOLOGY.pdf"), "Electrical_Technology")
        self.assertEqual(detect_drawing_subtype("Electrical", "LOW VOLTAGE PLAN.pdf"), "Electrical_Technology")
        self.assertEqual(detect_drawing_subtype("Electrical", "DATA COMM LAYOUT.pdf"), "Electrical_Technology")
        
        # Generic electrical (no specific subtype)
        self.assertEqual(detect_drawing_subtype("Electrical", "E-001_SYMBOLS_AND_ABBREVIATIONS.pdf"), "Electrical")
    
    def test_detect_architectural_subtypes(self):
        """Test architectural drawing subtype detection"""
        # Floor Plans
        self.assertEqual(detect_drawing_subtype("Architectural", "A-101_FLOOR_PLAN.pdf"), "Architectural_FloorPlan")
        self.assertEqual(detect_drawing_subtype("Architectural", "FLOOR LAYOUT.pdf"), "Architectural_FloorPlan")
        
        # Reflected Ceiling Plans
        self.assertEqual(detect_drawing_subtype("Architectural", "A-201_REFLECTED_CEILING_PLAN.pdf"), "Architectural_ReflectedCeiling")
        self.assertEqual(detect_drawing_subtype("Architectural", "RCP FIRST FLOOR.pdf"), "Architectural_ReflectedCeiling")
        
        # Partition Types
        self.assertEqual(detect_drawing_subtype("Architectural", "A-501_PARTITION_TYPES.pdf"), "Architectural_Partition")
        self.assertEqual(detect_drawing_subtype("Architectural", "WALL-TYPE DETAILS.pdf"), "Architectural_Partition")
        
        # Generic architectural (no specific subtype)
        self.assertEqual(detect_drawing_subtype("Architectural", "A-001_SYMBOLS_AND_ABBREVIATIONS.pdf"), "Architectural")
    
    def test_main_drawing_type_fallback(self):
        """Test that non-matched drawing types fall back to the main type"""
        self.assertEqual(detect_drawing_subtype("Mechanical", "M-101_HVAC_PLAN.pdf"), "Mechanical")
        self.assertEqual(detect_drawing_subtype("Plumbing", "P-201_WATER_PIPING.pdf"), "Plumbing")
    
    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        self.assertEqual(detect_drawing_subtype("", ""), "")
        self.assertEqual(detect_drawing_subtype("Electrical", ""), "Electrical")
        self.assertEqual(detect_drawing_subtype("", "PANEL_SCHEDULES.pdf"), "")
    
    def test_drawing_instructions_has_all_subtypes(self):
        """Verify that all subtypes mentioned in the detect_drawing_subtype function
        have corresponding entries in the DRAWING_INSTRUCTIONS dictionary"""
        expected_subtypes = [
            "Electrical_PanelSchedule",
            "Electrical_Lighting",
            "Electrical_Power",
            "Electrical_FireAlarm",
            "Electrical_Technology",
            "Architectural_FloorPlan",
            "Architectural_ReflectedCeiling",
            "Architectural_Partition"
        ]
        
        for subtype in expected_subtypes:
            self.assertIn(subtype, DRAWING_INSTRUCTIONS, f"Missing instructions for {subtype}")


class TestModelSelection(unittest.TestCase):
    """Test the model selection functionality"""
    
    @patch('config.settings.get_force_mini_model')
    def test_force_mini_model_enabled(self, mock_get_force_mini_model):
        """Test that when FORCE_MINI_MODEL is true, it always returns GPT-4o-mini"""
        # Mock the function to return True
        mock_get_force_mini_model.return_value = True
        
        # Test with various inputs that would normally trigger larger model
        specs_drawing = "Specifications"
        large_content = "A" * 100000  # 100k characters
        specs_file = "SPECIFICATION_DOCUMENT.pdf"
        
        # Test optimize_model_parameters with these inputs
        params = optimize_model_parameters(specs_drawing, large_content, specs_file)
        
        # Verify mini model is selected regardless of inputs
        self.assertEqual(params["model_type"], ModelType.GPT_4O_MINI)
    
    @patch('config.settings.get_force_mini_model')
    def test_normal_model_selection(self, mock_get_force_mini_model):
        """Test that normal model selection logic works when FORCE_MINI_MODEL is false"""
        # Mock the function to return False
        mock_get_force_mini_model.return_value = False
        
        # Test with inputs that would normally trigger larger model
        specs_drawing = "Specifications"
        large_content = "A" * 100000  # 100k characters
        specs_file = "SPECIFICATION_DOCUMENT.pdf"
        
        # Test optimize_model_parameters with these inputs
        params = optimize_model_parameters(specs_drawing, large_content, specs_file)
        
        # Verify larger model is selected based on input criteria
        self.assertEqual(params["model_type"], ModelType.GPT_4O)


if __name__ == "__main__":
    unittest.main() 