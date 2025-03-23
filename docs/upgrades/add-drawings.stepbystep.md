# Adding New Drawing Types/Subtypes to Ohmni Oracle

Here's your step-by-step guide for adding new drawing types and subtypes to process additional drawing categories:

## Step 1: Identify the New Drawing Type or Subtype

Determine if you need:
- A completely new main drawing type (e.g., "Structural")
- A new subtype of existing drawing type (e.g., "Electrical_OneLineDiagram")

## Step 2: Edit the DRAWING_INSTRUCTIONS Dictionary

Open: `services/ai_service.py`

Add your new entry:

```python
# For new main type:
"NewMainType": "Basic instructions for processing this drawing type.",

# OR for new subtype:
"MainType_NewSubtype": """
Focus on extracting these specific elements:

1. Feature A:
   - Detail 1
   - Detail 2
   
2. Feature B:
   - Spec 1
   - Spec 2
   
3. Feature C:
   - Item 1
   - Item 2
   
Ensure complete coverage of all elements in the drawing.
""",
```

## Step 3: Update the detect_drawing_subtype Function

For a new main type:
```python
# After the existing if/elif blocks
elif drawing_type == "NewMainType":
    if any(term in file_name_lower for term in ["keyword1", "keyword2"]):
        return "NewMainType_Subtype1"
    elif any(term in file_name_lower for term in ["keyword3", "keyword4"]):
        return "NewMainType_Subtype2"
```

For a new subtype of existing type:
```python
# Find the existing block for the main type
if drawing_type == "Electrical":
    # Add your new detection pattern in the appropriate position
    # (Order matters! More specific patterns should come first)
    if any(term in file_name_lower for term in ["oneline", "one-line", "one line"]):
        return "Electrical_OneLineDiagram"
    elif any(term in file_name_lower for term in ["light", "lighting", "fixture"]):
        return "Electrical_Lighting"
    # ... existing code ...
```

## Step 4: Add Model Parameter Optimization (Optional)

If your new drawing type needs specific model settings:
```python
# In the optimize_model_parameters function
if "NewMainType" in drawing_type:
    params["temperature"] = 0.15
    
    if "SpecificSubtype" in drawing_type:
        params["temperature"] = 0.05
```

## Step 5: Write Tests for Your New Types

Open: `tests/test_ai_service.py`

Add test cases:
```python
def test_detect_newtype_subtypes(self):
    """Test new drawing type subtype detection"""
    # Subtype 1
    self.assertEqual(detect_drawing_subtype("NewMainType", "NM-101_KEYWORD1.pdf"), "NewMainType_Subtype1")
    self.assertEqual(detect_drawing_subtype("NewMainType", "KEYWORD2 DRAWING.pdf"), "NewMainType_Subtype1")
    
    # Subtype 2
    self.assertEqual(detect_drawing_subtype("NewMainType", "NM-201_KEYWORD3.pdf"), "NewMainType_Subtype2")
    self.assertEqual(detect_drawing_subtype("NewMainType", "KEYWORD4 PLAN.pdf"), "NewMainType_Subtype2")
```

Then add your new types to the `test_drawing_instructions_has_all_subtypes` expected_subtypes list.

## Step 6: Test Your Changes

Run the tests:
```bash
python3 -m tests.test_ai_service
```

## Complete Example: Adding Structural Drawing Type

### 1. Add Main Type and Subtypes to DRAWING_INSTRUCTIONS
```python
# Main type
"Structural": "Focus on structural elements, dimensions, details, connections and specifications.",

# Subtypes
"Structural_FoundationPlan": """
Focus on foundation elements:

1. Create a 'foundation_elements' array with:
   - Foundation type (footings, piles, etc.)
   - Dimensions and depths
   - Reinforcement details
   - Elevations and levels
   
2. Document soil information:
   - Soil bearing capacity
   - Required treatments
   
3. Capture connection details to superstructure
""",

"Structural_FramingPlan": """
Focus on structural framing:

1. Document all structural members:
   - Beams and girders (sizes, materials)
   - Columns (sizes, spacing)
   - Bracing elements
   - Connection details
   
2. Record load information:
   - Dead loads
   - Live loads
   - Special loading conditions
   
3. Note any special framing conditions
""",
```

### 2. Update detect_drawing_subtype Function
```python
# After the existing architectural block
elif drawing_type == "Structural":
    if any(term in file_name_lower for term in ["foundation", "footing", "pile"]):
        return "Structural_FoundationPlan"
    elif any(term in file_name_lower for term in ["framing", "beam", "column", "steel"]):
        return "Structural_FramingPlan"
```

### 3. Update optimize_model_parameters (Optional)
```python
elif "Structural" in drawing_type:
    # Set precision for structural drawings
    params["temperature"] = 0.15
```

### 4. Add Tests
```python
def test_detect_structural_subtypes(self):
    """Test structural drawing subtype detection"""
    # Foundation Plans
    self.assertEqual(detect_drawing_subtype("Structural", "S-101_FOUNDATION_PLAN.pdf"), "Structural_FoundationPlan")
    self.assertEqual(detect_drawing_subtype("Structural", "FOOTING DETAILS.pdf"), "Structural_FoundationPlan")
    
    # Framing Plans
    self.assertEqual(detect_drawing_subtype("Structural", "S-201_FRAMING_PLAN.pdf"), "Structural_FramingPlan")
    self.assertEqual(detect_drawing_subtype("Structural", "STEEL COLUMN LAYOUT.pdf"), "Structural_FramingPlan")
    
    # Generic structural (no specific subtype)
    self.assertEqual(detect_drawing_subtype("Structural", "S-001_GENERAL_NOTES.pdf"), "Structural")
```

That's it! Just follow these steps for each new drawing type and subtype you need to add.
