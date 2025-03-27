"""
Base prompt templates to reduce duplication across specific drawing types.
"""

BASE_DRAWING_TEMPLATE = """
You are extracting information from a {drawing_type} drawing.
Document ALL elements following this general structure, adapting to project-specific terminology.

EXTRACTION PRIORITIES:
1. Identify and extract ALL {element_type} elements
2. Document specifications EXACTLY as shown for each element
3. Preserve ALL notes, reference numbers, and special requirements

{specific_instructions}

EXAMPLE STRUCTURE (adapt based on what you find in the drawing):
{example_structure}

CRITICAL INSTRUCTIONS:
1. CAPTURE everything in the drawing
2. PRESERVE original terminology and organization
3. GROUP similar elements together in logical sections
4. DOCUMENT all specifications and detailed information
5. Ensure your entire response is a single, valid JSON object.

{industry_context}
"""

SCHEDULE_TEMPLATE = """
You are extracting {schedule_type} information from {drawing_category} drawings. 
Document ALL {item_type} following the structure in this example, while adapting to project-specific terminology.

EXTRACTION PRIORITIES:
1. Capture EVERY {item_type} with ALL specifications
2. Document ALL {key_properties} EXACTLY as shown
3. Include ALL notes, requirements, and special conditions

EXAMPLE OUTPUT STRUCTURE (field names may vary by project):
{example_structure}

CRITICAL INSTRUCTIONS:
1. EXTRACT all {item_type}s shown on the {source_location}
2. PRESERVE exact {preservation_focus}
3. INCLUDE all technical specifications and requirements 
4. ADAPT the structure to match this specific drawing
5. MAINTAIN the overall hierarchical organization shown in the example
6. Format your output as a complete, valid JSON object.

{stake_holders} rely on this information for {use_case}.
Complete accuracy is essential for {critical_purpose}.
"""

def create_general_template(drawing_type, element_type, instructions, example, context):
    """Create a general prompt template with the provided parameters."""
    return BASE_DRAWING_TEMPLATE.format(
        drawing_type=drawing_type,
        element_type=element_type,
        specific_instructions=instructions,
        example_structure=example,
        industry_context=context
    )

def create_schedule_template(
    schedule_type, 
    drawing_category,
    item_type,
    key_properties,
    example_structure,
    source_location,
    preservation_focus,
    stake_holders,
    use_case,
    critical_purpose
):
    """Create a schedule template with the provided parameters."""
    return SCHEDULE_TEMPLATE.format(
        schedule_type=schedule_type,
        drawing_category=drawing_category,
        item_type=item_type,
        key_properties=key_properties,
        example_structure=example_structure,
        source_location=source_location,
        preservation_focus=preservation_focus,
        stake_holders=stake_holders,
        use_case=use_case,
        critical_purpose=critical_purpose
    )
