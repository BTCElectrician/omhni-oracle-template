# AI-First Optimization Strategies for Construction Drawing Processing

## 1. Enhanced Prompting Strategies

The foundation of AI-first optimization is crafting more effective prompts to get better results with fewer iterations.

### 1.1 Document-Type Specific Prompts

```python
def get_enhanced_system_message(drawing_type: str, file_name: str) -> str:
    """Generate specialized system message based on document type and file name."""
    
    # Base message with clear structure instructions
    base_message = f"""
    You are an expert construction document interpreter specializing in {drawing_type} drawings.
    Extract ONLY the essential information from this {drawing_type.lower()} drawing and organize it into a structured JSON.
    
    FOLLOW THESE STEPS EXACTLY:
    1. First, identify the document type and key elements
    2. Then extract data into these categories:
       - 'metadata': Basic document info (title, number, date)
       - 'specifications': Key technical details
       - 'schedules': Any tabular data
       - 'notes': Important annotations
    3. Format ALL output as a valid JSON object
    """
    
    # Architectural drawings
    if drawing_type == "Architectural":
        base_message += """
        FOR ARCHITECTURAL DRAWINGS:
        - Include a 'rooms' array with objects for each identifiable room
        - For each room, include: 'number', 'name', 'dimensions', 'finishes'
        - Extract wall types and door schedules if present
        - Identify ceiling heights and types
        """
    
    # Electrical drawings - enhanced panel schedule handling
    elif drawing_type == "Electrical":
        is_panel = "PANEL" in file_name.upper() or "SCHEDULE" in file_name.upper()
        if is_panel:
            base_message += """
            FOR PANEL SCHEDULES:
            - Format electrical panel data consistently 
            - Use 'circuit' for circuit numbers
            - Use 'load_name' for load descriptions
            - Use 'trip' for breaker sizes/trips
            - Preserve ALL circuit entries exactly as shown
            - Maintain correct phase relationships
            """
        else:
            base_message += """
            FOR ELECTRICAL DRAWINGS:
            - Capture fixture types and counts
            - Note special electrical requirements
            - Include device locations when specified
            """
    
    # Mechanical drawings - specialized schedule handling
    elif drawing_type == "Mechanical":
        if "SCHEDULE" in file_name.upper():
            base_message += """
            FOR MECHANICAL SCHEDULES:
            - Preserve all equipment data including model numbers
            - Maintain CFM values and other capacity metrics
            - Keep maintenance requirements and notes
            - Record electrical requirements for each unit
            """
        else:
            base_message += """
            FOR MECHANICAL DRAWINGS:
            - Note duct sizes and flow directions
            - Capture equipment connections and relationships
            - Identify control systems and sensors
            """
    
    # Plumbing drawings - emphasize fixture and pipe details
    elif drawing_type == "Plumbing":
        base_message += """
        FOR PLUMBING DRAWINGS:
        - Record fixture types, quantities, and connections
        - Note pipe sizes and materials
        - Capture flow directions and system types
        - Include fixture schedules with manufacturer info
        """
    
    # Specifications need special handling for their structure
    if "SPECIFICATION" in file_name.upper():
        base_message += """
        CRITICAL FOR SPECIFICATIONS:
        - Preserve the EXACT hierarchical structure of specifications
        - Maintain section numbers with their full content
        - Group related specifications under the same parent section
        """
    
    # Clear formatting instructions
    base_message += """
    FORMATTING REQUIREMENTS:
    - Ensure all JSON is valid and properly nested
    - Use consistent field names
    - Maintain arrays for collections of similar items
    - DO NOT truncate or summarize content unless explicitly stated
    """
    
    return base_message
```

### 1.2 Explicit Field Mapping Instructions

```python
def get_field_mapping_instructions(drawing_type: str) -> str:
    """Generate specific field mapping instructions based on drawing type."""
    
    if drawing_type == "Electrical":
        return """
        MAP THESE FIELDS CONSISTENTLY:
        - "Circuit", "CKT", "CKT #" → "circuit"
        - "Description", "Load", "Device" → "load_name"
        - "Poles", "P" → "poles"
        - "Amperes", "A", "Amps", "Trip" → "trip"
        - "Phase", "PH", "φ" → "phase"
        - "Wire Size", "AWG", "Conductor" → "wire_size"
        - "Mounting", "MTG", "Mount" → "mounting" 
        - "Voltage", "V", "Volts" → "voltage"
        """
    elif drawing_type == "Mechanical":
        return """
        MAP THESE FIELDS CONSISTENTLY:
        - "Equipment", "Unit", "Device" → "equipment_id"
        - "Manufacturer", "MFR", "Make" → "manufacturer"
        - "Model", "Model #", "Type" → "model"
        - "CFM", "Airflow" → "cfm"
        - "Tons", "Cooling Capacity" → "cooling_capacity"
        - "MBH", "Heating Capacity" → "heating_capacity"
        - "Power", "Electrical", "Voltage" → "power_requirements"
        """
    
    return ""
```

## 2. Intelligent Content Pre-processing

Pre-processing content before sending to AI can dramatically improve processing time and accuracy.

### 2.1 Structure Enhancement

```python
def preprocess_for_ai(content: str, drawing_type: str, file_name: str) -> str:
    """
    Format content before sending to AI to improve processing.
    Focus on highlighting structure rather than filtering.
    """
    processed_content = content
    
    # Add structural markers to help AI understand document organization
    if "TABLE:" in processed_content:
        # Enhance table visibility with clear markers
        processed_content = processed_content.replace("TABLE:", "\n\n--- TABLE BEGIN ---\n")
        processed_content = processed_content.replace("\nTABLE:", "\n\n--- TABLE BEGIN ---\n")
        
        # Find table sections and add end markers
        table_sections = processed_content.split("--- TABLE BEGIN ---")
        if len(table_sections) > 1:
            new_content = table_sections[0]
            for i in range(1, len(table_sections)):
                table_text = table_sections[i]
                # Find where table likely ends (2+ consecutive newlines)
                table_parts = re.split(r'\n{2,}', table_text, maxsplit=1)
                if len(table_parts) > 1:
                    new_content += "--- TABLE BEGIN ---\n" + table_parts[0] + "\n--- TABLE END ---\n\n" + table_parts[1]
                else:
                    new_content += "--- TABLE BEGIN ---\n" + table_text + "\n--- TABLE END ---\n"
            processed_content = new_content
    
    # Drawing-type specific preprocessing
    if drawing_type == "Architectural":
        # Highlight room information
        room_pattern = r'(?i)(room|space)\s*(\d+[A-Z]?)'
        processed_content = re.sub(room_pattern, r'[ROOM: \1 \2]', processed_content)
        
    elif drawing_type == "Electrical" and "PANEL" in file_name.upper():
        # Highlight circuit information
        processed_content = processed_content.replace("Circuit", "[CIRCUIT]")
        processed_content = processed_content.replace("CKT", "[CIRCUIT]")
        
    return processed_content
```

### 2.2 Content Filtering

```python
def filter_irrelevant_content(content: str, drawing_type: str) -> str:
    """Remove irrelevant content to reduce token usage."""
    
    # Common patterns to remove
    patterns_to_remove = [
        r"GENERAL NOTES:?\s*1\.\s*ALL WORK SHALL CONFORM TO.*?(?=\n\n)",  # Boilerplate notes
        r"(?i)REVISIONS?:?\s*.*?(?=\n\n)",  # Revision history
        r"(?i)DRAWN BY:.*?(?=\n\n)",  # Drawing attribution
        r"(?i)COPYRIGHT.*?RESERVED",  # Copyright notices
    ]
    
    filtered_content = content
    
    # Apply general filters
    for pattern in patterns_to_remove:
        filtered_content = re.sub(pattern, "", filtered_content, flags=re.DOTALL)
    
    # Drawing-specific filtering
    if drawing_type == "Electrical" and "PANEL" in content.upper():
        # Keep only panel schedule relevant sections
        sections = filtered_content.split("\nPAGE")
        relevant_sections = []
        
        for section in sections:
            if any(term in section.upper() for term in ["PANEL", "CIRCUIT", "LOAD", "BREAKER"]):
                relevant_sections.append(section)
        
        if relevant_sections:
            filtered_content = "\nPAGE".join(relevant_sections)
    
    # Ensure we're not removing too much content
    if len(filtered_content) < len(content) * 0.7:
        logging.warning(f"Content filtering removed too much content ({len(content) - len(filtered_content)} chars)")
        return content  # Revert to original if we've removed too much
        
    return filtered_content
```

## 3. Model Parameter Tuning

Optimizing model parameters based on document analysis can significantly improve processing efficiency.

### 3.1 Document Complexity Analysis

```python
def analyze_document_complexity(content: str, drawing_type: str, file_name: str) -> Dict[str, Any]:
    """
    Analyze document to determine optimal AI processing parameters.
    Returns temperature and token settings based on content analysis.
    """
    # Calculate content metrics
    char_count = len(content)
    table_count = content.count("TABLE:") 
    line_count = content.count("\n")
    
    # Calculate approximate token count (rough estimate)
    estimated_tokens = char_count / 4
    
    # Analyze content complexity factors
    has_tables = table_count > 0
    is_specification = "SPECIFICATION" in file_name.upper()
    is_panel_schedule = "PANEL" in file_name.upper() and "SCHEDULE" in file_name.upper()
    
    # Calculate complexity score (0-10)
    complexity = 3.0  # Base complexity
    
    # Adjust based on document size
    if char_count > 30000:
        complexity += 3.0
    elif char_count > 15000:
        complexity += 1.5
        
    # Adjust based on tables
    if table_count > 5:
        complexity += 2.0
    elif table_count > 0:
        complexity += 1.0
        
    # Adjust based on document type
    if is_specification:
        complexity += 2.0
    if is_panel_schedule:
        complexity -= 1.0  # Panel schedules are more structured, thus simpler
        
    # Document-specific optimizations
    if is_panel_schedule:
        # Panel schedules need precise, deterministic output
        return {
            "temperature": 0.05,  # Very low temperature for consistent formatting
            "max_tokens": 4000,   # Usually sufficient for panel schedules
            "model": "gpt-4o-mini"
        }
    elif is_specification:
        # Specifications need comprehensive content preservation
        return {
            "temperature": 0.2,   # Low but not too restrictive
            "max_tokens": 12000,  # Higher for complete specification content
            "model": "gpt-4o"
        }
    elif drawing_type == "Architectural":
        # Architectural drawings often need inference and relationship mapping
        if complexity > 7:
            return {
                "temperature": 0.2,   # Some creativity for inferring relationships
                "max_tokens": 8000,   # Moderate token limit
                "model": "gpt-4o"
            }
        else:
            return {
                "temperature": 0.2,
                "max_tokens": 8000,
                "model": "gpt-4o-mini"
            }
    
    # Default settings for other drawing types
    if complexity > 7:
        return {
            "temperature": 0.2,
            "max_tokens": 8000,
            "model": "gpt-4o"
        }
    else:
        return {
            "temperature": 0.1,
            "max_tokens": 8000,
            "model": "gpt-4o-mini"
        }
```

### 3.2 Token Budget Management

```python
def apply_token_budget(content: str, drawing_type: str, max_input_tokens: int = 8000) -> str:
    """
    Apply intelligent token budget to ensure we stay within limits.
    Prioritize important content based on drawing type.
    """
    # Estimate current tokens (rough approximation)
    estimated_tokens = len(content) / 4
    
    # If we're within budget, return the original content
    if estimated_tokens <= max_input_tokens:
        return content
    
    # We need to reduce content - let's be smart about it
    reduction_ratio = max_input_tokens / estimated_tokens
    
    # Drawing-type specific reduction strategies
    if drawing_type == "Electrical" and "PANEL" in content.upper():
        # For panel schedules, prioritize circuit data over notes
        sections = content.split("\n\n")
        circuit_sections = []
        other_sections = []
        
        for section in sections:
            if any(term in section.upper() for term in ["CIRCUIT", "LOAD", "BREAKER", "PANEL"]):
                circuit_sections.append(section)
            else:
                other_sections.append(section)
        
        # Keep all circuit data, then add other sections until we hit the limit
        budget_content = "\n\n".join(circuit_sections)
        estimated_tokens = len(budget_content) / 4
        
        for section in other_sections:
            section_tokens = len(section) / 4
            if estimated_tokens + section_tokens <= max_input_tokens:
                budget_content += "\n\n" + section
                estimated_tokens += section_tokens
            else:
                break
                
        return budget_content
    
    elif drawing_type == "Specifications":
        # For specifications, try to keep complete sections
        sections = re.split(r"(SECTION \d+\.\d+)", content)
        budget_content = ""
        estimated_tokens = 0
        
        # Always include the first section (index 0 is before first section marker)
        if sections:
            budget_content = sections[0]
            estimated_tokens = len(sections[0]) / 4
        
        # Add complete sections until we approach the limit
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                section_header = sections[i]
                section_content = sections[i+1]
                section_tokens = (len(section_header) + len(section_content)) / 4
                
                if estimated_tokens + section_tokens <= max_input_tokens:
                    budget_content += section_header + section_content
                    estimated_tokens += section_tokens
                else:
                    # At least include the section header
                    budget_content += section_header + " [Content truncated due to token limits]"
                    break
        
        return budget_content
    
    # Default strategy - proportional reduction from each paragraph
    paragraphs = content.split("\n\n")
    selected_paragraphs = []
    current_tokens = 0
    
    # Always include the first two paragraphs (likely to contain important metadata)
    for i in range(min(2, len(paragraphs))):
        selected_paragraphs.append(paragraphs[i])
        current_tokens += len(paragraphs[i]) / 4
    
    # For the rest, sample based on importance heuristic and remaining budget
    remaining_tokens = max_input_tokens - current_tokens
    remaining_paragraphs = paragraphs[2:]
    
    # Sort paragraphs by importance (length is a simple heuristic)
    remaining_paragraphs.sort(key=len, reverse=True)
    
    # Take paragraphs until we approach the limit
    for paragraph in remaining_paragraphs:
        paragraph_tokens = len(paragraph) / 4
        if current_tokens + paragraph_tokens <= max_input_tokens:
            selected_paragraphs.append(paragraph)
            current_tokens += paragraph_tokens
        
    # Re-assemble in original order
    paragraph_indices = {p: i for i, p in enumerate(paragraphs)}
    selected_paragraphs.sort(key=lambda p: paragraph_indices.get(p, 999))
    
    return "\n\n".join(selected_paragraphs)
```

## 4. Multi-Pass Processing for Complex Documents

Breaking complex documents into smaller, focused tasks can improve overall processing quality.

### 4.1 Two-Phase Processing

```python
async def process_complex_drawing(raw_content: str, drawing_type: str, client, file_name: str) -> Dict[str, Any]:
    """
    Process complex drawings using a two-phase approach:
    1. First pass: Extract overall structure and identify key components
    2. Second pass: Process specific components with targeted prompts
    """
    ai_service = DrawingAiService(client)
    
    # First pass - identify document structure
    first_pass_prompt = f"""
    Analyze this {drawing_type} drawing and identify the key components present.
    DO NOT extract detailed content yet - just identify what sections exist.
    Return a JSON with boolean flags for detected components:
    {{
      "has_schedules": true/false,
      "has_specifications": true/false,
      "has_room_data": true/false,
      "has_panel_data": true/false,
      "has_legends": true/false,
      "document_complexity": "low"/"medium"/"high"
    }}
    """
    
    # Use truncated content for first pass to save tokens
    first_pass = await ai_service.process_with_prompt(
        raw_content=raw_content[:min(len(raw_content), 15000)],
        temperature=0.1,
        max_tokens=1000,
        system_message=first_pass_prompt
    )
    
    try:
        structure = json.loads(first_pass)
        logging.info(f"First pass analysis: {structure}")
        
        # Initialize container for combined results
        final_result = {
            "metadata": {},
            "schedules": [],
            "notes": [],
            "specifications": []
        }
        
        # Always extract metadata first
        metadata_prompt = f"""
        Extract ONLY metadata from this {drawing_type} drawing.
        Include drawing number, title, date, and revision information.
        """
        
        metadata_result = await ai_service.process_with_prompt(
            raw_content=raw_content[:min(len(raw_content), 10000)],
            temperature=0.1,
            max_tokens=2000,
            system_message=metadata_prompt
        )
        
        try:
            metadata_data = json.loads(metadata_result)
            if "metadata" in metadata_data:
                final_result["metadata"] = metadata_data["metadata"]
        except json.JSONDecodeError:
            logging.error("Error parsing metadata result")
        
        # Process components based on first pass analysis
        tasks = []
        
        # Process room data if present
        if structure.get("has_room_data", False) and drawing_type == "Architectural":
            room_prompt = """
            Extract ONLY room information from this drawing.
            Create a detailed 'rooms' array with objects for each room including:
            number, name, dimensions, and any architectural/electrical details.
            """
            tasks.append(("room_data", room_prompt, "rooms"))
            
        # Process schedules if present
        if structure.get("has_schedules", False):
            schedule_prompt = f"""
            Extract ONLY schedule information from this {drawing_type} drawing.
            Focus on tabular data and create a detailed 'schedules' array.
            """
            tasks.append(("schedules", schedule_prompt, "schedules"))
            
        # Process notes if document has enough content
        if len(raw_content) > 5000:
            notes_prompt = f"""
            Extract ONLY notes and annotations from this {drawing_type} drawing.
            Create a 'notes' array with all relevant textual information.
            """
            tasks.append(("notes", notes_prompt, "notes"))
        
        # Process each component with specialized prompts
        for task_name, prompt, result_key in tasks:
            component_result = await ai_service.process_with_prompt(
                # Send appropriate content subset based on task
                raw_content=raw_content,
                temperature=0.1,
                max_tokens=8000,
                system_message=prompt
            )
            
            try:
                component_data = json.loads(component_result)
                if result_key in component_data:
                    final_result[result_key] = component_data[result_key]
            except json.JSONDecodeError:
                logging.error(f"Error parsing {task_name} result")
        
        return final_result
        
    except json.JSONDecodeError:
        logging.error("Error in first pass analysis, falling back to standard processing")
        # Fall back to standard processing
        return await process_drawing(raw_content, drawing_type, client, file_name)
```

### 4.2 Component-Specific Processing

```python
async def process_panel_schedule(content: str, client) -> Dict[str, Any]:
    """
    Specialized processing for electrical panel schedules with a structured approach.
    """
    # Extract panel metadata first
    metadata_prompt = """
    Extract ONLY the panel metadata from this content.
    Include: panel name, voltage, amperage, phases, mains rating, and location.
    """
    
    metadata_result = await process_with_prompt(
        content=content[:min(len(content), 5000)],  # First 5000 chars for metadata
        temperature=0.05,  # Very deterministic
        max_tokens=1000,
        system_message=metadata_prompt
    )
    
    # Then extract circuit data
    circuits_prompt = """
    Extract ONLY the circuit data from this panel schedule.
    For each circuit, include:
    - circuit: The circuit number
    - load_name: The load description
    - trip: The breaker size/trip rating
    - poles: Number of poles
    - phase: Which phase(s) the circuit uses
    - notes: Any special notes for this circuit

    Normalize field names to the above format.
    """
    
    circuits_result = await process_with_prompt(
        content=content,  # Full content for circuits
        temperature=0.05,
        max_tokens=4000,
        system_message=circuits_prompt
    )
    
    # Combine results
    try:
        metadata = json.loads(metadata_result)
        circuits = json.loads(circuits_result)
        
        return {
            "panel_name": metadata.get("panel_name", ""),
            "panel_metadata": {k: v for k, v in metadata.items() if k != "panel_name"},
            "circuits": circuits.get("circuits", [])
        }
    except json.JSONDecodeError:
        logging.error("Error parsing panel schedule results")
        return {}
```

## 5. Response Post-Processing and Validation

Ensuring the output meets expected formats and standards.

### 5.1 Field Name Normalization

```python
def normalize_field_names(data: Dict[str, Any], drawing_type: str) -> Dict[str, Any]:
    """Normalize field names for consistent output format."""
    if drawing_type == "Electrical":
        # Normalize electrical field names
        if "panels" in data and isinstance(data["panels"], list):
            for panel in data["panels"]:
                if "circuits" in panel and isinstance(panel["circuits"], list):
                    for circuit in panel["circuits"]:
                        # Normalize common field name variations
                        if "description" in circuit and "load_name" not in circuit:
                            circuit["load_name"] = circuit.pop("description")
                        if "breaker" in circuit and "trip" not in circuit:
                            circuit["trip"] = circuit.pop("breaker")
                        if "load_type" in circuit and "load_name" not in circuit:
                            circuit["load_name"] = circuit.pop("load_type")
        
        # Direct circuits list
        if "circuits" in data and isinstance(data["circuits"], list):
            for circuit in data["circuits"]:
                if "description" in circuit and "load_name" not in circuit:
                    circuit["load_name"] = circuit.pop("description")
                if "breaker" in circuit and "trip" not in circuit:
                    circuit["trip"] = circuit.pop("breaker")
    
    if drawing_type == "Architectural":
        # Normalize room field names
        if "rooms" in data and isinstance(data["rooms"], list):
            for room in data["rooms"]:
                if "room_number" in room and "number" not in room:
                    room["number"] = room.pop("room_number")
                if "room_name" in room and "name" not in room:
                    room["name"] = room.pop("room_name")
                if "ceiling_height" in room and "height" not in room:
                    room["height"] = room.pop("ceiling_height")
    
    return data
```

### 5.2 JSON Validation and Repair

```python
async def validate_and_repair_json(response: str, drawing_type: str) -> Dict[str, Any]:
    """
    Validate AI response JSON and fix common issues.
    If JSON is invalid, attempt to repair common issues.
    """
    try:
        # Try to parse JSON response
        data = json.loads(response)
        
        # Normalize common field name variations
        normalized = normalize_field_names(data, drawing_type)
        
        # Validate required fields based on drawing type
        required_fields = {"metadata", "notes", "specifications"}
        
        if drawing_type == "Architectural":
            required_fields.add("rooms")
        elif drawing_type == "Electrical" and "PANEL" in response:
            required_fields.add("circuits")
        
        missing_fields = required_fields - set(normalized.keys())
        
        # Add missing required fields with empty values
        for field in missing_fields:
            if field == "rooms" or field == "circuits":
                normalized[field] = []
            elif field == "metadata":
                normalized[field] = {}
            else:
                normalized[field] = []
                
        return normalized
        
    except json.JSONDecodeError:
        # Try to fix common JSON formatting issues
        fixed_response = fix_common_json_errors(response)
        
        try:
            fixed_data = json.loads(fixed_response)
            logging.info("Successfully fixed JSON formatting issues")
            return normalize_field_names(fixed_data, drawing_type)
        except json.JSONDecodeError:
            logging.error("Could not fix JSON formatting issues")
            
            # Last resort: extract what we can with regex
            emergency_data = emergency_json_extraction(response, drawing_type)
            if emergency_data:
                return emergency_data
                
            # If all else fails, return minimal valid structure
            return {
                "metadata": {},
                "notes": [],
                "specifications": [],
                "error": "Failed to parse response JSON"
            }
            
def fix_common_json_errors(response: str) -> str:
    """Try to fix common JSON formatting errors."""
    # Remove any non-JSON text at beginning or end
    json_start = response.find("{")
    json_end = response.rfind("}")
    
    if json_start >= 0 and json_end >= 0:
        response = response[json_start:json_end+1]
    
    # Fix missing quotes around property names
    def quote_properties(match):
        return f'"{match.group(1)}":'
    
    response = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*):', quote_properties, response)
    
    # Fix trailing commas
    response = re.sub(r',\s*}', '}', response)
    response = re.sub(r',\s*]', ']', response)
    
    # Balance brackets
    open_braces = response.count('{')
    close_braces = response.count('}')
    if open_braces > close_braces:
        response += '}' * (open_braces - close_braces)
    
    open_brackets = response.count('[')
    close_brackets = response.count(']')
    if open_brackets > close_brackets:
        response += ']' * (open_brackets - close_brackets)
    
    return response
```

## 6. Performance Monitoring and Optimization

Tracking AI performance and identifying opportunities for improvement.

### 6.1 Token Usage Tracking and Cost Analysis

```python
class TokenTracker:
    """Track token usage across different document types."""
    
    def __init__(self):
        self.usage_by_type = {}
        self.usage_by_file = {}
        
    def record_usage(self, drawing_type: str, file_name: str, 
                    input_tokens: int, output_tokens: int, 
                    model: str, processing_time: float):
        """Record token usage for a processing job."""
        # Track by drawing type
        if drawing_type not in self.usage_by_type:
            self.usage_by_type[drawing_type] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_time": 0,
                "count": 0,
                "by_model": {}
            }
            
        self.usage_by_type[drawing_type]["input_tokens"] += input_tokens
        self.usage_by_type[drawing_type]["output_tokens"] += output_tokens
        self.usage_by_type[drawing_type]["total_tokens"] += input_tokens + output_tokens
        self.usage_by_type[drawing_type]["total_time"] += processing_time
        self.usage_by_type[drawing_type]["count"] += 1
        
        # Track by model
        if model not in self.usage_by_type[drawing_type]["by_model"]:
            self.usage_by_type[drawing_type]["by_model"][model] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "count": 0
            }
            
        self.usage_by_type[drawing_type]["by_model"][model]["input_tokens"] += input_tokens
        self.usage_by_type[drawing_type]["by_model"][model]["output_tokens"] += output_tokens
        self.usage_by_type[drawing_type]["by_model"][model]["total_tokens"] += input_tokens + output_tokens
        self.usage_by_type[drawing_type]["by_model"][model]["count"] += 1
        
        # Track by file
        self.usage_by_file[file_name] = {
            "drawing_type": drawing_type,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": model,
            "processing_time": processing_time
        }
        
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of token usage."""
        total_input = sum(data["input_tokens"] for data in self.usage_by_file.values())
        total_output = sum(data["output_tokens"] for data in self.usage_by_file.values())
        total_tokens = total_input + total_output
        total_time = sum(data["processing_time"] for data in self.usage_by_file.values())
        
        # Calculate average tokens per second
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        # Find most and least efficient files
        if self.usage_by_file:
            efficiency_metric = {
                file: data["total_tokens"] / data["processing_time"] 
                for file, data in self.usage_by_file.items()
                if data["processing_time"] > 0
            }
            
            most_efficient = max(efficiency_metric.items(), key=lambda x: x[1]) if efficiency_metric else None
            least_efficient = min(efficiency_metric.items(), key=lambda x: x[1]) if efficiency_metric else None
        else:
            most_efficient = None
            least_efficient = None
        
        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_processing_time": total_time,
            "tokens_per_second": tokens_per_second,
            "by_drawing_type": self.usage_by_type,
            "most_efficient_file": most_efficient,
            "least_efficient_file": least_efficient,
            "token_cost_estimate": self._calculate_cost_estimate(self.usage_by_file)
        }
    
    def _calculate_cost_estimate(self, usage_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate estimated API costs."""
        # Approximate cost per 1M tokens (as of 2025)
        cost_rates = {
            "gpt-4o": {"input": 5.0, "output": 15.0},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60}
        }
        
        total_cost = 0.0
        cost_by_model = {}
        
        for file_data in usage_data.values():
            model = file_data["model"]
            input_tokens = file_data["input_tokens"]
            output_tokens = file_data["output_tokens"]
            
            # Default to gpt-4o-mini rates if model not found
            model_rates = cost_rates.get(model, cost_rates["gpt-4o-mini"])
            
            file_cost = (
                (input_tokens / 1_000_000) * model_rates["input"] +
                (output_tokens / 1_000_000) * model_rates["output"]
            )
            
            total_cost += file_cost
            
            if model not in cost_by_model:
                cost_by_model[model] = 0.0
            cost_by_model[model] += file_cost
        
        return {
            "total_cost": total_cost,
            "by_model": cost_by_model
        }
        
    def identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for token optimization."""
        opportunities = []
        
        # Look for files with high input-to-output token ratio
        for file_name, data in self.usage_by_file.items():
            if data["input_tokens"] > 5000:  # Only consider substantial files
                input_output_ratio = data["input_tokens"] / data["output_tokens"] if data["output_tokens"] > 0 else float('inf')
                
                if input_output_ratio > 5.0:  # High ratio indicates potential for input filtering
                    opportunities.append({
                        "file": file_name,
                        "drawing_type": data["drawing_type"],
                        "issue": "high_input_ratio",
                        "ratio": input_output_ratio,
                        "recommendation": "Apply content filtering to reduce input tokens",
                        "potential_savings": f"~{int(data['input_tokens'] * 0.3)} tokens"
                    })
        
        # Look for potential model downgrades (small files using expensive models)
        for file_name, data in self.usage_by_file.items():
            if data["model"] == "gpt-4o" and data["total_tokens"] < 3000:
                opportunities.append({
                    "file": file_name,
                    "drawing_type": data["drawing_type"],
                    "issue": "model_oversized",
                    "tokens": data["total_tokens"],
                    "recommendation": "Use gpt-4o-mini for this small file",
                    "potential_savings": "~90% cost reduction for this file"
                })
        
        return opportunities

### 6.2 Adaptive Processing Pipeline

```python
class AdaptiveProcessor:
    """
    Pipeline that adapts processing strategies based on real-time performance data.
    """
    
    def __init__(self, client, token_tracker=None):
        self.client = client
        self.token_tracker = token_tracker or TokenTracker()
        self.processing_stats = {}
        self.document_complexities = {}
        
    async def process_document(self, file_path: str, drawing_type: str, output_folder: str) -> Dict[str, Any]:
        """Process a document with adaptive strategies based on previous performance."""
        file_name = os.path.basename(file_path)
        
        # Step 1: Extract content
        extractor = create_extractor(drawing_type)
        extraction_result = await extractor.extract(file_path)
        if not extraction_result.success:
            return {"success": False, "error": extraction_result.error}
            
        raw_content = extraction_result.raw_text
        for table in extraction_result.tables:
            raw_content += f"\nTABLE:\n{table['content']}\n"
        
        # Step 2: Analyze document complexity
        complexity = self._analyze_complexity(raw_content, drawing_type, file_name)
        self.document_complexities[file_name] = complexity
        
        # Step 3: Select optimal processing strategy based on complexity
        strategy = self._select_strategy(complexity, drawing_type, file_name)
        
        # Step 4: Apply pre-processing if needed
        if strategy.get("preprocess", False):
            raw_content = preprocess_for_ai(raw_content, drawing_type, file_name)
            
        if strategy.get("filter_content", False):
            raw_content = filter_irrelevant_content(raw_content, drawing_type)
            
        if strategy.get("token_budget"):
            raw_content = apply_token_budget(raw_content, drawing_type, strategy["token_budget"])
        
        # Step 5: Process with selected parameters
        start_time = time.time()
        
        if strategy.get("multi_pass", False):
            result = await process_complex_drawing(raw_content, drawing_type, self.client, file_name)
            result_json = json.dumps(result)
        else:
            ai_service = DrawingAiService(self.client)
            system_message = get_enhanced_system_message(drawing_type, file_name)
            
            if "field_mapping" in strategy and strategy["field_mapping"]:
                system_message += "\n\n" + get_field_mapping_instructions(drawing_type)
            
            response = await ai_service.process_with_prompt(
                raw_content=raw_content,
                temperature=strategy.get("temperature", 0.1),
                max_tokens=strategy.get("max_tokens", 8000),
                model_type=ModelType(strategy.get("model", "gpt-4o-mini")),
                system_message=system_message
            )
            
            # Post-process result if needed
            if strategy.get("validate_json", True):
                result = await validate_and_repair_json(response, drawing_type)
                result_json = json.dumps(result)
            else:
                result_json = response
        
        # Calculate processing time and token usage
        processing_time = time.time() - start_time
        
        # Record token usage information from response
        input_tokens = len(raw_content) / 4  # Approximate
        output_tokens = len(result_json) / 4  # Approximate
        
        self.token_tracker.record_usage(
            drawing_type=drawing_type,
            file_name=file_name,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            model=strategy.get("model", "gpt-4o-mini"),
            processing_time=processing_time
        )
        
        # Record processing stats for this file type
        if drawing_type not in self.processing_stats:
            self.processing_stats[drawing_type] = []
            
        self.processing_stats[drawing_type].append({
            "file_name": file_name,
            "complexity": complexity,
            "processing_time": processing_time,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "strategy": strategy
        })
        
        # Save result to output folder
        storage = FileSystemStorage()
        type_folder = os.path.join(output_folder, drawing_type)
        os.makedirs(type_folder, exist_ok=True)
        output_path = os.path.join(type_folder, f"{os.path.splitext(file_name)[0]}_structured.json")
        
        await storage.save_text(result_json, output_path)
        
        return {
            "success": True,
            "file": output_path,
            "processing_time": processing_time,
            "token_usage": {
                "input": int(input_tokens),
                "output": int(output_tokens),
                "total": int(input_tokens + output_tokens)
            }
        }
    
    def _analyze_complexity(self, content: str, drawing_type: str, file_name: str) -> float:
        """Analyze document complexity on a scale of 0-10."""
        char_count = len(content)
        table_count = content.count("TABLE:")
        line_count = content.count("\n")
        
        # Base complexity based on content size
        if char_count > 30000:
            base_complexity = 8.0
        elif char_count > 15000:
            base_complexity = 6.0
        elif char_count > 5000:
            base_complexity = 4.0
        else:
            base_complexity = 2.0
            
        # Adjust for tables (tabular data is complex)
        table_factor = min(table_count * 0.5, 2.0)
        
        # Adjust for document type
        type_factor = 0.0
        if "SPECIFICATION" in file_name.upper():
            type_factor = 2.0
        elif drawing_type == "Electrical" and "PANEL" in file_name.upper():
            type_factor = 1.5
        elif drawing_type == "Architectural":
            type_factor = 1.0
            
        # Final complexity score (0-10 scale)
        complexity = min(base_complexity + table_factor + type_factor, 10.0)
        
        return complexity
    
    def _select_strategy(self, complexity: float, drawing_type: str, file_name: str) -> Dict[str, Any]:
        """Select optimal processing strategy based on document complexity."""
        # Default strategy (for mid-complexity documents)
        strategy = {
            "model": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 8000,
            "preprocess": True,
            "filter_content": False,
            "validate_json": True,
            "field_mapping": True,
            "multi_pass": False
        }
        
        # Adjust for very simple documents
        if complexity < 3.0:
            strategy.update({
                "temperature": 0.05,
                "max_tokens": 4000,
                "filter_content": False,
                "field_mapping": False
            })
            
        # Adjust for complex documents
        elif complexity > 7.0:
            # Very complex documents need GPT-4o and maybe multi-pass
            strategy.update({
                "model": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 12000,
                "filter_content": False,  # Don't risk losing important content
                "multi_pass": complexity > 8.5  # Only use multi-pass for very complex docs
            })
        
        # Document-specific adjustments
        if "SPECIFICATION" in file_name.upper():
            strategy.update({
                "model": "gpt-4o",
                "temperature": 0.2,
                "max_tokens": 12000,
                "filter_content": False,
                "token_budget": 32000  # Higher budget for specifications
            })
        elif drawing_type == "Electrical" and "PANEL" in file_name.upper():
            strategy.update({
                "temperature": 0.05,  # Very low temperature for deterministic output
                "field_mapping": True  # Especially important for panels
            })
            
        # Consider previous performance data if available
        similar_files = [
            stats for stats in self.processing_stats.get(drawing_type, [])
            if abs(stats["complexity"] - complexity) < 1.0
        ]
        
        if similar_files:
            # Find the strategy that worked best for similar files
            best_strategy = min(similar_files, key=lambda x: x["processing_time"])
            
            # Adopt aspects of the best strategy
            strategy.update({
                "model": best_strategy["strategy"]["model"],
                "temperature": best_strategy["strategy"]["temperature"],
                "multi_pass": best_strategy["strategy"]["multi_pass"]
            })
        
        return strategy
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        token_summary = self.token_tracker.get_summary()
        
        # Calculate averages by drawing type
        avg_time_by_type = {}
        for dtype, stats in self.processing_stats.items():
            if stats:
                avg_time = sum(s["processing_time"] for s in stats) / len(stats)
                avg_time_by_type[dtype] = avg_time
        
        # Find optimal strategies
        optimal_strategies = {}
        for dtype, stats in self.processing_stats.items():
            if not stats:
                continue
                
            # Group by complexity level (rounded to nearest integer)
            by_complexity = {}
            for stat in stats:
                complexity_level = round(stat["complexity"])
                if complexity_level not in by_complexity:
                    by_complexity[complexity_level] = []
                by_complexity[complexity_level].append(stat)
            
            # Find best strategy for each complexity level
            for level, level_stats in by_complexity.items():
                best_stat = min(level_stats, key=lambda x: x["processing_time"])
                
                if dtype not in optimal_strategies:
                    optimal_strategies[dtype] = {}
                    
                optimal_strategies[dtype][level] = {
                    "model": best_stat["strategy"]["model"],
                    "temperature": best_stat["strategy"]["temperature"],
                    "multi_pass": best_stat["strategy"]["multi_pass"],
                    "processing_time": best_stat["processing_time"],
                    "file_example": best_stat["file_name"]
                }
        
        return {
            "token_usage": token_summary,
            "avg_processing_time_by_type": avg_time_by_type,
            "optimal_strategies": optimal_strategies,
            "optimization_opportunities": self.token_tracker.identify_optimization_opportunities()
        }
```

### 6.3 Performance Visualization

```python
def generate_performance_charts(processor: AdaptiveProcessor, output_dir: str):
    """Generate performance visualization charts."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    report = processor.get_performance_report()
    token_usage = report["token_usage"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Chart 1: Token usage by drawing type
    plt.figure(figsize=(12, 8))
    drawing_types = list(token_usage["by_drawing_type"].keys())
    input_tokens = [token_usage["by_drawing_type"][dt]["input_tokens"] for dt in drawing_types]
    output_tokens = [token_usage["by_drawing_type"][dt]["output_tokens"] for dt in drawing_types]
    
    x = np.arange(len(drawing_types))
    width = 0.35
    
    plt.bar(x - width/2, input_tokens, width, label='Input Tokens')
    plt.bar(x + width/2, output_tokens, width, label='Output Tokens')
    
    plt.xlabel('Drawing Type')
    plt.ylabel('Token Count')
    plt.title('Token Usage by Drawing Type')
    plt.xticks(x, drawing_types)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_usage_by_type.png'))
    
    # Chart 2: Processing time vs complexity
    plt.figure(figsize=(12, 8))
    
    for drawing_type, stats in processor.processing_stats.items():
        if stats:
            complexities = [s["complexity"] for s in stats]
            times = [s["processing_time"] for s in stats]
            plt.scatter(complexities, times, label=drawing_type, alpha=0.7)
    
    plt.xlabel('Document Complexity')
    plt.ylabel('Processing Time (s)')
    plt.title('Processing Time vs Document Complexity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_vs_complexity.png'))
    
    # Chart 3: Model effectiveness
    plt.figure(figsize=(12, 8))
    
    model_stats = {}
    for drawing_type, stats in processor.processing_stats.items():
        for stat in stats:
            model = stat["strategy"]["model"]
            if model not in model_stats:
                model_stats[model] = {"complexities": [], "times": []}
            model_stats[model]["complexities"].append(stat["complexity"])
            model_stats[model]["times"].append(stat["processing_time"])
    
    for model, data in model_stats.items():
        plt.scatter(data["complexities"], data["times"], label=model, alpha=0.7)
    
    plt.xlabel('Document Complexity')
    plt.ylabel('Processing Time (s)')
    plt.title('Model Performance by Document Complexity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance.png'))
    
    # Save the report as JSON
    with open(os.path.join(output_dir, 'performance_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
```

## 7. Implementation Integration

Here's how to integrate these AI-first optimization strategies into your existing pipeline.

### 7.1 Phased Integration Plan

```python
# phase1.py - Initial integration of enhanced prompting

from config.settings import get_force_mini_model
from services.ai_service import process_drawing, optimize_model_parameters
from utils.performance_utils import get_tracker

# Step 1: Update drawing_instructions.py with enhanced prompts
DRAWING_INSTRUCTIONS = {
    "Electrical": """
    Focus on panel schedules, circuit info, equipment schedules with electrical characteristics, and installation notes.
    
    MAP THESE FIELDS CONSISTENTLY:
    - "Circuit", "CKT", "CKT #" → "circuit"
    - "Description", "Load", "Device" → "load_name"
    - "Poles", "P" → "poles"
    - "Amperes", "A", "Amps", "Trip" → "trip"
    - "Phase", "PH", "φ" → "phase"
    - "Wire Size", "AWG", "Conductor" → "wire_size"
    """,
    "Architectural": """
    Extract and structure the following information:
    1. Room details: Create a 'rooms' array with objects for each room, including:
       - 'number': Room number (as a string)
       - 'name': Room name
       - 'finish': Ceiling finish
       - 'height': Ceiling height
       - 'electrical_info': Any electrical specifications
       - 'architectural_info': Any additional architectural details
    
    ALWAYS include a 'rooms' array, even if you have to infer room information from context.
    Room entries should be comprehensive but avoid duplication.
    """,
    # Other drawing types remain unchanged for phase 1
}

# Step 2: Update process_drawing function to use enhanced prompting
async def process_drawing_enhanced(raw_content: str, drawing_type: str, client, file_name: str) -> str:
    """Enhanced version of process_drawing with improved prompting."""
    try:
        # Get optimized parameters (same as original)
        params = optimize_model_parameters(drawing_type, raw_content, file_name)
        
        # Enhanced system message with field mapping
        system_message = f"""
        You are processing a construction drawing. Extract all relevant information and organize it into a JSON object with the following sections:
        - 'metadata': Include drawing number, title, date, etc.
        - 'schedules': Array of schedules with type and data.
        - 'notes': Array of notes.
        - 'specifications': Array of specification sections.
        
        {DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"])}
        
        Ensure the output is valid JSON.
        """
        
        # Log token estimation for monitoring
        estimated_tokens = len(raw_content) // 4
        logging.info(f"Estimated input tokens: ~{estimated_tokens}")
        
        # Rest of the implementation remains the same as the original
        # ...
        
    except Exception as e:
        logging.error(f"Error processing {drawing_type} drawing: {str(e)}")
        raise
        
# Step 3: Integrate token tracking (minimal initial version)
token_tracker = {}

def record_token_usage(drawing_type: str, file_name: str, input_tokens: int, output_tokens: int):
    """Record token usage for analysis."""
    if drawing_type not in token_tracker:
        token_tracker[drawing_type] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "count": 0
        }
        
    token_tracker[drawing_type]["input_tokens"] += input_tokens
    token_tracker[drawing_type]["output_tokens"] += output_tokens
    token_tracker[drawing_type]["count"] += 1
    
    # Log for immediate visibility
    logging.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
```

### 7.2 Configuration Settings for Adaptive Processing

```python
# settings.py additions for adaptive processing

# Adaptive Processing Configuration
ENABLE_ADAPTIVE_PROCESSING = os.getenv('ENABLE_ADAPTIVE_PROCESSING', 'false').lower() == 'true'
USE_MULTI_PASS_FOR_COMPLEX = os.getenv('USE_MULTI_PASS_FOR_COMPLEX', 'false').lower() == 'true'
ENABLE_TOKEN_OPTIMIZATION = os.getenv('ENABLE_TOKEN_OPTIMIZATION', 'false').lower() == 'true'
MAX_INPUT_TOKENS = int(os.getenv('MAX_INPUT_TOKENS', '8000'))

# API Rate Limiting Settings (important for adaptive processing)
API_CONCURRENT_REQUESTS = int(os.getenv('API_CONCURRENT_REQUESTS', '5'))
API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '60'))
API_TIME_WINDOW = int(os.getenv('API_TIME_WINDOW', '60'))

# Performance Monitoring
ENABLE_PERFORMANCE_TRACKING = os.getenv('ENABLE_PERFORMANCE_TRACKING', 'true').lower() == 'true'
PERFORMANCE_LOG_DIR = os.getenv('PERFORMANCE_LOG_DIR', 'performance_logs')

def update_settings():
    """Reload settings from environment variables (useful for dynamic updates)."""
    global ENABLE_ADAPTIVE_PROCESSING, USE_MULTI_PASS_FOR_COMPLEX, ENABLE_TOKEN_OPTIMIZATION
    global MAX_INPUT_TOKENS, API_CONCURRENT_REQUESTS, API_RATE_LIMIT, API_TIME_WINDOW
    global ENABLE_PERFORMANCE_TRACKING, PERFORMANCE_LOG_DIR
    
    load_dotenv(override=True)
    
    ENABLE_ADAPTIVE_PROCESSING = os.getenv('ENABLE_ADAPTIVE_PROCESSING', 'false').lower() == 'true'
    USE_MULTI_PASS_FOR_COMPLEX = os.getenv('USE_MULTI_PASS_FOR_COMPLEX', 'false').lower() == 'true'
    ENABLE_TOKEN_OPTIMIZATION = os.getenv('ENABLE_TOKEN_OPTIMIZATION', 'false').lower() == 'true'
    MAX_INPUT_TOKENS = int(os.getenv('MAX_INPUT_TOKENS', '8000'))
    API_CONCURRENT_REQUESTS = int(os.getenv('API_CONCURRENT_REQUESTS', '5'))
    API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '60'))
    API_TIME_WINDOW = int(os.getenv('API_TIME_WINDOW', '60'))
    ENABLE_PERFORMANCE_TRACKING = os.getenv('ENABLE_PERFORMANCE_TRACKING', 'true').lower() == 'true'
    PERFORMANCE_LOG_DIR = os.getenv('PERFORMANCE_LOG_DIR', 'performance_logs')
```

### 7.3 Main Entry Point Updates

```python
# main.py updates for AI-first optimization

async def main_async():
    """
    Main async function with AI-first optimizations integrated.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_folder> [output_folder]")
        return 1
    
    job_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(job_folder, "output")
    
    if not os.path.exists(job_folder):
        print(f"Error: Input folder '{job_folder}' does not exist.")
        return 1
    
    # 1) Set up logging
    setup_logging(output_folder)
    logging.info(f"Processing files from: {job_folder}")
    logging.info(f"Output will be saved to: {output_folder}")
    logging.info(f"Application settings: {get_all_settings()}")
    
    try:
        # 2) Create OpenAI Client
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # 3) Initialize performance tracking
        tracker = get_tracker()
        if ENABLE_PERFORMANCE_TRACKING:
            perf_dir = os.path.join(output_folder, PERFORMANCE_LOG_DIR)
            os.makedirs(perf_dir, exist_ok=True)
        
        # 4) Process files based on configuration
        start_time = time.time()
        
        if ENABLE_ADAPTIVE_PROCESSING:
            # Use the adaptive processor for intelligent processing
            processor = AdaptiveProcessor(client)
            
            # Get list of PDF files
            pdf_files = traverse_job_folder(job_folder)
            logging.info(f"Found {len(pdf_files)} PDF files for adaptive processing")
            
            # Process each file with adaptive strategy
            results = []
            for pdf_file in pdf_files:
                drawing_type = get_drawing_type(pdf_file)
                result = await processor.process_document(pdf_file, drawing_type, output_folder)
                results.append(result)
                
            # Generate performance report
            if ENABLE_PERFORMANCE_TRACKING:
                report = processor.get_performance_report()
                with open(os.path.join(perf_dir, 'adaptive_report.json'), 'w') as f:
                    json.dump(report, f, indent=2)
                
                # Generate performance visualizations
                generate_performance_charts(processor, perf_dir)
        else:
            # Use standard processing with enhanced prompts
            await process_job_site_async(job_folder, output_folder, client)
        
        # 5) Calculate total processing time
        total_time = time.time() - start_time
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        
        # 6) Generate performance report
        tracker.log_report()
        
        return 0
    except Exception as e:
        logging.error(f"Unhandled exception in main process: {str(e)}")
        return 1
```

## 8. Scaling Beyond Local Processing

When you reach the limits of local processing, these strategies help scale to distributed processing.

### 8.1 Serverless Processing Architecture

```python
# cloudfunction.py - Example Google Cloud Function implementation

import json
import logging
import os
from google.cloud import storage
from typing import Dict, Any

from services.extraction_service import PyMuPdfExtractor
from services.ai_service import process_drawing_enhanced

# Setup logging
logging.basicConfig(level=logging.INFO)

# Initialize services
storage_client = storage.Client()

async def process_pdf_cloudfunction(event, context):
    """
    Cloud Function entrypoint to process a single PDF file.
    Triggered by a new PDF upload to a Cloud Storage bucket.
    
    Args:
        event: Cloud Storage event containing file info
        context: Metadata about the event
    """
    # Get file info from event
    bucket_name = event['bucket']
    file_name = event['name']
    
    logging.info(f"Processing file: {file_name} from bucket: {bucket_name}")
    
    try:
        # Download file to local storage
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        temp_file_path = f"/tmp/{os.path.basename(file_name)}"
        blob.download_to_filename(temp_file_path)
        
        # Determine drawing type from filename
        drawing_type = determine_drawing_type(file_name)
        
        # Extract content
        extractor = PyMuPdfExtractor()
        extraction_result = await extractor.extract(temp_file_path)
        
        if not extraction_result.success:
            raise Exception(f"Extraction failed: {extraction_result.error}")
        
        # Combine content
        raw_content = extraction_result.raw_text
        for table in extraction_result.tables:
            raw_content += f"\nTABLE:\n{table['content']}\n"
        
        # Process with AI
        client = get_openai_client()
        result_json = await process_drawing_enhanced(raw_content, drawing_type, client, file_name)
        
        # Save result back to cloud storage
        output_filename = f"{os.path.splitext(file_name)[0]}_processed.json"
        output_blob = bucket.blob(f"output/{drawing_type}/{output_filename}")
        output_blob.upload_from_string(result_json, content_type='application/json')
        
        # Clean up
        os.remove(temp_file_path)
        
        # Publish completion message
        publish_completion_message(file_name, True, None)
        
        return {"success": True, "file": file_name}
        
    except Exception as e:
        logging.error(f"Error processing {file_name}: {str(e)}")
        publish_completion_message(file_name, False, str(e))
        return {"success": False, "file": file_name, "error": str(e)}
        
def determine_drawing_type(file_name: str) -> str:
    """Determine drawing type from filename."""
    file_name = file_name.upper()
    
    if any(s in file_name for s in ["A-", "ARCH-", "FLOOR-PLAN"]):
        return "Architectural"
    elif any(s in file_name for s in ["E-", "ELEC-", "PANEL", "LIGHTING"]):
        return "Electrical"
    elif any(s in file_name for s in ["M-", "MECH-", "HVAC"]):
        return "Mechanical"
    elif any(s in file_name for s in ["P-", "PLUMB-"]):
        return "Plumbing"
    elif "SPEC" in file_name:
        return "Specifications"
    
    return "General"
    
def get_openai_client():
    """Get OpenAI client."""
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=os.environ.get('OPENAI_API_KEY'))
    
def publish_completion_message(file_name: str, success: bool, error: str = None):
    """Publish completion message to a Pub/Sub topic."""
    from google.cloud import pubsub_v1
    
    project_id = os.environ.get('GCP_PROJECT_ID')
    topic_id = os.environ.get('COMPLETION_TOPIC_ID')
    
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_id)
    
    message = {
        "file_name": file_name,
        "success": success,
        "timestamp": time.time(),
    }
    
    if error:
        message["error"] = error
        
    message_data = json.dumps(message).encode('utf-8')
    publisher.publish(topic_path, data=message_data)
```

### 8.2 Distributed Job Orchestration

```python
# orchestrator.py - Distributed job orchestration

import os
import json
import asyncio
import logging
from typing import List, Dict, Any
from google.cloud import storage, pubsub_v1

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedJobOrchestrator:
    """
    Orchestrates processing of construction drawings across distributed cloud functions.
    Handles job tracking, retries, and result aggregation.
    """
    
    def __init__(self, project_id: str, input_bucket: str, output_bucket: str):
        """Initialize the orchestrator."""
        self.project_id = project_id
        self.input_bucket = input_bucket
        self.output_bucket = output_bucket
        self.storage_client = storage.Client()
        self.publisher = pubsub_v1.PublisherClient()
        self.job_status = {}
        
    async def process_job_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        Process all files in a job folder using distributed cloud functions.
        
        Args:
            folder_path: Path to the folder containing PDFs
            
        Returns:
            Dictionary with job status and results
        """
        # Upload files to input bucket
        logger.info(f"Uploading files from {folder_path} to gs://{self.input_bucket}")
        uploaded_files = await self._upload_files(folder_path)
        
        # Create job tracking record
        job_id = f"job_{int(time.time())}"
        self.job_status[job_id] = {
            "total_files": len(uploaded_files),
            "completed": 0,
            "failed": 0,
            "pending": len(uploaded_files),
            "files": {file: {"status": "pending"} for file in uploaded_files}
        }
        
        # Trigger processing for each file
        logger.info(f"Triggering processing for {len(uploaded_files)} files")
        for file_path in uploaded_files:
            await self._trigger_processing(file_path, job_id)
            
        # Wait for job completion
        while self.job_status[job_id]["pending"] > 0:
            logger.info(f"Waiting for job completion. Pending: {self.job_status[job_id]['pending']}")
            await asyncio.sleep(5)
            await self._update_job_status(job_id)
            
        # Collect and aggregate results
        results = await self._collect_results(job_id)
        
        return {
            "job_id": job_id,
            "total_files": len(uploaded_files),
            "successful": self.job_status[job_id]["completed"],
            "failed": self.job_status[job_id]["failed"],
            "results": results
        }
        
    async def _upload_files(self, folder_path: str) -> List[str]:
        """Upload files to input bucket and return list of uploaded file paths."""
        uploaded_files = []
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    local_path = os.path.join(root, file)
                    relative_path = os.path.relpath(local_path, folder_path)
                    
                    # Upload to cloud storage
                    bucket = self.storage_client.bucket(self.input_bucket)
                    blob = bucket.blob(relative_path)
                    blob.upload_from_filename(local_path)
                    
                    uploaded_files.append(relative_path)
                    
        return uploaded_files
        
    async def _trigger_processing(self, file_path: str, job_id: str):
        """Trigger cloud function processing for a file."""
        # Create a message with file info
        message = {
            "bucket": self.input_bucket,
            "name": file_path,
            "job_id": job_id
        }
        
        # Publish to trigger topic
        topic_path = self.publisher.topic_path(self.project_id, "pdf-processing-trigger")
        message_data = json.dumps(message).encode("utf-8")
        self.publisher.publish(topic_path, data=message_data)
        
    async def _update_job_status(self, job_id: str):
        """Update job status by checking completion messages."""
        # In a real implementation, this would check a database or message queue
        # For this example, we'll simulate by checking output bucket
        
        bucket = self.storage_client.bucket(self.output_bucket)
        
        for file_path, file_status in self.job_status[job_id]["files"].items():
            if file_status["status"] == "pending":
                # Check if result file exists
                output_path = f"results/{os.path.splitext(file_path)[0]}_result.json"
                blob = bucket.blob(output_path)
                
                if blob.exists():
                    # Result exists, download and check status
                    content = blob.download_as_text()
                    result = json.loads(content)
                    
                    if result.get("success", False):
                        file_status["status"] = "completed"
                        self.job_status[job_id]["completed"] += 1
                        self.job_status[job_id]["pending"] -= 1
                    else:
                        file_status["status"] = "failed"
                        file_status["error"] = result.get("error", "Unknown error")
                        self.job_status[job_id]["failed"] += 1
                        self.job_status[job_id]["pending"] -= 1
        
    async def _collect_results(self, job_id: str) -> Dict[str, Any]:
        """Collect and aggregate processing results."""
        bucket = self.storage_client.bucket(self.output_bucket)
        results_by_type = {}
        
        for file_path, file_status in self.job_status[job_id]["files"].items():
            if file_status["status"] == "completed":
                output_path = f"results/{os.path.splitext(file_path)[0]}_result.json"
                blob = bucket.blob(output_path)
                
                if blob.exists():
                    content = blob.download_as_text()
                    result = json.loads(content)
                    
                    # Group by drawing type
                    drawing_type = result.get("drawing_type", "Unknown")
                    if drawing_type not in results_by_type:
                        results_by_type[drawing_type] = []
                        
                    results_by_type[drawing_type].append(result)
        
        return results_by_type
```

## 9. Conclusion and Best Practices

### 9.1 Key Principles of AI-First Optimization

1. **Start with effective prompting**: The foundation of AI-first optimization is crafting prompts that guide the model to produce the desired output format and content.

2. **Pre-process intelligently**: Highlight structure rather than just removing content, helping the model understand the document organization.

3. **Use the right model for the right task**: Match model capabilities to document complexity for optimal performance and cost-effectiveness.

4. **Implement selective processing**: Complex documents may benefit from multi-pass processing with focused prompts for different components.

5. **Validate and normalize outputs**: Post-process AI outputs to ensure consistency in field naming and data structure.

6. **Track performance and learn**: Monitor token usage and processing times to identify optimization opportunities.

7. **Scale horizontally when needed**: When local processing reaches its limits, transition to distributed cloud-based processing.

### 9.2 Implementation Checklist

- [ ] Enhance system prompts with document-type specific instructions
- [ ] Implement field name normalization for consistent outputs
- [ ] Add content pre-processing to highlight document structure
- [ ] Integrate token usage tracking and performance monitoring
- [ ] Implement selective model usage based on document complexity
- [ ] Add validation and repair for AI-generated JSON
- [ ] Set up scaling pathways for processing larger document sets
```