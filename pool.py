import json
import re

def build_college_pool(txt_filepath):
    colleges = {"tier_1": [], "tier_2": [], "tier_3": []}
    
    with open(txt_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 1. Strip out the PDF tags
        clean_line = re.sub(r'\\s*', '', line)
        
        # 2. Find the College Name and Tier (handles missing spaces like "NameTier 1")
        # This regex captures everything up to "Tier X" as the name, and "Tier X" as the tier.
        match = re.search(r'^(.*?)\s*(Tier\s*[123])$', clean_line, re.IGNORECASE)
        
        if match:
            name = match.group(1).strip()
            tier_str = match.group(2).lower().replace(' ', '_') # Formats to "tier_1"
            
            # Ignore headers or empty names that slip through
            if name and "Institution Name" not in name:
                colleges[tier_str].append(name)

    # 3. Deduplicate the lists (just in case)
    for t in colleges:
        colleges[t] = list(set(colleges[t]))

    # Save to JSON
    with open('colleges_pool.json', 'w', encoding='utf-8') as f:
        json.dump(colleges, f, indent=4)
        
    print(f"✅ Processed {sum(len(v) for v in colleges.values())} colleges into colleges_pool.json")

def parse_ctc(ctc_str):
    """
    Extracts the highest numerical value from a CTC string for dynamic tiering.
    Example: '₹80-90 LPA' -> 90.0, '₹67.3 LPA' -> 67.3
    """
    # Find all numbers (including decimals) in the string
    numbers = re.findall(r'\d+\.?\d*', ctc_str.replace(',', ''))
    if numbers:
        # If it's a range, sort to evaluate the upper limit
        return max(float(n) for n in numbers)
    return 0.0

def build_company_pool(markdown_filepath):
    # Initialize the entity pools
    companies = {
        "tier_1": [], "tier_2": [], "tier_3": [], "tier_4": [], "tier_5": []
    }
    
    current_tier = None
    current_domain = "General Software Engineering"

    with open(markdown_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        
        # 1. Detect Tiers from headers (e.g., "## TIER 1: PREMIUM TECH...")
        tier_match = re.search(r'## TIER (\d)', line, re.IGNORECASE)
        if tier_match:
            current_tier = f"tier_{tier_match.group(1)}"
            continue
            
        # 2. Detect Specialized Sectors 
        if '## SPECIALIZED SECTORS' in line.upper():
            current_tier = "specialized"
            continue
            
        # 3. Detect Domain Subheadings (e.g., "### High-Frequency Trading")
        if line.startswith('### '):
            current_domain = line.replace('### ', '').strip()
            continue

        # 4. Parse Table Rows
        # Validates that it's a table row, not a header row, and not a separator
        if line.startswith('|') and 'Company' not in line and '---' not in line:
            # Split by pipe and clean whitespace
            parts = [p.strip() for p in line.split('|')]
            
            # A standard markdown table row like | Name | CTC | Intern | Process | Notes |
            # split by '|' results in empty strings at the ends: ['', 'Name', 'CTC', ..., 'Notes', '']
            if len(parts) >= 6:
                name = parts[1]
                ctc = parts[2]
                tags_notes = parts[5]
                
                if not name: # Skip empty rows
                    continue
                
                # 5. Handle Tier Assignment (especially for specialized sectors)
                assigned_tier = current_tier
                if assigned_tier == "specialized" or not assigned_tier:
                    max_ctc = parse_ctc(ctc)
                    if max_ctc >= 30:
                        assigned_tier = "tier_1"
                    elif max_ctc >= 20:
                        assigned_tier = "tier_2"
                    elif max_ctc >= 15:
                        assigned_tier = "tier_3"
                    elif max_ctc >= 10:
                        assigned_tier = "tier_4"
                    else:
                        assigned_tier = "tier_5"

                # 6. Append to the structured JSON payload
                if assigned_tier in companies:
                    companies[assigned_tier].append({
                        "name": name,
                        "domain": current_domain,
                        "tags": [tags_notes.lower()] if tags_notes else [],
                        "estimated_ctc": ctc
                    })

    # Save the output
    with open('companies_pool.json', 'w', encoding='utf-8') as f:
        json.dump(companies, f, indent=4)
    
    # Print Generation Stats
    total = sum(len(v) for v in companies.values())
    print(f"✅ Processed {total} companies into 'companies_pool.json'")
    for t, comps in companies.items():
        print(f"  - {t}: {len(comps)} companies")

if __name__ == "__main__":
    # Ensure your file is named Company_Lists.md in the same directory
    build_company_pool('Company_Lists.md')