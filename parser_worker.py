import re
from pydantic import BaseModel, ValidationError
from typing import Optional

class CardMetadata(BaseModel):
    emp_id: Optional[str] = None
    name: Optional[str] = None
    company_name: Optional[str] = None  # Corporate Client Account Name
    policy_no: Optional[str] = None     # Specific Insurance Contract Number
    policy_type: Optional[str] = None
    card_no: Optional[str] = None
    relationship: Optional[str] = None
    age: Optional[int] = None
    valid_up_to: Optional[str] = None

def extract_metadata_from_text(text: str) -> CardMetadata:
    data = {}
    
    # Standardize separator spacing to make extraction highly reliable
    text = text.replace('\n:', ':')
    
    # 1. Employee ID
    emp_match = re.search(r"(?i)Emp\.?\s*ID\.?\s*[:\-]?\s*([^\n]+)", text)
    if emp_match: 
        data['emp_id'] = emp_match.group(1).split('\n')[0].replace(" ", "").strip().upper()
    
    # 2. Name
    name_match = re.search(r"(?i)Name\s*[:\-]?\s*([^\n]+)", text)
    if name_match: 
        data['name'] = name_match.group(1).split('\n')[0].strip().upper()
        
    # 3. Company / Corporate Name
    company_match = re.search(r"(?i)(Company\s*Name|Corporate\s*Name|Group\s*Name|Corporate|Client\s*Name)\s*[:\-]?\s*([^\n]+)", text)
    if company_match:
        data['company_name'] = company_match.group(2).split('\n')[0].strip().upper()
        
    # 4. Smart Heuristic for Unlabeled Company Names (e.g., ICICI Lombard header-based matching)
    if not data.get('company_name'):
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        for idx, line in enumerate(lines):
            if "Health Care Card" in line or "Healthcare Card" in line or "ICICI" in line:
                if idx + 1 < len(lines):
                    next_line = lines[idx+1]
                    if "NAME" not in next_line and "POLICY" not in next_line and "CARD" not in next_line:
                        data['company_name'] = next_line.strip().upper()
                        break

    # 5. Policy Number (Line-based capture)
    policy_match = re.search(r"(?i)Policy\s*No\.?\s*[:\-]?\s*([^\n]+)", text)
    if policy_match: 
        raw_policy = policy_match.group(1).split('\n')[0].replace(" ", "").strip().upper()
        data['policy_no'] = re.sub(r"[^A-Z0-9\/\-]", "", raw_policy)

    # 6. Policy Type
    type_match = re.search(r"(?i)Policy\s*Type\s*[:\-]?\s*([^\n]+)", text)
    if type_match: 
        data['policy_type'] = type_match.group(1).split('\n')[0].strip().upper()
        
    # 7. Card Number
    card_match = re.search(r"(?i)Card\s*No\.?\s*[:\-]?\s*([^\n]+)", text)
    if card_match: 
        data['card_no'] = card_match.group(1).split('\n')[0].replace(" ", "").strip().upper()
        
    # 8. Relationship
    rel_match = re.search(r"(?i)Relationship\s*[:\-]?\s*([^\n]+)", text)
    if rel_match: 
        data['relationship'] = rel_match.group(1).split('\n')[0].strip().upper()
        
    # 9. Age
    age_match = re.search(r"(?i)Age\s*[:\-]?\s*(\d+)", text)
    if age_match: 
        data['age'] = int(age_match.group(1).strip())
        
    # 10. Valid Up To / Valid To
    valid_to_match = re.search(r"(?i)Valid\s*To\s*[:\-]?\s*([A-Za-z0-9\s\-]+)", text)
    if valid_to_match:
        data['valid_up_to'] = valid_to_match.group(1).split('\n')[0].strip().upper()
    else:
        valid_match = re.search(r"(?i)Valid\s*(Up\s*To|To)\s*[:\-]?\s*([^\n]+)", text)
        if valid_match: 
            data['valid_up_to'] = valid_match.group(2).split('\n')[0].strip().upper()

    try:
        return CardMetadata(**data)
    except ValidationError:
        return CardMetadata()