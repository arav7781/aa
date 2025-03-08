from dataclasses import dataclass
from typing import List, Optional

@dataclass
class InputData:
    syllabus_name: str
    data_path: str
    data_description: str
    units: Optional[List[str]] = None
    objectives: Optional[List[str]] = None
    text_content: Optional[str] = None