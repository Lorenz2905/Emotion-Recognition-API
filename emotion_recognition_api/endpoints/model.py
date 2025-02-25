from pydantic import BaseModel
from typing import List, Dict, Union

class ContentItem(BaseModel):
    type: str
    image: Union[str, None] = None
    text: Union[str, None] = None

class Message(BaseModel):
    role: str
    content: List[ContentItem]

class AnalyseRequest(BaseModel):
    messages: List[Message]
