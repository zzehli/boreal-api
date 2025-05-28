from typing import List

from pydantic import BaseModel, Field


class ReferenceItem(BaseModel):
    index: int = Field(description="The index of the document in the context")
    title: str = Field(description="The title or identifier of the document")
    url: str = Field(description="The url of the document")

class ResponseWithCitation(BaseModel):
    response: str = Field(description="The response to user's question")
    reference: List[ReferenceItem] = Field(description="A list of references from the context used to generate the response")
