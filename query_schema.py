

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union


class FilterSchema(BaseModel):
    genre: Optional[str] = Field(None, description="Genre of the movie")
    min_year: Optional[int] = Field(None, description="Minimum release year")
    max_year: Optional[int] = Field(None, description="Maximum release year")
    min_imdb_rating: Optional[float] = Field(None, description="Minimum IMDb rating")
    min_meta_score: Optional[int] = Field(None, description="Minimum Meta score")
    min_gross: Optional[float] = Field(None, description="Minimum gross")
    min_num_votes: Optional[int] = Field(None, description="Minimum number of votes")
    director: Optional[str] = Field(None, description="Director's name")
    series_title: Optional[str] = Field(None, description="Movie or series title")

class AnalysisSchema(BaseModel):
    operation: str = Field(..., description="Type of analysis", enum=["count", "sum", "average", "percentage"])
    field: str = Field(..., description="Field to analyze", enum=["Meta_score", "Gross", "IMDB_Rating"])

class StructuredQueryInput(BaseModel):
    filters: Optional[FilterSchema] = Field(None, description="Filters for querying movies")
    sort_by: Optional[str] = Field(None, description="Column to sort by")
    sort_order: Optional[str] = Field("desc", description="Sorting order ('asc' or 'desc')", enum=['asc', 'desc'])
    group_by: Optional[str] = Field(None, description="Column to group by")
    limit: Optional[int] = Field(None, description="Number of results to return")
    analysis: Optional[AnalysisSchema] = Field(None, description="Analysis operation to perform")
    input_data: Optional[List[Dict]] = Field(None, description="Input data from previous query results")

class VectorSearchInput(BaseModel):
    query: str = Field(..., description="Query for vector search")
    filters: Optional[FilterSchema] = Field(None, description="Filters for querying movies")
    limit: Optional[int] = Field(None, description="Number of results to return")
