

import json
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from pandas import DataFrame
from langchain.schema import SystemMessage
from query_schema import *


class IMDBBot:
    def __init__(self, data_frame, vector_store):
        self.df = data_frame.copy()
        self.vectorstore = vector_store
        self.llm = ChatOpenAI(model="gpt-4o")
        
        # Enhanced tool descriptions for sub-query handling
        self.structured_query_tool = StructuredTool.from_function(
            func=self.execute_structured_query,
            name="structured_query",
            args_schema=StructuredQueryInput,
            description="""
            Query movies based on filters and perform analysis.
            function can perform filtering, aggregation, sorting, limiting amd analysis.
            Can process results from previous queries using input_data.
            Examples: 
            1. Filter movies then analyze: First call with filters, then pass results to second call for analysis
            2. Chain analyses: Use results from one query output as input for another query
            3. break down complex queries into smaller sub-queries
            4. call function on all the sub queries.
            5. keep the sub queries simple; DO NOT perform multiple operations in a single query.
            6. perform only ONE action per query.
            """
        )

        self.vector_search_tool = StructuredTool.from_function(
            func=self.execute_vector_search,
            name="vector_search",
            args_schema=VectorSearchInput,
            description="""Semantic search for movies. 
            Results can be passed to structured_query_tool for further analysis.
            This tool can also perform basic filtering of the data.
            It can be used to answer questions about the movie plot.
            Examples:
            1. Summarize the plots of the movies
            2. Which movies have a business theme
            3. Summarize the plot of crime movies by Steven Spielberg
            """
        )
        
        self.tools = [self.structured_query_tool, self.vector_search_tool]
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Central state to store intermediate data
        self.intermediate_data = None 
        
        
        # Customize agent prompt to handle sub-queries in CoT format
        prompt = """
        Use the following format:
 
        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question
       

        Note: When using structured query tool, make sure to call the tool for only ONE type of QUERY.
        The call should be for only one of filtering, grouping, analysis, sorting
        If a a query is complex, break it into multiple steps, where each step will have only ONE type of query. 

        """
        system_message = SystemMessage(content="""You are an expert at breaking down complex movie queries into sub-queries. 
        For complex requests:
        1. Break down the user query into smaller sub-queries
        2. Use the appropriate tool for each sub-query
        3. Pass results from one query as input to the next using the input_data parameter
        4. Combine results as needed
        5. Break down structured queries into multiple smaller sub-queries
        6. simplify the queries as much as possible.
        7. call tool on simple queries. 
        8. perform only ONE action per query.
        
        Keep track of intermediate results and use them in subsequent queries.""")
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            prompt = prompt,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,  # Increased for complex query chains
            agent_kwargs={
                "system_message": system_message,
            }
        )

    
    
    def process_query_result(self, result: Union[DataFrame, Dict]) -> List[Dict]:
        """Convert query results to a format suitable for chaining"""
        if isinstance(result, DataFrame):
            return result.to_dict('records')  # DataFrame to list of dictionaries
        elif isinstance(result, dict):
            return [result]  # Single dictionary to list containing the dictionary
        return result

    
    
    
    
    def execute_vector_search(self, query: str, filters: Optional[FilterSchema] = None, limit: Optional[int] = None):
        query = query.lower()
        chroma_filters = {}
        if filters:
            if filters.min_year:
                chroma_filters['Released_Year'] = {"$gte": int(filters.min_year)}
            if filters.max_year:
                chroma_filters['Released_Year'] = {"$lte": int(filters.max_year)}
            if filters.min_num_votes:
                chroma_filters['No_of_Votes'] = {"$gte": int(filters.min_num_votes)}
            if filters.min_imdb_rating:
                chroma_filters['IMDB_Rating'] = {"$gte": float(filters.min_imdb_rating)}
            if filters.min_meta_score:
                chroma_filters['Meta_score'] = {"$gte": int(filters.min_meta_score)}
            if filters.director:
                chroma_filters['Director'] = {"$eq": filters.director.lower()}            

        if not chroma_filters:
            results_with_scores = self.vectorstore.similarity_search_with_score(query, k=limit or 10)
            results = [doc for doc, _ in results_with_scores]  # Extract documents
        else:
            results = self.vectorstore.similarity_search(
                query=query,
                filter=chroma_filters,
                k=limit or 10
            )
     
        if filters.genre:
            results = [r for r in results if str(filters.genre).lower() in r.metadata.get("Genre", "")]
        
        titles = [doc.metadata['Series_Title'].lower() for doc in results]
        filtered_df = self.df[self.df['Series_Title'].isin(titles)]
        self.intermediate_data = self.process_query_result(filtered_df) 
        
        return self.intermediate_data

    
    
    
    
    
    
    def apply_filters(self, df, filters: FilterSchema):
        """ Apply filtering conditions on the dataset """
        if not filters:
            return df  # No filters, return original dataset

        mask = pd.Series(True, index=df.index)  # Initialize a mask with all True values

        if filters.genre:
            mask &= df['Genre'].apply(lambda g: filters.genre.lower() in g)
        if filters.min_year:
            mask &= df['Released_Year'] >= filters.min_year
        if filters.max_year:
            mask &= df['Released_Year'] <= filters.max_year
        if filters.min_imdb_rating:
            mask &= df['IMDB_Rating'] >= filters.min_imdb_rating
        if filters.min_meta_score:
            mask &= df['Meta_score'] >= filters.min_meta_score
        if filters.min_gross:
            mask &= df['Gross'] >= filters.min_gross
        if filters.min_num_votes:
            mask &= df['No_of_Votes'] >= filters.min_num_votes
        if filters.director:
            mask &= df['Director'] == filters.director.lower()
        if filters.series_title:
            mask &= df['Series_Title'] == filters.series_title.lower()

        return df[mask]

  


    def apply_grouping(self, df, group_by: Optional[str]):
        """ Groups the dataset if `group_by` is specified """
        if group_by and group_by in df.columns:
            return df.groupby(group_by, as_index=False)  # Keeps group_by column in result
        return df  # No grouping applied


    def apply_analysis(self, df, analysis: Optional[AnalysisSchema], group_by: bool):
        """ Perform aggregation (sum, count, avg, percentage) on dataset """
        if not analysis:
            return df  # No analysis requested

        field = analysis.field
        operation = analysis.operation

        if operation == "count":
            df = df[[field]].count().reset_index() if group_by else pd.DataFrame({field: [df[field].count()]})
    
        elif operation == "sum":
            df = df[field].sum().reset_index() if group_by else pd.DataFrame({field: [df[field].sum()]})

        elif operation == "average":
            df = df[field].mean().reset_index() if group_by else pd.DataFrame({field: [df[field].mean()]})

        elif operation == "percentage":
            total = df[field].sum()
            if total == 0:
                df["percentage"] = 0  # Avoid division by zero
            else:
                df["percentage"] = (df[field] / total) * 100
            df = df["percentage"].sum().reset_index() if group_by else pd.DataFrame({field: [df["percentage"].sum()]})

        return df


    def apply_sorting_and_limiting(self, df, sort_by, sort_order):
        """ Sorts & Limits the dataset """
        if sort_by and sort_by in df.columns:
            df = df.sort_values(by=sort_by, ascending= sort_order == "asc")
        
        return df

    def execute_structured_query(self, 
                               filters: Optional[FilterSchema] = None,
                               sort_by: Optional[str] = None,
                               sort_order: str = "desc",
                               group_by: Optional[str] = None,
                               limit: Optional[int] = None,
                               analysis: Optional[AnalysisSchema] = None,
                               input_data: Optional[List[Dict]] = None):
        df = pd.DataFrame(input_data) if input_data else self.df.copy()

        # Step 1: Apply Filters
        if filters:
          df = self.apply_filters(df, filters)

        # Step 2: Apply Grouping
        if group_by:
          df = self.apply_grouping(df, group_by)

        # Step 3: Perform Analysis (If specified)
        if analysis:
            df = self.apply_analysis(df, analysis, True if group_by else False)

        # Step 4: Apply Sorting
        df = self.apply_sorting_and_limiting(df, sort_by, sort_order)
      
        #step 5: applying limit

        if limit:
            df = df.head(limit)
        
        self.intermediate_data = self.process_query_result(df) 
        
        return df
       

    def query(self, user_query: str):
        """Process user query using the agent with sub-query support"""
        
        self.intermediate_data = None
        response = self.agent.run(user_query)
        return response



def intialize_agent(data_frame, vector_store):
    bot = IMDBBot(data_frame, vector_store)
    return bot

        
