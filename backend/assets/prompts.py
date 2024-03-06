MAKE_QUERY_PROMPT = """
Create a search engine query that retrieves documents similar to the following user query: {query}

Format your response as a JSON object following this template: {{"search_query": "natural language search query for search engine", "keywords": "keyword1, keyword2, etc."}}

Your search query:
"""

RAG_PROMPT = """
REFERENCES are provided below as a knowledge source. 

================================
REFERENCES: 
{references}
================================

Respond to the user QUERY below using the supplied REFERENCES. 

================================
QUERY: 
{query}
================================

Keep your response objective and factual and use an informational report-style tone. Do not include content outside of the supplied REFERENCES. Cite ANY and ALL sentences that come from text in the REFERENCES using "[docN]" notation. For example, if a given sentence uses the first document in CITATIONS, add "[doc1]" to the end of that sentence. Likewise, use "[doc2]" if citing the second document in CITATIONS. DO NOT use any citation convention other than "[docN]".

Your response:
"""