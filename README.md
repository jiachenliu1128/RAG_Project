# RAG_Project

Self-RAG Retrieve Decision Implementation:
1. Rule-based: Use rule-based triggers (time-sensitive, explicitly mentioned use docs...)
2. Classifier-based: 
    - Use embeddings and classification layer.
    - Should we classify more from the models perspective (If the models have enough knowledge to answer)？ Or from the documents perspective (If the documents are relevant to the question)?
    - Need to generate new labels by ourselves, since it depends on the specific use case and available data.
    - How to generate labels? Use LLM to decide? User feedback?
3. LLM-based: Use the LLM itself to decide whether to retrieve information from documents or answer directly by prompt engineering.