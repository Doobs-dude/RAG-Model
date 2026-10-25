## The Program Initation and File Structure

the notebooks files is used for the inital testing phase, where a sample of the database was used to embed, encode and query to save on computation costs

the git ignore file keeps the API key hidden and keeps environment variables in line

## The Program Objective

It is a RAG model which redefines the scope for researach for an LLM model with an academic database with over 100,000 entries

It does this by embedding the database, and using a combination of bi encoders and cross encoders. To make this embedding for effcient, the Program detects if the system contains a CUDA GPU (might not work for non nvidia GPUs)

A bi encoder is used to pick the 20 most relavent papers with relatively less accuracy compared to cross encoder, due to prompt and papers encoding discrepancy

A cross encoder is used over the 20 final retrieved documents, to effectively pick out the three most relevant papers

these three papers are pushed with the query itself to an LLM model, in two cycles

the initial cycle is for the LLM to make a proper assessment and retrieve the needed information for the user

the second cycle is for the LLM to review the inital output and refine it for hallucinations and false citations etc

## The Program Interface

The interface itself is coded on streamlit, as proposed by a AI engineer senior. the interface is locally hosted and is fast for retrieval.

## Setup configuration

```bash
git clone <your-repo-url>
cd rag-research-assistant
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

