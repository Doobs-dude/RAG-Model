import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

class RAGChain:
    def __init__(self, api_key: str = None, model_name: str = "nvidia/nemotron-3.5-lightning:free"):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    @staticmethod
    def format_context(results_df: pd.DataFrame) -> str:
        #Formats metadata DataFrame into a structured context block for the LLM.
        blocks = []
        for idx, (_, row) in enumerate(results_df.iterrows(), 1):
            blocks.append(
                f"Paper [{idx}]:\n"
                f"Title: {row['title']}\n"
                f"Authors: {row['authors']}\n"
                f"Category: {row['category']}\n"
                f"Summary: {row['summary']}\n"
            )
        return "\n".join(blocks)

    def run_2_pass_generation(self, query: str, search_results: pd.DataFrame) -> str:
        #Executes Pass 1 synthesis and Pass 2 citation/grounding refinement.
        context = self.format_context(search_results)
        
        # Pass 1: Synthesis
        pass1_prompt = f"""You are an academic assistant. Answer the user's question using ONLY the provided research paper contexts. Cite sources using [Paper 1], [Paper 2].
Context:
{context}

Question: {query}

Answer:"""

        pass1_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": pass1_prompt}],
            temperature=0.3
        )
        initial_answer = pass1_response.choices[0].message.content

        # Pass 2: Citation & Grounding Refinement
        pass2_prompt = f"""You are a strict editorial reviewer. Review the draft against the contexts.
1. Ensure all claims are strictly grounded in the summaries.
2. Ensure citations are accurate.
3. Remove hallucinations. Format in Markdown.

Contexts:
{context}

Draft:
{initial_answer}

Refined Answer:"""

        pass2_response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": pass2_prompt}],
            temperature=0.1
        )
        refined_answer = pass2_response.choices[0].message.content.strip()
        
        # Clean up: Strip markdown code block wrappers if the LLM still includes them
        if refined_answer.startswith("```markdown"):
            refined_answer = refined_answer[11:]
        elif refined_answer.startswith("```"):
            refined_answer = refined_answer[3:]
            
        if refined_answer.endswith("```"):
            refined_answer = refined_answer[:-3]
            
        return refined_answer.strip()
