from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains.summarize import load_summarize_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
# from langchain.chat_models import init_chat_model



load_dotenv()


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
# llm = init_chat_model("mistral-large-latest", model_provider="mistralai")

def split_summaries(selected_indices, docs):
    map_prompt = """
    You will receive a passage from a research paper enclosed in triple backticks (```).
Your task is to provide a comprehensive and informative summary of this section.

**Extract and format the following elements if present in the passage:**

*   **Title:**  [Title of the Section]
*   **Authors:**  [List all authors]
*   **Abstract:**  [Abstract of the Section]

**If these elements are not present, leave the corresponding sections blank.**

**Provide a detailed summary of the section, ensuring clarity and accuracy. The length of the summary should be appropriate to the content and complexity of the section. Aim for a minimum of three paragraphs for each summary.**

```{text}```
FULL SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    map_chain = load_summarize_chain(llm=llm,
                                chain_type="stuff",
                                prompt=map_prompt_template)
    selected_docs = [docs[doc] for doc in selected_indices]

    summary_list = []
    for i, doc in enumerate(selected_docs):
        chunk_summary = map_chain.invoke([doc])
        chunk_summary = chunk_summary["output_text"]
        summary_list.append(chunk_summary)
        # print(len(chunk_summary))
        # print (f"Summary #{i} (chunk #{selected_indices[i]}) - Preview: {chunk_summary[:250]} \n")
    return summary_list

def prepare_final_summary(summary_list):
    summaries = "\n".join(summary_list)
    summaries = Document(page_content=summaries)
    print (f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")

    combine_prompt = """
    You are an expert at synthesising information from multiple sources. You will be provided with a set of summaries from the same research paper, enclosed in triple backticks (```).  Your task is to integrate these summaries into a single, comprehensive, and coherent verbose summary.

Ensure your final summary includes the following sections, extracted from the individual summaries:

*   Title: [Title of the Research Paper]
*   Authors: [List all authors]
*   Abstract: [Abstract of the Research Paper]

**Synthesise the information from the individual summaries into a detailed and insightful 'Summary of the Paper' section.  Avoid repetition and focus on clarity and coherence. The length of the summary should be appropriate to the complexity and content of the paper, aiming for a minimum of 2000 words.**

```{text}```
VERBOSE SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    reduce_llm = ChatGroq(model_name="llama3-8b-8192")
    reduce_chain = load_summarize_chain(llm=reduce_llm,
                             chain_type="stuff",
                             prompt=combine_prompt_template,
                             verbose=False)
    # print("Runnning summaries....")
    output = reduce_chain.invoke([summaries])
    # print(output)
    return output
    # return "\n".join(output.split("\n"))
