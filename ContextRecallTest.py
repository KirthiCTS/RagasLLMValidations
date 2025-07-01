import os
import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, context_precision, LLMContextRecall


@pytest.mark.asyncio
async def test_context_recall():
    os.environ[
        "OPENAI_API_KEY"] = "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"

    question = "how many articles are there in the selenium webdriver python"
    llm = ChatOpenAI(model = "gpt-4.1", temperature = 0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    context_recall = LLMContextRecall(llm = lang_chain_llm)
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json = {"question": question,
                                          "chat_history":[]
                                         }).json()
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=[
            responseDict["retrieved_docs"][0]["page_content"],
            responseDict["retrieved_docs"][1]["page_content"],
            responseDict["retrieved_docs"][2]["page_content"]
        ],
        reference="23"
    )
    score = await context_recall.single_turn_ascore(sample)
    print(score)
    assert score > 0.7