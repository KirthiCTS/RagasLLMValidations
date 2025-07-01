import os

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from utils import load_test_data

os.environ[
    "OPENAI_API_KEY"] = "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"


@pytest.mark.parametrize("get_data",indirect = True)
@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    return lang_chain_llm
