import os

import pytest
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from utils import load_test_data

#OPENAIAPIk

@pytest.mark.parametrize("get_data",indirect = True)
@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)
    lang_chain_llm = LangchainLLMWrapper(llm)
    return lang_chain_llm
