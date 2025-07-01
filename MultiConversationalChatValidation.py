import pytest
from ragas import MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import TopicAdherenceScore

from utils import load_test_data, get_llm_response


#@pytest.mark.asyncio("getData", load_test_data(""), indirect=True)
@pytest.mark.asyncio
async def test_topic_adherence(llm_wrapper, getData):
    topicScore = TopicAdherenceScore(llm=llm_wrapper)
    score =  topicScore.multi_turn_score(getData)
    print(score)
    assert score > 0.8


@pytest.fixture
def getData(request):
  #  test_data = request.param
  #  responseDict = get_llm_response(test_data)
    conversation = [
        HumanMessage(content="how many articles are there in selenium webdriver python?"),
        AIMessage(content="There are 23 articles in the course"),
        HumanMessage(content="how many downloadable resources are there in this course"),
        AIMessage(content="There are 9 downloadable resources in the course"),
    ]
    reference = ["""
    The AI should :
    1. Give results related to the selenium webdriver python course
    2. There are 23 articles and 9 downloadable resources in the course"""]
    sample = MultiTurnSample(user_input=conversation, reference_topics=reference)
    return sample
