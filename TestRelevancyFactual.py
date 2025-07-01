import pytest
from ragas import EvaluationDataset, evaluate, SingleTurnSample
from ragas.metrics import ResponseRelevancy, FactualCorrectness

from conftest import llm_wrapper
from utils import get_llm_response, load_test_data


@pytest.mark.parametrize("getData", load_test_data("relevancy_factual.json"), indirect=True)
@pytest.mark.asyncio
async def test_relevancey_factual(llm_wrapper, getData):
    metrics = [ResponseRelevancy(llm=llm_wrapper),
               FactualCorrectness(llm=llm_wrapper)]

    eval_dataset = EvaluationDataset([getData])
    results = evaluate(dataset=eval_dataset, metrics=metrics)
    print(results)


@pytest.fixture
def getData(request):
    test_data = request.param
    responseDict = get_llm_response(test_data)
    sample = SingleTurnSample(
              user_input=test_data["question"],
              response=responseDict["answer"],
              retrieved_contexts = [doc["page_content"] for doc in responseDict.get("retrieved_docs")],
              reference = test_data["reference"]
    )
    return sample







