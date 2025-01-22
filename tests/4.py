import pytest
from kwwutils import clock, printit
from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._4_ContextualCompressionRetriever import (
    main,
)


@pytest.mark.smoke
@clock
def test_a(options, few_models_arg):
    printit("options", options)
    printit("few_models_arg", few_models_arg)
    model = few_models_arg
    options["model"] = model
    options["llm_type"] = "llm"
    options["question"] = "What did they say about fossil fuel projects?"
    response = main(**options)
    printit("response", response)
    printit("question", options["question"])
    printit("response len", len(response))
    printit("response metadata", response[0].metadata)
    assert response[-1].metadata["source"].endswith("machinelearning-lecture01.pdf")


@pytest.mark.slow
@clock
def test_b(options, all_models_arg):
    printit("options", options)
    printit("all_models_arg", all_models_arg)
    model = all_models_arg
    options["model"] = model
    options["llm_type"] = "llm"
    options["question"] = "What did they say about fossil fuel projects?"
    response = main(**options)
    printit("question", options["question"])
    printit("response len", len(response))
    printit("response metadata", response[0].metadata)
    assert response[-1].metadata["source"].endswith("machinelearning-lecture01.pdf")
