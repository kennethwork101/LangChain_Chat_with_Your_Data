from kwwutils import clock, printit
from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._12_ConversationalRetrievalChain import (
    main,
)


@clock
def test_func(options, model):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["llm_type"] = "chat"
    options["question"] = "Why are those prerequesites needed?"
    response = main(**options)
    printit("response", response)
    printit("response keys", response.keys())
    assert sorted(response.keys()) == ["answer", "chat_history", "question"]
    assert response["question"] == options["question"]
    assert "prerequisites" in response["answer"]
