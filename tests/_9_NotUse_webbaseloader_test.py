import pytest
from kwwutils import clock, printit
from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._9_NotUse_webbaseloader import (
    main,
)


@clock
@pytest.mark.parametrize(
    "url",
    [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "http://cnn.com",
        "http://foxnews.com",
        "http://msnbc.com",
    ],
)
def test_func(options, model, url):
    printit("options", options)
    printit("model", model)
    options["model"] = model
    options["url"] = url
    options["llm_type"] = "llm"
    options["question"] = "What did they say about fossil fuel projects?"
    response = main(**options)
    printit("response", response)
    printit("question", options["question"])
    printit("response len", len(response))
    sources = [r.metadata["source"] for r in response]
    printit("sources", sources)
    checks = all([s == url for s in sources])
    printit(f"{model} {url} checks", checks)
    assert checks
