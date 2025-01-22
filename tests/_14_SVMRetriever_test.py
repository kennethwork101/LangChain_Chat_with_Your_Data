import os
import pytest

from kwwutils import clock, printit
from uvprog2025.LangChain_Chat_with_Your_Data.src.langchain_chat_with_your_data._14_SVMRetriever import (
    main,
)


@pytest.mark.testme
@clock
def test_func(pytestconfig, options, model):
    package_root = pytestconfig.rootpath
    filepath = "data/data_all/pdf_files/kwong_resume_3.pdf"
    filename = os.path.join(package_root, filepath)
    printit("options", options)
    printit("model", model)
    printit("filename", filename)
    options["model"] = model
    options["filename"] = filename
    options["question"] = "What did they say about Kenneth?"
    response = main(**options)
    printit("response", response)
    printit("question", options["question"])
    assert "Kenneth" in response[0].page_content
