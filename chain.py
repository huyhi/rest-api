from operator import itemgetter
from os import path
from typing import Sequence

from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough, ConfigurableField
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from config import COLLECTION_NAME, DB_FOLDER_NAME
from prompt import RESPONSE_TEMPLATE, LITERATURE_REVIEW_PROMPT, \
    SUMMARIZE_PROMPT


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    print(type(docs), docs)
    for i, doc in enumerate(docs):
        doc_string = (f"<doc id='{i}'>"
                      f"<Title>{doc.metadata.get('Title', '')}/<Title>"
                      f"<Abstract>{doc.metadata.get('Abstract', '')}/<Abstract>"
                      f"<Authors>{doc.metadata.get('Authors', '')}/<Authors>"
                      f"<Keywords>{doc.metadata.get('Keywords', '')}/<Keywords>"
                      f"<Source>{doc.metadata.get('Source', '')}/<Source>"
                      f"<Year>{doc.metadata.get('Year', '')}/<Year>"
                      f"</doc>")
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def get_retriever() -> BaseRetriever:
    chroma_client = Chroma(
        collection_name=f'{COLLECTION_NAME}',
        embedding_function=OpenAIEmbeddings(),
        persist_directory=f'./{DB_FOLDER_NAME}'
    )
    return chroma_client.as_retriever(search_kwargs=dict(k=5))


def create_retriever_chain(retriever: BaseRetriever) -> Runnable:
    return (
        RunnableLambda(itemgetter("question")).with_config(
            run_name="Itemgetter:question"
        )
        | retriever
    ).with_config(run_name="RetrievalChainWithNoHistory")


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(retriever).with_config(run_name="FindDocs")
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            ("human", "{question}"),
        ]
    )
    default_response_synthesizer = prompt | llm

    response_synthesizer = (
        default_response_synthesizer.configurable_alternatives(
            ConfigurableField("llm"),
            default_key="openai_gpt_3_5_turbo",
            anthropic_claude_3_sonnet=default_response_synthesizer,
            fireworks_mixtral=default_response_synthesizer,
            google_gemini_pro=default_response_synthesizer,
        )
        | StrOutputParser()
    ).with_config(run_name="GenerateResponse")
    return (
        context
        | response_synthesizer
    )


basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))

llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0,
    streaming=True,
)

answer_chain = create_chain(llm, get_retriever())


def chat_streaming_output(question: str):
    return answer_chain.stream({'question': question})


def summarize_output(prompt_data):
    return (i.content for i in llm.stream(SUMMARIZE_PROMPT.format(**prompt_data)))


def literature_review_output(prompt_data):
    return (i.content for i in llm.stream(LITERATURE_REVIEW_PROMPT.format(**prompt_data)))
