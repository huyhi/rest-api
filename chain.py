from operator import itemgetter
from os import path
from typing import Sequence, Dict, List

from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableSequence, \
    ConfigurableField
from langchain_core.runnables import chain
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

from config import COLLECTION_NAME, DB_FOLDER_NAME
from prompt import RESPONSE_TEMPLATE, LITERATURE_REVIEW_PROMPT, \
    SUMMARIZE_PROMPT, REPHRASE_TEMPLATE, COHERE_RESPONSE_TEMPLATE


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


def serialize_history(request: List[Dict[str, str]]):
    chat_history = request.get("chat_history", [])
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def get_retriever() -> BaseRetriever:
    chroma_client = Chroma(
        collection_name=f'{COLLECTION_NAME}',
        embedding_function=OpenAIEmbeddings(),
        persist_directory=f'./{DB_FOLDER_NAME}'
    )
    return chroma_client.as_retriever(search_kwargs=dict(k=5))


def create_retriever_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")

    # return (
    #     RunnableLambda(itemgetter("question")).with_config(
    #         run_name="Itemgetter:question"
    #     )
    #     | retriever
    # ).with_config(run_name="RetrievalChainWithNoHistory")


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(llm, retriever).with_config(run_name="FindDocs")
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    default_response_synthesizer = prompt | llm

    cohere_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COHERE_RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    @chain
    def cohere_response_synthesizer(input: dict) -> RunnableSequence:
        return cohere_prompt | llm.bind(source_documents=input["docs"])

    response_synthesizer = (
        default_response_synthesizer.configurable_alternatives(
            ConfigurableField("llm"),
            default_key="openai_gpt_3_5_turbo",
            anthropic_claude_3_sonnet=default_response_synthesizer,
            fireworks_mixtral=default_response_synthesizer,
            google_gemini_pro=default_response_synthesizer,
            cohere_command=cohere_response_synthesizer,
        )
        | StrOutputParser()
    ).with_config(run_name="GenerateResponse")
    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )



basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    streaming=True,
)

answer_chain = create_chain(llm, get_retriever())


def chat_streaming_output(question: str, chat_history: []):
    return answer_chain.stream({
        'question': question,
        'chat_history': chat_history
    })


def summarize_output(prompt_data):
    return (i.content for i in llm.stream(SUMMARIZE_PROMPT.format(**prompt_data)))


def literature_review_output(prompt_data):
    return (i.content for i in llm.stream(LITERATURE_REVIEW_PROMPT.format(**prompt_data)))
