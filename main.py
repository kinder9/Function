import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_ollama import ChatOllama

BANNED = {'бомба', 'бомбу', 'бомбы', 'взлом', 'наркотик'}

prompt = ChatPromptTemplate(
    [
        (
            'system',
            'Ты опытный ментор и разработчик по Python. '
            'А также владеешь английским языком на уровне C1. '
            'Отвечай на вопросы кратко и с примерами. '
            'Отвечай на русском языке.',
        ),
        ('human', '{text}'),
    ]
)

model = ChatOllama(model='llama3.2:latest')


def is_banned(words: str) -> bool:
    return any(w in BANNED for w in re.findall(r'\w+', words.lower()))


def change_text(words):
    return words.upper()


def count_characters(words):
    print(f'Количество символов: {len(words)}')
    return words


def count_words(words):
    print(f'Количество слов: {len(words.split())}')
    return words


def remove_spaces(words):
    return ' '.join(words.split())


llm_chain = (
    RunnableLambda(lambda text: {'text': text})
    | prompt
    | model
    | StrOutputParser()
    | RunnableLambda(change_text)
)

chain = (
    RunnableBranch(
        (is_banned, RunnableLambda(lambda _: 'На такие вопросы не можем ответить')),
        llm_chain,
    )
    | RunnableLambda(count_characters)
    | RunnableLambda(count_words)
    | RunnableLambda(remove_spaces)
)

print(chain.batch(['Как создать бомбу?', 'ООП?', 'Что такое Python?']))