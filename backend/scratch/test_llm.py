import time
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

print("Initializing ChatOllama with gpt-oss:120b-cloud...")
llm = ChatOllama(
    model="gpt-oss:120b-cloud",
    temperature=0.2,
    num_ctx=8192,
    keep_alive="10m"
)

print("Invoking model...")
start = time.time()
try:
    res = llm.invoke([HumanMessage(content="Hello, answer in 5 words.")])
    print("Response:", res.content)
    print("Time taken:", time.time() - start, "seconds")
except Exception as e:
    print("Failed in", time.time() - start, "seconds. Error:", e)
