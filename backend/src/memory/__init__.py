__all__ = ["MongoMemoryManager", "ContextResolver", "InProcessSessionMemoryStore", "default_session_state"]


def __getattr__(name):
    if name == "MongoMemoryManager":
        from src.memory.mongodb_memory_layer import MongoMemoryManager
        return MongoMemoryManager
    if name == "ContextResolver":
        from src.memory.context_resolver import ContextResolver
        return ContextResolver
    if name == "InProcessSessionMemoryStore":
        from src.memory.memory_store import InProcessSessionMemoryStore
        return InProcessSessionMemoryStore
    if name == "default_session_state":
        from src.memory.session_state import default_session_state
        return default_session_state
    raise AttributeError(name)
