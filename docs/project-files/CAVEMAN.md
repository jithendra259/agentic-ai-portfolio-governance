# Caveman Token Compression 🦴

Caveman is a token-saving optimization layer integrated into the Agentic Portfolio Governance system. It uses linguistically minimal prompting to reduce token usage and latency while maintaining logical integrity.

## How it Works

Caveman intercepts specific keywords in your messages to toggle compression levels. When active, it modifies the system instructions to force the LLM into a concise, high-density communication style.

### Compression Levels

| Level | Intensity | Goal | Use Case |
| :--- | :--- | :--- | :--- |
| **LITE** | Low | Remove fluff, keep grammar | Standard daily interaction |
| **FULL** | Medium | Omit articles/prepositions | Technical summaries, quick checks |
| **ULTRA** | Max | Keyword-only, telegraphed | **Internal infinite memory summarization** |

## Instructions & Commands

You can trigger Caveman mode using the following keywords in your chat:

- **Enable Caveman**: "enter caveman mode", "caveman mode", "/caveman"
- **Disable Caveman**: "exit caveman mode", "human mode", "/human"
- **Set Intensity**: Include "lite", "full", or "ultra" when enabling.
    - *Example*: "Enter caveman mode ultra"

## Internal Optimization

The system automatically uses **Ultra Caveman** style for the "Infinite Memory" summarization process. This ensures that the conversation history consumes the absolute minimum number of tokens, allowing for much larger context windows and lower operational costs.

> [!IMPORTANT]
> Caveman mode is automatically bypassed for security warnings, compliance errors, or irreversible financial actions to ensure full clarity during critical governance events.
