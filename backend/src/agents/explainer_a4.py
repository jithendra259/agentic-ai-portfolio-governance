import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama


class GenerativeExplainerAgent:
    """
    Agent 4: The Generative AI component.
    Reads the deterministic graph-risk output (I_t, c_vector, weights) and generates
    an advisory Human-in-the-Loop (HITL) supervisory governance report.
    """

    def __init__(self):

        model_name = (os.getenv("PORTFOLIO_OLLAMA_MODEL") or "gpt-oss:120b-cloud").strip()
        ollama_base_url = (os.getenv("PORTFOLIO_OLLAMA_BASE_URL") or os.getenv("OLLAMA_BASE_URL") or "").strip() or None
        if model_name.startswith("ashna") or model_name == "ashnaai" or (os.getenv("ASHNA_API_KEY") and model_name == "gpt-oss:120b-cloud"):
            active_model = "ashnaai" if model_name == "gpt-oss:120b-cloud" else model_name
            api_key = os.getenv("ASHNA_API_KEY")
            base_url = os.getenv("ASHNA_BASE_URL")
            
            if api_key and base_url:
                # Normalize base URL: remove /api suffix if present (should be just /v1)
                base_url = base_url.rstrip("/")
                if base_url.endswith("/api"):
                    base_url = base_url[:-4]  # Remove /api suffix
                if not base_url.endswith("/v1"):
                    base_url = base_url + "/v1"
                
                try:
                    from langchain_openai import ChatOpenAI
                    self.llm = ChatOpenAI(
                        model=active_model,
                        temperature=0.5,
                        api_key=api_key,
                        base_url=base_url,
                        timeout=30,
                        max_retries=2,
                    )
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to initialize Ashna API in explainer_a4: {e}. Using Ollama fallback.")
                    self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.5)
            else:
                kwargs = {"model": "gpt-oss:120b-cloud", "temperature": 0.5}
                if ollama_base_url:
                    kwargs["base_url"] = ollama_base_url.rstrip("/")
                self.llm = ChatOllama(**kwargs)
        else:
            kwargs = {"model": model_name, "temperature": 0.5}
            if ollama_base_url:
                kwargs["base_url"] = ollama_base_url.rstrip("/")
            self.llm = ChatOllama(**kwargs)

        self.prompt = PromptTemplate(
            input_variables=[
                "date",
                "universe",
                "i_t",
                "lambda_t",
                "governance_flags",
                "risky_stocks",
                "safe_stocks",
            ],
            template="""
            You are an elite Quantitative Risk Officer AI for an institutional portfolio.
            A systemic risk assessment has been completed by the G-CVaR architecture.

            [DETERMINISTIC STATE DATA]
            Date: {date}
            Universe: {universe}
            Instability Index (I_t): {i_t} (Crisis/HITL threshold is 0.85)
            Graph Penalty Applied (lambda_t): {lambda_t}
            Governance Trigger(s): {governance_flags}

            Top 3 Structurally Risky Stocks Identified (Institutional Graph Centrality):
            {risky_stocks}

            Recommended Safe Haven Target Allocations:
            {safe_stocks}

            [TASK]
            Write a brief, highly professional, 3-paragraph Supervisory Governance Report for the Portfolio Manager.
            1. State the current market regime based purely on the I_t metric.
            2. Explain the structural graph risk from overlapping institutional ownership.
            3. Present the optimized target weights for Human-in-the-Loop (HITL) review.

            CRITICAL RULE: You are an advisory system. Do NOT use words like "liquidated", "dumped", "execute", or "buy/sell". Discuss "allocation adjustments", "exposure reduction", and "target weights".
            """,
        )
        self.chain = self.prompt | self.llm

    def execute(self, state):
        print("[Agent 4] Generative AI waking up. Drafting advisory HITL governance report...")

        c_vec = state["c_vector"]
        top_risky = c_vec.nlargest(3).to_dict()
        risky_str = ", ".join([f"{ticker} (Risk Score: {score:.2f})" for ticker, score in top_risky.items()])

        weights = state["optimal_weights"]
        top_safe = weights[weights > 0.0].nlargest(3).to_dict()
        safe_str = ", ".join([f"{ticker} (Target Weight: {weight:.2%})" for ticker, weight in top_safe.items()])

        governance_flags = ", ".join(state.get("hitl_reasons", [])) or "manual review requested"

        response = self.chain.invoke(
            {
                "date": state["target_date"],
                "universe": state["universe_id"],
                "i_t": round(state["instability_index"], 4),
                "lambda_t": round(state["lambda_t"], 4),
                "governance_flags": governance_flags,
                "risky_stocks": risky_str,
                "safe_stocks": safe_str,
            }
        )

        report = response.content
        print("   -> Governance report drafted successfully.")
        return {"hitl_report": report}
