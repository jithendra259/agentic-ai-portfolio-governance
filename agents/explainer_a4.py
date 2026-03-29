from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate  # 🚨 Updated path!

class GenerativeExplainerAgent:
    """
    Agent 4: The Generative AI component (100% Local via Ollama). 
    Reads the deterministic CAG output (I_t, c_vector, weights) and generates 
    a Human-in-the-Loop (HITL) Supervisory Governance Report.
    """
    def __init__(self):
        # 🚨 Pointing exactly to your local Qwen 2.5 model
        self.llm = ChatOllama(model="qwen2.5:7b", temperature=0.2)
        
        # The strict template prevents hallucination (CAG grounding)
        self.prompt = PromptTemplate(
            input_variables=["date", "universe", "i_t", "lambda_t", "risky_stocks", "safe_stocks"],
            template="""
            You are a Quantitative Risk Officer AI for an institutional portfolio.
            A systemic risk event has been detected by the G-CVaR architecture.
            
            [DETERMINISTIC STATE DATA]
            Date: {date}
            Universe: {universe}
            Instability Index (I_t): {i_t} (Threshold is 0.5)
            Graph Penalty Applied (λ_t): {lambda_t}
            
            Top 3 Structurally Risky Stocks Liquidated (Graph Centrality): 
            {risky_stocks}
            
            Capital Reallocated to Safe Haven Assets: 
            {safe_stocks}
            
            [TASK]
            Write a brief, highly professional, 3-paragraph Supervisory Governance Report for the Human Portfolio Manager. 
            1. State the detection of the regime shift based purely on the I_t metric.
            2. Explain the structural graph liquidation (why the risky stocks were dumped).
            3. Request Human-in-the-Loop (HITL) approval to execute the convex optimization trades.
            """
        )

    def execute(self, state):
        print(f"🧠 [Agent 4] Generative AI waking up (Local Ollama: Qwen2.5). Drafting HITL Governance Report...")
        
        # Extract the top 3 risky stocks from Agent 2's vector
        c_vec = state["c_vector"]
        top_risky = c_vec.nlargest(3).to_dict()
        risky_str = ", ".join([f"{k} (Risk: {v:.2f})" for k, v in top_risky.items()])
        
        # Extract the top 3 safe stocks from Agent 3's weights
        weights = state["optimal_weights"]
        top_safe = weights[weights > 0.0].nlargest(3).to_dict()
        safe_str = ", ".join([f"{k} (Weight: {v:.2%})" for k, v in top_safe.items()])

        # Format the prompt with the strict deterministic data (CAG)
        formatted_prompt = self.prompt.format(
            date=state["target_date"],
            universe=state["universe_id"],
            i_t=round(state["instability_index"], 4),
            lambda_t=round(state["lambda_t"], 4),
            risky_stocks=risky_str,
            safe_stocks=safe_str
        )

        # Generate the report natively on your local machine
        response = self.llm.invoke(formatted_prompt)
        report = response.content
        
        print("   -> Governance Report drafted successfully.")
        return {"hitl_report": report}