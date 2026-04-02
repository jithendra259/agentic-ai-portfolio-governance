from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

class GenerativeExplainerAgent:
    """
    Agent 4: The Generative AI component. 
    Reads the deterministic CAG output (I_t, c_vector, weights) and generates 
    an advisory Human-in-the-Loop (HITL) Supervisory Governance Report.
    """
    def __init__(self):
        # Update this model name to match exactly what you pulled via `ollama run ...`
        self.llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0.5)
        
        # The strict template prevents hallucination and enforces ADVISORY ONLY rules
        self.prompt = PromptTemplate(
            input_variables=["date", "universe", "i_t", "lambda_t", "risky_stocks", "safe_stocks"],
            template="""
            You are an elite Quantitative Risk Officer AI for an institutional portfolio.
            A systemic risk assessment has been completed by the G-CVaR architecture.
            
            [DETERMINISTIC STATE DATA]
            Date: {date}
            Universe: {universe}
            Instability Index (I_t): {i_t} (Threshold is 0.5)
            Graph Penalty Applied (λ_t): {lambda_t}
            
            Top 3 Structurally Risky Stocks Identified (Graph Centrality): 
            {risky_stocks}
            
            Recommended Safe Haven Target Allocations: 
            {safe_stocks}
            
            [TASK]
            Write a brief, highly professional, 3-paragraph Supervisory Governance Report for the Portfolio Manager. 
            1. State the current market regime based purely on the I_t metric.
            2. Explain the structural graph risk (why the top risky stocks are flagged for exposure reduction).
            3. Present the optimized target weights for Human-in-the-Loop (HITL) review.
            
            CRITICAL RULE: You are an advisory system. Do NOT use words like "liquidated", "dumped", "execute", or "buy/sell". Discuss "allocation adjustments", "exposure reduction", and "target weights".
            """
        )

        # Create a modern LangChain execution chain
        self.chain = self.prompt | self.llm

    def execute(self, state):
        print("🧠 [Agent 4] Generative AI waking up. Drafting advisory HITL Governance Report...")
        
        # Extract the top 3 risky stocks from Agent 2's vector
        c_vec = state["c_vector"]
        top_risky = c_vec.nlargest(3).to_dict()
        risky_str = ", ".join([f"{k} (Risk Score: {v:.2f})" for k, v in top_risky.items()])
        
        # Extract the top 3 safe stocks from Agent 3's weights
        weights = state["optimal_weights"]
        top_safe = weights[weights > 0.0].nlargest(3).to_dict()
        safe_str = ", ".join([f"{k} (Target Weight: {v:.2%})" for k, v in top_safe.items()])

        # Execute the chain directly (LCEL)
        response = self.chain.invoke({
            "date": state["target_date"],
            "universe": state["universe_id"],
            "i_t": round(state["instability_index"], 4),
            "lambda_t": round(state["lambda_t"], 4),
            "risky_stocks": risky_str,
            "safe_stocks": safe_str
        })
        
        report = response.content
        
        print("   -> Governance Report drafted successfully.")
        return {"hitl_report": report}