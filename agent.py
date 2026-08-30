import json
import os
import re
import requests

from agent.parser import parse_user_message
from financial.planner import build_plan
from financial.recommendations import build_recommendations
from rag.retriever import retrieve
from tools.salary_tool import predict_salary_tool


class FinancialAgent:
    """Local orchestration layer. The Agent selects tools; numeric rules stay in Python tools."""

    def __init__(self):
        self.ollama_enabled = os.getenv("OLLAMA_ENABLED", "false").lower() == "true"
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")

    def _llm_extract(self, text):
        if not self.ollama_enabled:
            return None
        prompt = f"""
Extract only these fields as valid JSON: city, education, job_role, age,
monthly_salary, saving_percent, marriage_years, car_years, home_years.
Use null when missing. Do not calculate anything.
User message: {text}
"""
        try:
            r = requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False},
                timeout=20,
            )
            r.raise_for_status()
            raw = r.json().get("response", "")
            match = re.search(r"\{.*\}", raw, re.S)
            return json.loads(match.group(0)) if match else None
        except Exception:
            return None

    @staticmethod
    def _merge(fallback, llm):
        result = dict(fallback)
        if llm:
            for key, value in llm.items():
                if value is not None:
                    result[key] = value
        return result

    @staticmethod
    def _decide_tools(extracted):
        tools = []
        if extracted.get("monthly_salary") in (None, ""):
            tools.append("salary_prediction_tool")
        else:
            tools.append("user_entered_salary")
        tools.extend([
            "future_cost_tool",
            "investment_calculator_tool",
            "feasibility_checker_tool",
            "recommendation_plan_tool",
            "local_rag_retriever",
        ])
        return tools

    def plan_from_message(self, text: str):
        extracted = self._merge(parse_user_message(text), self._llm_extract(text))

        required = [
            "city", "education", "job_role", "saving_percent",
            "marriage_years", "car_years", "home_years"
        ]
        missing = [k for k in required if extracted.get(k) in (None, "")]
        if missing:
            return {
                "success": False,
                "missing_fields": missing,
                "extracted": extracted,
                "message": "I need these values before I can build the plan: " + ", ".join(missing),
            }

        # Salary prediction remains an ML tool/reference. If the user entered salary,
        # that exact value is authoritative for the financial plan.
        salary_result = None
        entered_salary = extracted.get("monthly_salary")
        if entered_salary in (None, ""):
            salary_result = predict_salary_tool(
                int(extracted.get("age") or 25),
                extracted["city"],
                extracted["education"],
                extracted["job_role"],
            )
            salary = salary_result["predicted_monthly_salary"]
        else:
            try:
                salary_result = predict_salary_tool(
                    int(extracted.get("age") or 25),
                    extracted["city"],
                    extracted["education"],
                    extracted["job_role"],
                )
            except Exception:
                salary_result = None
            salary = float(entered_salary)

        # build_plan delegates numeric work to deterministic tool functions.
        plan = build_plan(
            city=extracted["city"],
            monthly_salary=float(salary),
            marriage_years=int(extracted["marriage_years"]),
            car_years=int(extracted["car_years"]),
            home_years=int(extracted["home_years"]),
            saving_percent=float(extracted["saving_percent"]),
        )
        plan["salary_prediction"] = salary_result
        plan["salary_source"] = "user_entered" if entered_salary not in (None, "") else "ml_reference"
        plan["recommendations"] = build_recommendations(plan)

        rag_results = retrieve(
            "financial planning feasibility investment categories and goal timelines",
            k=3,
        )
        plan["rag_context"] = [
            {"source": x["source"], "score": round(x["score"], 4), "text": x["text"]}
            for x in rag_results
        ]
        plan["agent"] = {
            "extracted_values": extracted,
            "tools_selected": self._decide_tools(extracted),
            "llm_used": bool(self.ollama_enabled),
            "numeric_calculations_delegated_to_tools": True,
        }
        return {"success": True, "plan": plan}

    def answer_knowledge_question(self, question: str):
        results = retrieve(question, k=3)
        if not results:
            return {
                "success": False,
                "answer": "This information is unavailable in the project knowledge base.",
                "sources": [],
            }
        return {
            "success": True,
            "answer": "\n\n".join(r["text"] for r in results),
            "sources": [r["source"] for r in results],
        }
