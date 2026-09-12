"""Central configuration: project paths, provider/model settings and directory constants (loads .env if present)."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = PROJECT_ROOT / "code"
SRC_DIR = CODE_DIR / "src"

ENV_PATH = CODE_DIR / ".env"

if load_dotenv is not None and ENV_PATH.exists():
    load_dotenv(ENV_PATH, override=False)

DATA_ROOT = PROJECT_ROOT / "data"
DATA_DIR = DATA_ROOT / "processed"
RESULTS_DIR = Path(os.getenv("FAMILY_MATRIP_RESULTS_DIR", str(PROJECT_ROOT / "results")))
CACHE_DIR = DATA_ROOT / "cache"
ONLINE_SEARCH_CACHE_DIR = CACHE_DIR / "online_grounding"
RAW_SEARCH_CACHE_DIR = ONLINE_SEARCH_CACHE_DIR / "raw_search"
STRUCTURED_GROUNDING_CACHE_DIR = ONLINE_SEARCH_CACHE_DIR / "structured"

PERSONA_CSV = DATA_DIR / "FamilyTravel_Mini.csv"

MODULE1_DIR = RESULTS_DIR / "module_1"
MODULE2_DIR = RESULTS_DIR / "module_2"
MODULE3_DIR = RESULTS_DIR / "module_3"
MODULE4_DIR = RESULTS_DIR / "module_4"
MODULE5_DIR = RESULTS_DIR / "module_5"
MODULE6_DIR = RESULTS_DIR / "module_6"
MODULE7_DIR = RESULTS_DIR / "module_7"
FINAL_ITINERARY_TXT_DIR = RESULTS_DIR / "final_itinerary_txt"
ABLATION_RESULTS_DIR = RESULTS_DIR / "ablation"

MEMORY_DIR = MODULE2_DIR / "agent_memories"
MODULE4_MEMORY_DIR = MODULE4_DIR / "agent_memories"
MODULE7_TEMP_DIR = MODULE7_DIR / "module7_temp"

PERSONA_AGENTS_JSON = MODULE1_DIR / "persona_agents.json"
INDIVIDUAL_PROPOSALS_CSV = MODULE2_DIR / "individual_agent_proposals.csv"
INDIVIDUAL_PROPOSALS_JSON = MODULE2_DIR / "individual_agent_proposals.json"
CONFLICT_RESULTS_CSV = MODULE3_DIR / "conflict_detection_results.csv"
CONFLICT_RESULTS_JSON = MODULE3_DIR / "conflict_detection_results.json"
NEGOTIATION_RESULTS_CSV = MODULE4_DIR / "negotiation_protocol_results.csv"
NEGOTIATION_RESULTS_JSON = MODULE4_DIR / "negotiation_protocol_results.json"
GENERATED_PLANS_CSV = MODULE5_DIR / "generated_plans.csv"
CONSENSUS_ITINERARY_JSON = MODULE5_DIR / "consensus_itinerary_results.json"
EVALUATION_MEMBER_CSV = MODULE6_DIR / "evaluation_member_level.csv"
EVALUATION_PLAN_CSV = MODULE6_DIR / "evaluation_plan_level.csv"
EVALUATION_REPORT_JSON = MODULE6_DIR / "evaluation_report.json"
FINAL_REVISED_PLANS_CSV = MODULE7_DIR / "generated_plans_feedback_final.csv"
REVISION_RESULTS_JSON = MODULE7_DIR / "feedback_revision_results.json"
REVISED_EVALUATION_MEMBER_CSV = MODULE7_DIR / "evaluation_member_level_feedback.csv"
REVISED_EVALUATION_PLAN_CSV = MODULE7_DIR / "evaluation_plan_level_feedback.csv"
REVISED_EVALUATION_REPORT_JSON = MODULE7_DIR / "evaluation_report_feedback.json"
BACKTRACKING_FEEDBACK_JSON = MODULE7_DIR / "backtracking_to_module2_feedback.json"
FINAL_ITINERARY_ALL_TXT = FINAL_ITINERARY_TXT_DIR / "all_families_final_itinerary_input.txt"

RESULT_DIRECTORIES = [
    RESULTS_DIR,
    MODULE1_DIR,
    MODULE2_DIR,
    MODULE3_DIR,
    MODULE4_DIR,
    MODULE5_DIR,
    MODULE6_DIR,
    MODULE7_DIR,
    FINAL_ITINERARY_TXT_DIR,
    MEMORY_DIR,
    MODULE4_MEMORY_DIR,
    MODULE7_TEMP_DIR,
    CACHE_DIR,
    ONLINE_SEARCH_CACHE_DIR,
    RAW_SEARCH_CACHE_DIR,
    STRUCTURED_GROUNDING_CACHE_DIR,
]

# Main pipeline switches are defined in code/scripts/main_pipeline.py.

# Generation and judge provider settings.
GENERATION_LLM_PROVIDER = os.getenv("GENERATION_LLM_PROVIDER", "openrouter")
GENERATION_MODEL_NAME = os.getenv(
    "OPENROUTER_GENERATION_MODEL",
    os.getenv("OPENROUTER_MODEL_NAME", os.getenv("GENERATION_MODEL_NAME", "openai/gpt-4o-mini")),
)

LLM_JUDGE_PROVIDER = os.getenv("LLM_JUDGE_PROVIDER", "deepseek")
LLM_JUDGE_MODEL_NAME = os.getenv(
    "DEEPSEEK_LLM_JUDGE_MODEL",
    os.getenv("DEEPSEEK_MODEL_NAME", os.getenv("LLM_JUDGE_MODEL_NAME", "deepseek-chat")),
)
