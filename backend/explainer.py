import os
import json
import logging
import re

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Read a positive integer from environment, with safe fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("must be > 0")
        return parsed
    except ValueError:
        logger.warning("Invalid value for %s=%r; using default %d", name, value, default)
        return default


def _is_rate_limited(message: str) -> bool:
    """Return True when Groq reports a token/rate-limit response."""
    msg = message.lower()
    return "rate limit" in msg or "rate_limit_exceeded" in msg


GROQ_EXPLAIN_MODEL = os.getenv("GROQ_EXPLAIN_MODEL", "llama-3.3-70b-versatile")
GROQ_QUERY_MODEL = os.getenv("GROQ_QUERY_MODEL", GROQ_EXPLAIN_MODEL)
GROQ_BATCH_SIZE = _env_int("GROQ_BATCH_SIZE", 10)
GROQ_MAX_TOKENS = _env_int("GROQ_EXPLAIN_MAX_TOKENS", 1800)
GROQ_QUERY_MAX_TOKENS = _env_int("GROQ_QUERY_MAX_TOKENS", 220)
GROQ_MAX_EXPLAIN_ROWS = _env_int("GROQ_MAX_EXPLAIN_ROWS", 120)
GROQ_MAX_QUERY_CONTEXT_ROWS = _env_int("GROQ_MAX_QUERY_CONTEXT_ROWS", 8)

EXPLAIN_SYSTEM_PROMPT = (
    "You are an industrial sensor analyst specializing in manufacturing equipment "
    "health monitoring. You receive sensor readings from CNC machines and provide "
    "concise diagnostic insights."
)

QUERY_SYSTEM_PROMPT = (
    "You are a manufacturing data analyst. Answer questions about sensor anomaly "
    "data concisely and accurately based only on the provided data context."
)


class GroqExplainer:
    def __init__(self) -> None:
        """Initialize the Groq client with the API key from environment.

        Raises:
            RuntimeError: If GROQ_API_KEY is not set in the environment.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set in environment")
        self.client = Groq(api_key=api_key)

    def explain_anomalies(self, anomaly_rows: list) -> list[dict]:
        """Generate LLM explanations for anomaly rows in batches.

        Args:
            anomaly_rows: List of anomaly row dicts with sensor values.

        Returns:
            List of dicts with row_id and explanation for each anomaly.
        """
        rows_to_explain = anomaly_rows[:GROQ_MAX_EXPLAIN_ROWS]
        skipped_rows = anomaly_rows[GROQ_MAX_EXPLAIN_ROWS:]

        batches = [
            rows_to_explain[i : i + GROQ_BATCH_SIZE]
            for i in range(0, len(rows_to_explain), GROQ_BATCH_SIZE)
        ]

        all_explanations = []

        for batch_index, batch in enumerate(batches):
            try:
                formatted_rows = "\n".join(
                    f"Row {r['row_id']}: {r['air_temp']} | {r['process_temp']} | "
                    f"{r['rotational_speed']} | {r['torque']} | {r['tool_wear']}"
                    for r in batch
                )

                user_prompt = (
                    f"The following {len(batch)} sensor readings were flagged as "
                    f"anomalous by an Isolation Forest model on manufacturing equipment data.\n\n"
                    f"Columns: Air Temp (K) | Process Temp (K) | Rotational Speed (rpm) | "
                    f"Torque (Nm) | Tool Wear (min)\n\n"
                    f"Readings:\n{formatted_rows}\n\n"
                    f"For each row, provide a concise 1-2 sentence plain English explanation of why "
                    f"this sensor combination is abnormal and what machine condition it may indicate.\n\n"
                    f'Respond ONLY as a valid JSON array with no markdown fencing:\n'
                    f'[{{"row_id": <id>, "explanation": "<1-2 sentences>"}}]'
                )

                response = self.client.chat.completions.create(
                    model=GROQ_EXPLAIN_MODEL,
                    max_tokens=GROQ_MAX_TOKENS,
                    messages=[
                        {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                )

                text = response.choices[0].message.content.strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
                text = text.strip()

                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as json_err:
                    # LLM truncated the last item mid-string — recover valid items via regex
                    logger.warning("Groq JSON parse failed, attempting regex recovery: %s", json_err)
                    matches = re.findall(
                        r'"row_id"\s*:\s*(\d+)\s*,\s*"explanation"\s*:\s*"((?:[^"\\]|\\.)*)"',
                        text,
                    )
                    parsed = [{"row_id": int(m[0]), "explanation": m[1]} for m in matches]
                    if parsed:
                        logger.info("Regex recovered %d/%d items from truncated response", len(parsed), len(batch))
                    else:
                        logger.error("Groq batch unrecoverable: %s", json_err)

                returned_ids = {item["row_id"] for item in parsed}
                all_explanations.extend(parsed)
                # Fill in any rows the LLM omitted or returned with wrong IDs
                for r in batch:
                    if r["row_id"] not in returned_ids:
                        logger.warning(
                            "Groq omitted row_id %s from response; using fallback",
                            r["row_id"],
                        )
                        all_explanations.append(
                            {"row_id": r["row_id"], "explanation": "Explanation unavailable"}
                        )

            except Exception as e:
                logger.error("Groq batch failed: %s", e)
                fallback = "Explanation unavailable"
                if _is_rate_limited(str(e)):
                    fallback = "Explanation skipped due to Groq rate limits. Retry later."
                for r in batch:
                    all_explanations.append(
                        {"row_id": r["row_id"], "explanation": fallback}
                    )
                if _is_rate_limited(str(e)):
                    for pending_batch in batches[batch_index + 1 :]:
                        for r in pending_batch:
                            all_explanations.append(
                                {"row_id": r["row_id"], "explanation": fallback}
                            )
                    break

        for r in skipped_rows:
            all_explanations.append(
                {
                    "row_id": r["row_id"],
                    "explanation": "Explanation skipped to conserve free-tier token budget.",
                }
            )

        return all_explanations

    def answer_query(self, question: str, cached_result: dict, context_rows: int = 20) -> str:
        """Answer a natural language question using cached anomaly data as context.

        Args:
            question: The user's natural language question.
            cached_result: The cached /analyze response dict.
            context_rows: Number of top anomaly rows to include as context.

        Returns:
            The LLM's answer string.

        Raises:
            ValueError: If cached_result is None (no analysis has been run).
        """
        if cached_result is None:
            raise ValueError("Run analysis first")

        context_rows = max(1, min(context_rows, GROQ_MAX_QUERY_CONTEXT_ROWS))

        anomalies = cached_result["anomalies"][:context_rows]

        formatted_anomaly_rows = "\n".join(
            f"{a['row_id']} | {a['air_temp']} | {a['process_temp']} | "
            f"{a['rotational_speed']} | {a['torque']} | {a['tool_wear']} | "
            f"{a['anomaly_score']:.4f} | {a.get('failure_types', 'None')}"
            for a in anomalies
        )

        user_prompt = (
            f"You have the following manufacturing sensor anomaly data:\n\n"
            f"Dataset summary:\n"
            f"- Total rows: {cached_result['total_rows']}\n"
            f"- Anomaly count: {cached_result['anomaly_count']}\n"
            f"- Contamination rate used: {cached_result['contamination_used']}\n"
            f"- Features analysed: {', '.join(a for a in cached_result.get('features', ['air_temp', 'process_temp', 'rotational_speed', 'torque', 'tool_wear']))}\n\n"
            f"Top {len(anomalies)} anomaly rows (row_id | air_temp | process_temp | "
            f"rotational_speed | torque | tool_wear | score | failure_type):\n"
            f"{formatted_anomaly_rows}\n\n"
            f"Question: {question}\n\n"
            f"Answer in 2-4 sentences with specific data references where possible."
        )

        try:
            response = self.client.chat.completions.create(
                model=GROQ_QUERY_MODEL,
                max_tokens=GROQ_QUERY_MAX_TOKENS,
                messages=[
                    {"role": "system", "content": QUERY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("Groq query call failed: %s", e)
            if _is_rate_limited(str(e)):
                raise RuntimeError(
                    "Groq free-tier token limit reached. Retry in a few minutes, "
                    "or reduce query size."
                ) from e
            raise RuntimeError(f"Groq query failed: {e}") from e
