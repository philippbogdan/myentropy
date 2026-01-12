import json
import os
import urllib.error
import urllib.request


CATEGORIES = ("core", "self-care", "peripheral", "waste")
DEFAULT_MODEL = "grok-4-1-fast-non-reasoning"
DEFAULT_BASE_URL = "https://api.x.ai/v1"
DEFAULT_USER_AGENT = "curl/8.4.0"


SYSTEM_PROMPT = (
    "You are a classifier. Output exactly one of: core, self-care, peripheral, waste. "
    "No punctuation or extra text."
)
SYSTEM_PROMPT_BATCH = "You are a classifier. Return JSON only."

USER_PROMPT = """Goals context:
{goals_context}

Classify this activity into exactly one category.

Activity: {activity}

Definitions:
- core:
  activities that directly advance the user’s current primary goals.
  these are the big rock actions tied to defined outcomes or milestones.
- self-care:
  essential activities required to maintain physical health, mental health, and basic daily functioning.
  includes sleep, meals, basic cooking to eat, hygiene, commuting, chores, errands, exercise, medical needs, and essential admin.
- peripheral:
  positive and intentional activities that do not directly advance current core goals.
  they may build breadth, enjoyment, or long-term optionality, but are not goal-critical right now.
- waste:
  low-value activities that primarily consume time or attention without meaningful benefit.
  includes mindless scrolling, passive consumption, entertainment binges, watching sports, and reading news unless explicitly goal-driven.

Decision chain:
1) Does this directly advance one of the stated goals/projects? -> core
2) Is it necessary for health or basic functioning? -> self-care
3) Is it primarily passive consumption with low return? -> waste
4) Otherwise -> peripheral

Important examples:
- deliberate piano practice -> core ONLY if user has a goal involving piano professionally / exams; if goals describe piano as a hobby or casual interest, classify as peripheral
- learning to cook -> core ONLY if user’s goals include cooking professionally/content; self-care if just to feed self; otherwise peripheral
- reading news -> waste by default; core ONLY if explicitly needed for a current goal/project (research/industry tracking)
"""

BATCH_PROMPT = """Goals context:
{goals_context}

Classify each activity into exactly one category.
Return a JSON object mapping each activity string to its category.

Activities (JSON array):
{activities_json}

Definitions:
- core:
  activities that directly advance the user’s current primary goals.
  these are the big rock actions tied to defined outcomes or milestones.
- self-care:
  essential activities required to maintain physical health, mental health, and basic daily functioning.
  includes sleep, meals, basic cooking to eat, hygiene, commuting, chores, errands, exercise, medical needs, and essential admin.
- peripheral:
  positive and intentional activities that do not directly advance current core goals.
  they may build breadth, enjoyment, or long-term optionality, but are not goal-critical right now.
- waste:
  low-value activities that primarily consume time or attention without meaningful benefit.
  includes mindless scrolling, passive consumption, entertainment binges, watching sports, and reading news unless explicitly goal-driven.

Decision chain:
1) Does this directly advance one of the stated goals/projects? -> core
2) Is it necessary for health or basic functioning? -> self-care
3) Is it primarily passive consumption with low return? -> waste
4) Otherwise -> peripheral

Important examples:
- deliberate piano practice -> core ONLY if user has a goal involving piano professionally / exams; if goals describe piano as a hobby or casual interest, classify as peripheral
- learning to cook -> core ONLY if user’s goals include cooking professionally/content; self-care if just to feed self; otherwise peripheral
- reading news -> waste by default; core ONLY if explicitly needed for a current goal/project (research/industry tracking)

Return JSON only, no extra text.
"""


def format_goals_context(goals_context):
    if goals_context is None:
        return "none"
    if isinstance(goals_context, str):
        return goals_context.strip() or "none"
    if isinstance(goals_context, dict):
        lines = []
        goals = goals_context.get("goals") or []
        projects = goals_context.get("projects") or []
        if goals:
            lines.append("goals:")
            lines.extend(f"- {goal}" for goal in goals)
        if projects:
            lines.append("projects:")
            lines.extend(f"- {project}" for project in projects)
        return "\n".join(lines) if lines else "none"
    if isinstance(goals_context, (list, tuple, set)):
        items = [str(item) for item in goals_context if str(item).strip()]
        if not items:
            return "none"
        return "\n".join(f"- {item}" for item in items)
    return str(goals_context)


def parse_json_object(text):
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


class GrokClassifier:
    def __init__(self, api_key, base_url=None, model=None, timeout=20, user_agent=None):
        self.api_key = api_key
        self.base_url = (base_url or os.environ.get("XAI_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.user_agent = user_agent or DEFAULT_USER_AGENT

    def classify(self, activity, goals_context=None):
        goals_text = format_goals_context(goals_context)
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 8,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(
                        activity=activity,
                        goals_context=goals_text,
                    ),
                },
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self.user_agent,
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = response["choices"][0]["message"]["content"]
        token = content.strip().lower()
        if token:
            token = token.split()[0].strip(".,;:!\"'`")
        if token not in CATEGORIES:
            raise ValueError(f"LLM returned invalid category '{content}'")
        return token

    def classify_many(self, activities, goals_context=None, retry_count=0):
        activities_list = list(activities)
        if not activities_list:
            return {}

        # Track call count if attribute exists
        if hasattr(self, 'call_count'):
            self.call_count += 1

        goals_text = format_goals_context(goals_context)
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 1024,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_BATCH},
                {
                    "role": "user",
                    "content": BATCH_PROMPT.format(
                        goals_context=goals_text,
                        activities_json=json.dumps(activities_list, ensure_ascii=True),
                    ),
                },
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self.user_agent,
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response = json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed: {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = response["choices"][0]["message"]["content"]
        parsed = parse_json_object(content)
        if not isinstance(parsed, dict):
            raise ValueError(f"LLM returned non-object JSON: {content}")

        mapping = {}
        missing = []
        for item in activities_list:
            if item in parsed:
                value = parsed[item]
                token = str(value).strip().lower()
                if token in CATEGORIES:
                    mapping[item] = token
                else:
                    missing.append(item)
            else:
                missing.append(item)

        # Retry missing items (up to 2 retries)
        if missing and retry_count < 2:
            print(f"Retrying {len(missing)} missing items: {missing}")
            retry_results = self.classify_many(missing, goals_context, retry_count + 1)
            mapping.update(retry_results)

        # Final fallback for anything still missing
        for item in activities_list:
            if item not in mapping:
                print(f"Warning: Giving up on '{item}', defaulting to peripheral")
                mapping[item] = "peripheral"

        return mapping


def load_cache(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        raise ValueError("LLM cache must be a JSON object")
    normalized = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        norm = value.strip().lower()
        if norm in CATEGORIES:
            normalized[key.strip().lower()] = norm
    return normalized


def save_cache(path, cache):
    with open(path, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)


def load_api_key(path=".env", env_var="XAI_API_KEY"):
    env_value = os.environ.get(env_var)
    if env_value:
        return env_value.strip()
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == env_var and value.strip():
                    return value.strip()
    except FileNotFoundError:
        pass
    raise ValueError(f"{env_var} not found in environment or {path}")
