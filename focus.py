import csv

from label_classifier import CATEGORIES, GrokClassifier, load_api_key


CORE_ACTIVITIES = {
    "work",
    "lecture",
    "lab",
    "coding",
    "class-english",
    "class-math",
    "class-science",
    "class-history",
    "class-pe",
    "study",
    "homework",
    "writing",
    "office-work",
    "meetings",
    "design",
    "product",
    "fundraising",
    "email",
    "paperwork",
    "ward-rounds",
    "clinic",
    "hospital-duty",
    "mining",
    "dance-practice",
    "vocal-practice",
    "rehearsal",
    "performance",
    "show",
    "media",
}

SELF_CARE_ACTIVITIES = {
    "sleep",
    "sleeping",
    "alarm",
    "hygiene",
    "bathroom",
    "brushing-teeth",
    "shower",
    "skincare",
    "getting-dressed",
    "breakfast",
    "lunch",
    "dinner",
    "snack",
    "break-snack",
    "food",
    "coffee",
    "commute",
    "walking",
    "walk",
    "bus-ride",
    "travel",
    "locker",
    "corridor",
    "packing-bag",
    "changing-clothes",
    "chores",
    "housework",
    "washing-dishes",
    "childcare",
    "errands",
    "exercise",
    "gym",
    "rest",
    "prepare-next-day",
    "self",
}

WASTE_ACTIVITIES = {
    "social-media",
    "phone-scrolling",
    "gaming",
    "tv",
}

PERIPHERAL_ACTIVITIES = {
    "reading",
    "family",
    "socializing",
    "chatting-friends",
    "gardening",
}


SWITCH_WEIGHTS = {
    ("core", "self-care"): 0.6,
    ("core", "waste"): 1.0,
    ("core", "peripheral"): 0.5,
    ("self-care", "core"): 0.2,
    ("self-care", "waste"): 0.7,
    ("self-care", "peripheral"): 0.3,
    ("waste", "core"): 0.0,
    ("waste", "self-care"): 0.4,
    ("waste", "peripheral"): 0.6,
    ("peripheral", "core"): 0.2,
    ("peripheral", "self-care"): 0.4,
    ("peripheral", "waste"): 0.8,
}


def build_activity_map():
    mapping = {}
    for category, labels in (
        ("core", CORE_ACTIVITIES),
        ("self-care", SELF_CARE_ACTIVITIES),
        ("waste", WASTE_ACTIVITIES),
        ("peripheral", PERIPHERAL_ACTIVITIES),
    ):
        for label in labels:
            key = label.strip().lower()
            if key in mapping and mapping[key] != category:
                raise ValueError(f"Activity '{label}' mapped twice")
            mapping[key] = category
    return mapping


def validate_weights():
    expected = {(a, b) for a in CATEGORIES for b in CATEGORIES if a != b}
    missing = expected - SWITCH_WEIGHTS.keys()
    extra = SWITCH_WEIGHTS.keys() - expected
    if missing:
        raise ValueError(f"Missing weights for: {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected weights for: {sorted(extra)}")


def parse_time_to_minute(value):
    parts = value.strip().split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid time '{value}'")
    hour = int(parts[0])
    minute = int(parts[1])
    if hour == 24 and minute == 0:
        return 1440
    if not (0 <= hour < 24) or not (0 <= minute < 60):
        raise ValueError(f"Invalid time '{value}'")
    return hour * 60 + minute


def load_schedule_rows(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"Expected 3 columns, got {row}")
            start_s, end_s, activity = row
            start = parse_time_to_minute(start_s)
            end = parse_time_to_minute(end_s)
            if not (0 <= start < end <= 1440):
                raise ValueError(f"Invalid range {start_s}-{end_s}")
            rows.append((start, end, activity))
    return rows


def build_resolved_activity_map(rows, activity_map, classifier=None, cache=None, goals_context=None):
    resolved = dict(activity_map)
    if cache:
        resolved.update({key: value for key, value in cache.items() if value in CATEGORIES})
    unknown = sorted(
        {
            activity.strip().lower()
            for _, _, activity in rows
            if activity.strip().lower() not in resolved
        }
    )
    if unknown:
        if classifier is None:
            preview = ", ".join(unknown[:10])
            suffix = "..." if len(unknown) > 10 else ""
            raise ValueError(f"Unknown activity label(s): {preview}{suffix}")
        classified = classifier.classify_many(unknown, goals_context=goals_context)
        resolved.update(classified)
        if cache is not None:
            cache.update(classified)
    return resolved


def load_schedule_categories(rows, activity_map):
    minutes = [None] * 1440
    for start, end, activity in rows:
        key = activity.strip().lower()
        if key not in activity_map:
            raise ValueError(f"Unknown activity label '{activity}'")
        category = activity_map[key]
        for t in range(start, end):
            if minutes[t] is not None:
                raise ValueError(f"Overlapping schedule at minute {t}")
            minutes[t] = category
    if any(v is None for v in minutes):
        raise ValueError("Schedule does not cover the full 24h day")
    return minutes


def compute_switch_penalty(categories):
    total = 0.0
    switches = 0
    for t in range(1, 1440):
        prev = categories[t - 1]
        curr = categories[t]
        if prev != curr:
            switches += 1
            total += SWITCH_WEIGHTS[(prev, curr)]
    return total, switches


def compute_focus_from_categories(categories):
    penalty, switches = compute_switch_penalty(categories)
    if penalty == 0:
        focus = float("inf")
    else:
        focus = 1000 / penalty
    return {
        "focus": focus,
        "penalty": penalty,
        "switches": switches,
    }


def build_classifier(api_key=None, base_url=None, model=None, key_path="env"):
    if not api_key:
        api_key = load_api_key(path=key_path)
    return GrokClassifier(api_key=api_key, base_url=base_url, model=model)


def compute_focus_from_csv(
    path,
    activity_map=None,
    classifier=None,
    cache=None,
    goals_context=None,
):
    validate_weights()
    activity_map = activity_map or build_activity_map()
    rows = load_schedule_rows(path)
    activity_map = build_resolved_activity_map(
        rows,
        activity_map,
        classifier=classifier,
        cache=cache,
        goals_context=goals_context,
    )
    categories = load_schedule_categories(rows, activity_map)
    return compute_focus_from_categories(categories)
