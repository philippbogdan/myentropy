import argparse
import csv
import math


LABELS = {
    "deep": ["work", "reading"],
    "misc": ["self"],
    "routine": ["food", "commute"],
    "waste": ["social-media"],
}


def build_cat2mode(labels):
    cat2mode = {}
    for mode, cats in labels.items():
        for cat in cats:
            cat2mode[cat] = mode
    return cat2mode


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


def load_schedule_minutes(path, cat2mode):
    minutes = [None] * 1440
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                raise ValueError(f"Expected 3 columns, got {row}")
            start_s, end_s, category = row
            start = parse_time_to_minute(start_s)
            end = parse_time_to_minute(end_s)
            if not (0 <= start < end <= 1440):
                raise ValueError(f"Invalid range {start_s}-{end_s}")
            label = cat2mode.get(category, category)
            for t in range(start, end):
                if minutes[t] is not None:
                    raise ValueError(f"Overlapping schedule at minute {t}")
                minutes[t] = label
    if any(v is None for v in minutes):
        raise ValueError("Schedule does not cover the full 24h day")
    return minutes


def build_block_labels(minutes):
    block_labels = []
    block_durations = {}
    block_order = []
    block_index = -1
    prev = None
    for label in minutes:
        if label != prev:
            block_index += 1
            block_name = f"{label}_{block_index}"
            block_order.append(block_name)
        block_labels.append(block_name)
        block_durations[block_name] = block_durations.get(block_name, 0) + 1
        prev = label
    return block_labels, block_durations, block_order


def compute_entropy(block_durations):
    h = 0.0
    for count in block_durations.values():
        p = count / 1440
        h -= p * math.log(p)
    return h


def main():
    parser = argparse.ArgumentParser(description="Daily entropy score")
    parser.add_argument(
        "--csv",
        default="data/example_perfect.csv",
        help="Path to schedule CSV",
    )
    args = parser.parse_args()

    cat2mode = build_cat2mode(LABELS)
    minutes = load_schedule_minutes(args.csv, cat2mode)
    _, block_durations, block_order = build_block_labels(minutes)
    k = len(block_durations)
    h = compute_entropy(block_durations)

    h_max = math.log(1440)
    h_norm = h / h_max
    h_scaled = h_norm * k
    if h_scaled == 0:
        antientropy = float("inf")
    else:
        antientropy = 1000 / h_scaled
    print(f"H={h:.2g}")
    print(f"H_max={h_max:.2g}")
    print(f"H_norm={h_norm:.2g}")
    print(f"H_norm*K={h_scaled:.2g}")
    print(f"antientropy={antientropy:.2g}")
    print(f"K={k}")
    print("time_per_label:")
    for name in block_order:
        minutes_in_block = block_durations[name]
        print(f"  {name}: {minutes_in_block}")


if __name__ == "__main__":
    main()
