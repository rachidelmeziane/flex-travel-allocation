"""Train scheduling optimisation: trains scheduled under a sales vs an activation plan."""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, ".")

import pandas as pd

from conf.params import PARAMS
from flex_travel_allocation.train_scheduling import train_optimisation_summary, trains_by_hour

SEATS_PER_TRAIN = PARAMS["train_scheduling"]["seats_per_train"]
TARGET_LOAD_FACTOR = PARAMS["train_scheduling"]["target_load_factor"]

capacity = pd.read_csv("streamlit_app/artifacts/output_capacity_by_hour.csv")
trains = trains_by_hour(capacity, seats_per_train=SEATS_PER_TRAIN, target_load_factor=TARGET_LOAD_FACTOR)
trains.to_csv("streamlit_app/artifacts/output_trains_by_hour.csv", index=False)
summary = train_optimisation_summary(trains)

lines = []
lines.append("TRAIN SCHEDULING: sales plan vs activation plan")
lines.append("=" * 60)
lines.append(f"(seats per train = {SEATS_PER_TRAIN}, target load factor = {TARGET_LOAD_FACTOR:.0%})")
lines.append("")
lines.append("%4s %13s %18s %10s" % ("hour", "trains_sales", "trains_activation", "saved"))
for row in trains.itertuples():
    lines.append("%4d %13d %18d %10d" % (row.travel_hour, row.trains_sales, row.trains_activation, row.trains_saved))
lines.append("-" * 60)
lines.append(
    "%-18s sales %s  activation %s  saved %s"
    % (
        "trains/day:",
        format(summary["trains_sales"], ","),
        format(summary["trains_activation"], ","),
        format(summary["trains_saved"], ","),
    )
)
lines.append("Reduction by planning to activation: -%.1f%% of scheduled trains." % (summary["reduction_pct"] * 100))
lines.append(
    "Peak-hour trains: sales %s vs activation %s."
    % (format(summary["peak_trains_sales"], ","), format(summary["peak_trains_activation"], ","))
)

report = "\n".join(lines)
with open("train_scheduling_simulation.txt", "w", encoding="utf-8") as handle:
    handle.write(report)
print(report)
