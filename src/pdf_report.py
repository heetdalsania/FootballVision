"""Printable multi-page PDF report built entirely from local session data."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from src.session_analytics import build_session_analytics


TEAM_COLORS = {0: "#e5484d", 1: "#3b82f6"}


def build_pdf_report(
    report_image: str | Path,
    output_path: str | Path,
    snapshots: list[dict],
    events: list[dict],
    source_name: str,
    session_id: int,
) -> str:
    """Create a two-page printable report and return its path."""
    report_image, output_path = Path(report_image), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    analytics = build_session_analytics(snapshots, events)
    accepted = [
        event for event in events if event.get("review_status") != "rejected"
    ]

    with PdfPages(output_path) as pdf:
        _overview_page(pdf, report_image, session_id)
        _intelligence_page(
            pdf, snapshots, accepted, analytics, source_name, session_id
        )
    return str(output_path)


def _overview_page(pdf: PdfPages, report_image: Path, session_id: int) -> None:
    image = plt.imread(report_image)
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([.035, .045, .93, .91])
    ax.imshow(image)
    ax.axis("off")
    fig.text(.5, .018, f"FootballVision - Session {session_id} - Page 1 of 2",
             ha="center", fontsize=7, color="#777")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _intelligence_page(
    pdf: PdfPages,
    snapshots: list[dict],
    events: list[dict],
    analytics: dict,
    source_name: str,
    session_id: int,
) -> None:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    grid = fig.add_gridspec(
        4, 2, height_ratios=[.18, 1.05, .72, .72],
        left=.07, right=.95, top=.96, bottom=.06, hspace=.38, wspace=.28,
    )
    header = fig.add_subplot(grid[0, :])
    header.axis("off")
    header.set_xlim(0, 1)
    header.set_ylim(0, 1)
    header.text(0, .72, "Match intelligence appendix", fontsize=20,
                fontweight="bold", color="#111")
    header.text(
        0, .28,
        f"Session {session_id} - {source_name} - "
        f"{analytics['snapshot_count']} measured states - "
        f"{analytics['event_count']} accepted events",
        fontsize=8.5, color="#666",
    )
    header.plot([0, 1], [.04, .04], color="#222", lw=1)

    event_ax = fig.add_subplot(grid[1, :])
    _event_map(event_ax, events)
    _formation_panel(fig.add_subplot(grid[2, 0]), snapshots)
    _phase_panel(fig.add_subplot(grid[2, 1]), snapshots)
    _team_summary(fig.add_subplot(grid[3, 0]), analytics)
    _review_panel(fig.add_subplot(grid[3, 1]), analytics)
    fig.text(.5, .018, f"FootballVision - Session {session_id} - Page 2 of 2",
             ha="center", fontsize=7, color="#777")
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def _event_map(ax, events: list[dict]) -> None:
    ax.set_facecolor("#f8faf8")
    ax.add_patch(Rectangle((0, 0), 105, 68, fill=False, ec="#9ca39d", lw=1))
    ax.axvline(52.5, color="#b9beb9", lw=.8)
    circle = plt.Circle((52.5, 34), 9.15, fill=False, ec="#b9beb9", lw=.8)
    ax.add_patch(circle)
    symbols = {
        "pass": "o", "turnover": "X", "carry": "s",
        "restart": "D", "shot_candidate": "*", "possession_start": "^",
    }
    for event in events:
        if event.get("x") is None or event.get("y") is None:
            continue
        team = event.get("team")
        ax.scatter(
            event["x"], event["y"], s=46,
            marker=symbols.get(event.get("type"), "o"),
            color=TEAM_COLORS.get(team, "#6b7280"),
            edgecolor="white", linewidth=.5, alpha=.85,
        )
    ax.set_xlim(-2, 107)
    ax.set_ylim(-2, 70)
    ax.set_aspect("equal")
    ax.set_title("Accepted event locations", fontsize=10, loc="left")
    ax.set_xlabel("pitch length (m)", fontsize=7)
    ax.set_ylabel("pitch width (m)", fontsize=7)
    ax.tick_params(labelsize=7, colors="#777")
    ax.spines[:].set_visible(False)


def _formation_panel(ax, snapshots: list[dict]) -> None:
    counts = {0: Counter(), 1: Counter()}
    for snapshot in snapshots:
        formations = (snapshot.get("intelligence") or {}).get("formations") or {}
        for team in (0, 1):
            name = (formations.get(str(team)) or {}).get("name")
            if name and name != "insufficient data":
                counts[team][name] += 1
    labels = sorted(set(counts[0]) | set(counts[1]))
    if labels:
        y = range(len(labels))
        ax.barh([value + .18 for value in y], [counts[0][name] for name in labels],
                height=.34, color=TEAM_COLORS[0], label="Team A")
        ax.barh([value - .18 for value in y], [counts[1][name] for name in labels],
                height=.34, color=TEAM_COLORS[1], label="Team B")
        ax.set_yticks(list(y), labels)
    ax.set_title("Formation samples", fontsize=10, loc="left")
    ax.set_xlabel("saved states", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def _phase_panel(ax, snapshots: list[dict]) -> None:
    phases = Counter()
    for snapshot in snapshots:
        phase = ((snapshot.get("intelligence") or {}).get("phase") or {}).get("label")
        if phase:
            phases[phase] += 1
    if phases:
        labels, values = zip(*phases.most_common())
        ax.barh(range(len(labels)), values, color="#ff6f3c")
        ax.set_yticks(range(len(labels)), [label.replace("-", " ") for label in labels])
    ax.set_title("Observed match phases", fontsize=10, loc="left")
    ax.set_xlabel("saved states", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)


def _team_summary(ax, analytics: dict) -> None:
    ax.axis("off")
    ax.set_title("Team summary", fontsize=10, loc="left", pad=8)
    lines = []
    for team, name in (("0", "Team A"), ("1", "Team B")):
        values = analytics["teams"][team]
        lines.extend([
            (name, TEAM_COLORS[int(team)], True),
            (
                f"Possession samples {values['possession_sample_pct']}%  |  "
                f"control {values['avg_control_pct']}%",
                "#333", False,
            ),
            (
                f"Final-third {values['final_third_presence_pct']}%  |  "
                f"compactness {values['avg_compactness_m']} m",
                "#555", False,
            ),
            (
                f"Width {values['avg_width_m']} m  |  "
                f"passes {values['passes']}  |  recoveries {values['turnovers_won']}",
                "#555", False,
            ),
        ])
    y = .88
    for text, color, bold in lines:
        ax.text(.02, y, text, fontsize=7.6, color=color,
                fontweight="bold" if bold else "normal", va="top")
        y -= .125 if bold else .105


def _review_panel(ax, analytics: dict) -> None:
    ax.axis("off")
    ax.set_title("Event review status", fontsize=10, loc="left", pad=8)
    review = analytics.get("review_counts") or {}
    events = analytics.get("event_counts") or {}
    y = .86
    for label, count in review.items():
        ax.text(.02, y, label.replace("_", " ").title(), fontsize=8, color="#555")
        ax.text(.96, y, str(count), fontsize=8, ha="right", color="#111")
        y -= .1
    y -= .035
    ax.text(.02, y, "Accepted timeline", fontsize=8.5, fontweight="bold")
    y -= .115
    for label, count in events.items():
        ax.text(.02, y, label.replace("_", " "), fontsize=8, color="#555")
        ax.text(.96, y, str(count), fontsize=8, ha="right", color="#111")
        y -= .095
