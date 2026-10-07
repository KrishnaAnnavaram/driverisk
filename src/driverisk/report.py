"""Model card in Markdown."""

from __future__ import annotations


def model_card(summary: dict, source: str, context_summary: str) -> str:
    t = summary["test"]
    lines = [
        "# Model card: driverisk",
        "",
        f"- Data source: `{source}`",
        f"- Trips: {summary['trips']}, devices: {summary['devices']} "
        f"(train {summary['train_devices']}, held out {summary['test_devices']})",
        "- Context: " + context_summary.replace("\n", "; "),
        "- Target: harsh events per trip (braking <= -3.0 m/s2, acceleration >= 2.5 m/s2, lateral >= 3.0 m/s2)",
        "- Exposure: trip distance in km",
        f"- Chosen model: `{summary['chosen_model']}` ({summary['chosen_by']})",
        "",
        "## Grouped cross-validation (training devices)",
        "",
        "| Model | Poisson deviance | D2 vs fleet rate | Normalised Gini |",
        "|---|---|---|---|",
    ]
    for name, cv in summary["cv"].items():
        lines.append(
            f"| {name} | {cv['poisson_deviance_mean']:.4f} ± {cv['poisson_deviance_std']:.4f} "
            f"| {cv['d2_mean']:.3f} | {cv['normalised_gini_mean']:.3f} |"
        )
    lines += [
        "",
        "## Held-out devices (used once)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Trips / events / km | {t['trips']} / {t['events']:.0f} / {t['km']:.0f} |",
        f"| Poisson deviance (model / fleet rate) | {t['poisson_deviance']:.4f} / {t['reference_deviance']:.4f} |",
        f"| D2 vs fleet rate | {t['d2']:.3f} |",
        f"| Normalised Gini | {t['normalised_gini']:.3f} |",
        f"| Expected / observed events | {t['expected_over_observed']:.3f} |",
        f"| Driver-level Spearman (predicted vs observed rate, {t['drivers']} drivers) | {t['driver_spearman']:.3f} |",
        "",
        "## Top features (permutation importance, Poisson deviance increase)",
        "",
        "| Feature | Increase |",
        "|---|---|",
    ]
    for row in summary["importance"]:
        lines.append(f"| {row['feature']} | {row['deviance_increase']:.4f} |")
    lines += [
        "",
        "## Limits and responsible use",
        "",
        "- The target is harsh events, not claims. A harsh event is a proxy for risk, not a crash.",
        "- The score is not a pricing or underwriting decision. A person must review it.",
        "- Telematics traces are location data. Keep them private and get consent from each driver.",
        "- Results on synthetic data show that the code works. They do not predict results on real fleets.",
    ]
    return "\n".join(lines) + "\n"
