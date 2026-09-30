"""Portable JSON/CSV logs and a standalone HTML experiment report."""

import csv
import html
import json
from pathlib import Path

import mujoco
import numpy as np


def save_run(sim, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {**sim.result(), "versions": {"mujoco": mujoco.__version__, "numpy": np.__version__}}
    (directory / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (directory / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sim.trace[0]))
        writer.writeheader()
        writer.writerows(sim.trace)
    charts = []
    for field, title, unit, guides in (
        ("lateral_error_mm", "Lateral alignment", "mm", [(3.0, "3 mm success limit")]),
        ("insertion_depth_mm", "Insertion depth", "mm", [(36.0, "min 36"), (44.0, "max 44")]),
        ("angle_error_deg", "Axis alignment", "deg", [(2.0, "2° success limit")]),
        (
            "contact_force_n",
            "Peg / socket contact force",
            "N",
            [(sim.config.force_limit, f"{sim.config.force_limit:g} N abort")],
        ),
    ):
        values = [row[field] for row in sim.trace]
        low = min(0, min(values))
        high = max(1, max(values), *(value for value, _ in guides))
        end = max(sim.data.time, 0.001)

        def scale_y(value, low=low, high=high):
            return 170 - (value - low) / (high - low) * 140

        points = " ".join(
            f"{45 + row['time'] / end * 690:.1f},{scale_y(row[field]):.1f}" for row in sim.trace
        )
        guide_marks = "".join(
            f'<line x1="45" x2="740" y1="{scale_y(value):.1f}" y2="{scale_y(value):.1f}"'
            f' stroke="#8d6a3f" stroke-dasharray="6 5"/>'
            f'<text x="452" y="{scale_y(value) - 5:.1f}" fill="#d9a45f" font-size="11">'
            f"{html.escape(label)}</text>"
            for value, label in guides
        )
        charts.append(f'''<section><h2>{title}</h2><svg viewBox="0 0 760 205" role="img"
          aria-label="{title} over simulation time"><path d="M45 25V170H740" stroke="#536273"
          fill="none"/>{guide_marks}<polyline points="{points}" fill="none" stroke="#52dbca"
          stroke-width="2.5"/>
          <g fill="#a8b8c9" font-size="12"><text x="3" y="27">{high:.1f}</text>
          <text x="3" y="170">{low:.1f}</text><text x="45" y="195">0 s</text>
          <text x="680" y="195">{end:.2f} s</text><text x="45" y="17">{unit}</text></g>
          </svg></section>''')
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>DUET | Trial report</title>
    <style>body{margin:0;background:#0c1420;color:#e5edf5;font:16px system-ui}
    main{max-width:1000px;margin:auto;padding:48px 24px}h1{font-size:42px;margin:12px 0}
    h2{font-size:17px}p{color:#a8b8c9;line-height:1.6}.eyebrow{color:#52dbca;letter-spacing:3px}
    .cards{display:flex;gap:16px;flex-wrap:wrap}.card,section{background:#152130;border:1px solid #293849;
    border-radius:14px;padding:22px;margin:12px 0}.card{flex:1;min-width:160px}
    strong{display:block;font-size:28px;margin-top:8px}svg{width:100%}pre{overflow:auto;font-size:13px}
    a{color:#52dbca}</style><main><div class="eyebrow">DUET / BIMANUAL ASSEMBLY</div>
    <h1>Two arms. One precise fit.</h1><p>Physics-driven peg insertion · MuJoCo · deterministic baseline</p>"""
    page += '<div class="cards">'
    for title, value in (
        ("Trial outcome", sim.status.upper()),
        ("Final lateral error", f"{result['lateral_error_mm']:.3f} mm"),
        ("Final insertion depth", f"{result['insertion_depth_mm']:.1f} mm"),
        ("Peak contact force", f"{sim.peak_force:.2f} N"),
    ):
        page += f'<div class="card">{title}<strong>{value}</strong></div>'
    page += "</div>" + "".join(charts)
    page += "<section><h2>Experiment details</h2><pre>"
    page += html.escape(json.dumps(result, indent=2)) + "</pre></section>"
    page += """<p>Ideal fixed grasps; exact simulator state; only peg/socket collisions enabled.
    A centered peg can pass through the bore with zero contact force.
    Force is the sum of contact-force magnitudes, not a wrist sensor measurement.</p>
    <a href="trace.csv">Download trace CSV</a> · <a href="result.json">Result JSON</a></main></html>"""
    (directory / "report.html").write_text(page, encoding="utf-8")
    return directory
