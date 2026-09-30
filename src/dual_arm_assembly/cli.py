"""Interactive demo, headless trials, recording, and model export."""

import argparse
import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

from .reporting import save_run
from .scene import build_scene
from .simulation import AssemblySimulation, Config


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def parser():
    root = argparse.ArgumentParser(description="DUET: coordinated dual-arm assembly in MuJoCo")
    sub = root.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("demo", "Open the interactive simulation"),
        ("run", "Run one trial without a display"),
        ("benchmark", "Evaluate seeded workspace variations"),
        ("record", "Render an annotated MP4 video and/or animated GIF"),
    ):
        p = sub.add_parser(command, help=help_text)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--randomize", action="store_true", help="Perturb the assembly location")
        p.add_argument(
            "--offset-mm",
            type=float,
            default=0,
            help="Deliberate lateral target bias; try 12 to produce a jam",
        )
        p.add_argument("--force-limit", type=float, default=35, help="Abort threshold in newtons")
        p.add_argument("--duration", type=float, default=12, help="Timeout in simulated seconds")
        p.add_argument("--output", type=Path, default=Path("runs") / command)
        if command in ("demo", "record"):
            p.add_argument("--camera", choices=("overview", "closeup"), default="overview")
        if command == "benchmark":
            p.add_argument("--trials", type=positive_int, default=20)
        if command == "record":
            p.add_argument("--fps", type=positive_int, default=20)
            p.add_argument(
                "--format",
                choices=("mp4", "gif", "both"),
                default="both",
                help="Container(s) to write; MP4 needs ffmpeg via imageio-ffmpeg",
            )
    export = sub.add_parser("export", help="Write the complete scene as MJCF XML")
    export.add_argument("--output", type=Path, default=Path("runs/scene.xml"))
    return root


def demo(sim, args):
    import mujoco.viewer

    events = deque()
    paused = False
    last_phase = None
    print("SPACE: pause/resume | R: restart same trial | C: switch camera | ESC: close")
    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=events.append,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = sim.model.camera(args.camera).id
        while viewer.is_running():
            started = time.perf_counter()
            with viewer.lock():
                while events:
                    key = events.popleft()
                    if key == 32:
                        paused = not paused
                    elif key in (ord("R"), ord("r")):
                        sim.reset()
                        paused = False
                    elif key in (ord("C"), ord("c")):
                        viewer.cam.fixedcamid = (viewer.cam.fixedcamid + 1) % 2
                if not paused and not sim.done:
                    for _ in range(5):
                        sim.step()
                if sim.phase != last_phase:
                    print(f"{sim.data.time:5.2f}s  {sim.phase}")
                    last_phase = sim.phase
                    if sim.done:
                        save_run(sim, args.output)
                        print(json.dumps(sim.result(), indent=2))
                        print("Press R to replay; ESC to close.")
            viewer.sync()
            time.sleep(max(0, sim.control_dt - (time.perf_counter() - started)))
    if not sim.done:
        sim.status = "interrupted"
    save_run(sim, args.output)


# Progress-bar colors keyed by state-machine phase, matched to the report palette.
PHASE_COLORS = {
    "SETTLE": (110, 126, 143),
    "TRANSFER": (86, 156, 214),
    "ALIGN": (222, 196, 120),
    "INSERT": (245, 133, 51),
    "HOLD": (82, 219, 202),
    "COMPLETE": (95, 210, 120),
    "ABORT": (235, 87, 87),
}


def annotate(sim, frame):
    from PIL import ImageDraw, ImageFont

    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=16)
    width, height = frame.size
    draw = ImageDraw.Draw(frame)
    color = PHASE_COLORS.get(sim.phase, (225, 235, 244))
    draw.rectangle((0, 0, width, 83), fill=(12, 20, 32))
    draw.text((24, 10), "DUET / DUAL-ARM ASSEMBLY", font=font, fill=(82, 219, 202))
    draw.text((24, 44), f"{sim.phase}   |   t = {sim.data.time:.2f} s", font=small, fill=color)
    draw.rectangle((24, 71, width - 24, 77), fill=(41, 56, 73))
    draw.rectangle((24, 71, 24 + (width - 48) * sim.progress, 77), fill=color)
    metrics = sim.measurements()
    draw.rectangle((0, height - 47, width, height), fill=(12, 20, 32))
    label = (
        f"Lateral: {metrics['lateral_error_mm']:.2f} mm    "
        f"Depth: {metrics['insertion_depth_mm']:.1f} mm    "
        f"Axis: {metrics['angle_error_deg']:.2f} deg    "
        f"Contact: {sim.contact_force:.1f} N"
    )
    draw.text((24, height - 33), label, font=small, fill=(225, 235, 244))
    if sim.done:
        banner = "ASSEMBLY COMPLETE" if sim.status == "success" else sim.status.upper()
        box = draw.textbbox((0, 0), banner, font=font)
        draw.rectangle((0, 96, width, 148), fill=(12, 20, 32))
        draw.text(((width - box[2]) / 2, 108), banner, font=font, fill=color)
    return frame


def record(sim, args):
    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError('Recording needs: pip install -e ".[record]"') from error
    if args.fps > 60:
        raise ValueError("Recording fps must be at most 60")
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    next_frame = 0.0
    # 960x544 keeps both dimensions divisible by 16, so MP4 encoding never rescales.
    with mujoco.Renderer(sim.model, height=544, width=960) as renderer:
        while True:
            if sim.data.time >= next_frame or sim.done:
                renderer.update_scene(sim.data, camera=args.camera)
                frame = annotate(sim, Image.fromarray(renderer.render()))
                frames.append(np.array(frame))
                next_frame += 1 / args.fps
            if sim.done:
                break
            sim.step()
        frame.save(args.output / "final.png")
    frames.extend([frames[-1]] * args.fps)
    written = []
    if args.format in ("mp4", "both"):
        try:
            imageio.mimsave(args.output / "assembly.mp4", frames, fps=args.fps, quality=8)
            written.append("assembly.mp4")
        except Exception as error:
            raise RuntimeError(
                'MP4 encoding needs ffmpeg: pip install -e ".[record]" imageio-ffmpeg'
            ) from error
    if args.format in ("gif", "both"):
        imageio.mimsave(args.output / "assembly.gif", frames, duration=1000 / args.fps, loop=0)
        written.append("assembly.gif")
    save_run(sim, args.output)
    for name in written:
        print(f"Recording saved to {args.output.resolve() / name}")


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command == "export":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(build_scene(), encoding="utf-8")
        print(args.output.resolve())
        return 0
    try:
        config = Config(
            seed=args.seed,
            randomize=args.randomize,
            offset_mm=args.offset_mm,
            force_limit=args.force_limit,
            duration=args.duration,
        )
        if args.command == "benchmark":
            results = []
            for i in range(args.trials):
                sim = AssemblySimulation(replace(config, seed=config.seed + i, randomize=True))
                result = sim.run()
                results.append(result)
                save_run(sim, args.output / f"seed-{config.seed + i:04d}")
                print(f"[{i + 1}/{args.trials}] seed={config.seed + i}: {sim.status}")
            successes = sum(result["success"] for result in results)
            statuses = {}
            for result in results:
                statuses[result["status"]] = statuses.get(result["status"], 0) + 1
            stats = {
                field: {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                for field in ("lateral_error_mm", "angle_error_deg", "peak_contact_force_n")
                for values in [[result[field] for result in results]]
            }
            summary = {
                "trials": args.trials,
                "successes": successes,
                "success_rate": successes / args.trials,
                "statuses": statuses,
                "stats": stats,
                "results": results,
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(f"Success rate: {successes}/{args.trials} ({100 * successes / args.trials:.1f}%)")
            print(
                f"Lateral error mm: median {stats['lateral_error_mm']['median']:.2f}, "
                f"max {stats['lateral_error_mm']['max']:.2f} | "
                f"peak force N: median {stats['peak_contact_force_n']['median']:.2f}, "
                f"max {stats['peak_contact_force_n']['max']:.2f}"
            )
            return 0 if successes == args.trials else 1
        sim = AssemblySimulation(config)
        if args.command == "demo":
            demo(sim, args)
            return 0
        if args.command == "record":
            record(sim, args)
        else:
            sim.run()
            save_run(sim, args.output)
        print(json.dumps(sim.result(), indent=2))
        print(f"Report: {(args.output / 'report.html').resolve()}")
        return 0 if sim.status == "success" else 1
    except (ValueError, RuntimeError) as error:
        p.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
