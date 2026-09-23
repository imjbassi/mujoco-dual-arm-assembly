"""Interactive demo, headless trials, recording, and model export."""

import argparse
import json
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import mujoco

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
        ("record", "Render an annotated animated GIF"),
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


def record(sim, args):
    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise RuntimeError('Recording needs: pip install -e ".[record]"') from error
    if args.fps > 60:
        raise ValueError("Recording fps must be at most 60")
    args.output.mkdir(parents=True, exist_ok=True)
    frames = []
    next_frame = 0.0
    font = ImageFont.load_default(size=22)
    small = ImageFont.load_default(size=16)
    with mujoco.Renderer(sim.model, height=540, width=960) as renderer:
        while True:
            if sim.data.time >= next_frame or sim.done:
                renderer.update_scene(sim.data, camera=args.camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 79), fill=(12, 20, 32))
                draw.text((24, 12), "DUET / DUAL-ARM ASSEMBLY", font=font, fill=(82, 219, 202))
                draw.text(
                    (24, 46),
                    f"{sim.phase}   |   t = {sim.data.time:.2f} s",
                    font=small,
                    fill=(225, 235, 244),
                )
                metrics = sim.measurements()
                draw.rectangle((0, 493, 960, 540), fill=(12, 20, 32))
                label = (
                    f"Lateral: {metrics['lateral_error_mm']:.2f} mm     "
                    f"Depth: {metrics['insertion_depth_mm']:.1f} mm     "
                    f"Contact: {sim.contact_force:.1f} N"
                )
                draw.text((24, 507), label, font=small, fill=(225, 235, 244))
                frames.append(np.array(frame))
                next_frame += 1 / args.fps
            if sim.done:
                break
            sim.step()
        frame.save(args.output / "final.png")
    frames.extend([frames[-1]] * args.fps)
    imageio.mimsave(args.output / "assembly.gif", frames, duration=1000 / args.fps, loop=0)
    save_run(sim, args.output)
    print(f"Recording saved to {args.output.resolve() / 'assembly.gif'}")


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
            summary = {
                "trials": args.trials,
                "successes": successes,
                "success_rate": successes / args.trials,
                "results": results,
            }
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(f"Success rate: {successes}/{args.trials} ({100 * successes / args.trials:.1f}%)")
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
