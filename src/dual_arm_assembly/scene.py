"""Original primitive-only robot model; no external meshes or downloads."""

import xml.etree.ElementTree as ET

import mujoco


def build_scene() -> str:
    root = ET.Element("mujoco", model="DUET | dual-arm assembly")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(
        root,
        "option",
        timestep="0.002",
        integrator="implicitfast",
        cone="elliptic",
        iterations="80",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(visual, "headlight", ambient="0.3 0.3 0.3")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="5", armature="0.05", range="-3.05 3.05")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", density="700")
    ET.SubElement(default, "position", kp="1200", kv="90", forcerange="-180 180")
    contact = ET.SubElement(default, "default", **{"class": "contact"})
    ET.SubElement(
        contact,
        "geom",
        contype="1",
        conaffinity="1",
        friction="0.35 0.01 0.001",
        solref="0.008 1",
        solimp="0.95 0.99 0.001",
        margin="0.0002",
    )
    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1="0.10 0.13 0.18",
        rgb2="0.14 0.18 0.23",
        width="512",
        height="512",
    )
    ET.SubElement(
        assets, "material", name="floor", texture="grid", texrepeat="8 8", reflectance="0.1"
    )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", diffuse="0.8 0.85 1")
    ET.SubElement(world, "light", pos="1 1 2.5", dir="-0.5 -0.5 -1", diffuse="0.5 0.6 0.7")
    ET.SubElement(world, "geom", type="plane", size="3 3 0.1", material="floor")
    ET.SubElement(
        world, "geom", type="box", pos="0 0 0.36", size="0.9 0.52 0.04", rgba="0.19 0.24 0.30 1"
    )
    for x in (-0.78, 0.78):
        for y in (-0.4, 0.4):
            ET.SubElement(
                world,
                "geom",
                type="box",
                pos=f"{x} {y} 0.16",
                size="0.035 0.035 0.16",
                rgba="0.09 0.12 0.16 1",
            )
    ET.SubElement(
        world,
        "camera",
        name="overview",
        pos="1.15 -1.6 1.4",
        xyaxes="0.812 0.584 0 -0.190 0.265 0.945",
    )
    ET.SubElement(
        world,
        "camera",
        name="closeup",
        pos="0.35 -0.55 1.0",
        xyaxes="0.814 0.581 0 -0.184 0.257 0.950",
    )
    actuators = ET.SubElement(root, "actuator")
    for name, x, color in (
        ("left", -0.48, "0.10 0.72 0.69 1"),
        ("right", 0.48, "0.96 0.52 0.20 1"),
    ):
        base = ET.SubElement(world, "body", name=f"{name}_base", pos=f"{x} 0.12 0.4")
        ET.SubElement(
            base,
            "geom",
            type="cylinder",
            size="0.095 0.045",
            pos="0 0 0.045",
            rgba="0.12 0.16 0.21 1",
        )
        parent = base
        # Seven revolute joints provide redundant Cartesian pose control.
        axes = ["0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"]
        lengths = [0.08, 0.34, 0.06, 0.32, 0.06, 0.08, 0.06]
        previous = 0.09
        for i, (axis, length) in enumerate(zip(axes, lengths)):
            body = ET.SubElement(
                parent, "body", name=f"{name}_link{i}", pos=f"0 0 {previous}", gravcomp="1"
            )
            ET.SubElement(body, "joint", name=f"{name}_j{i}", axis=axis)
            ET.SubElement(
                body, "geom", type="capsule", fromto=f"0 0 0 0 0 {length}", size="0.035", rgba=color
            )
            ET.SubElement(body, "geom", type="sphere", size="0.041", rgba="0.17 0.21 0.27 1")
            ET.SubElement(
                actuators,
                "position",
                name=f"{name}_a{i}",
                joint=f"{name}_j{i}",
                ctrlrange="-3.05 3.05",
            )
            parent, previous = body, length
        tool = ET.SubElement(
            parent, "body", name=f"{name}_tool", pos=f"0 0 {previous}", gravcomp="1"
        )
        if name == "right":
            # The wrist approaches from above; peg extends away from the forearm.
            tool.set("quat", "0 0 1 0")
        ET.SubElement(tool, "geom", type="box", size="0.045 0.035 0.025", rgba="0.12 0.16 0.21 1")
        if name == "left":
            ET.SubElement(
                tool,
                "geom",
                type="box",
                pos="0 -0.065 0",
                size="0.018 0.065 0.016",
                rgba="0.55 0.61 0.67 1",
            )
            socket = ET.SubElement(tool, "body", name="socket", pos="0 -0.16 0", gravcomp="1")
            # Four boxes form a real open square bore, 28 mm wide and 50 mm deep.
            for label, pos, size in (
                ("east", "0.032 0 0", "0.018 0.05 0.025"),
                ("west", "-0.032 0 0", "0.018 0.05 0.025"),
                ("north", "0 0.032 0", "0.014 0.018 0.025"),
                ("south", "0 -0.032 0", "0.014 0.018 0.025"),
            ):
                ET.SubElement(
                    socket,
                    "geom",
                    name=f"socket_{label}",
                    type="box",
                    pos=pos,
                    size=size,
                    rgba="0.10 0.72 0.69 1",
                    **{"class": "contact"},
                )
            ET.SubElement(socket, "site", name="socket_center", size="0.002", rgba="0.2 1 0.7 0.4")
        else:
            for y in (-0.022, 0.022):
                ET.SubElement(
                    tool,
                    "geom",
                    type="box",
                    pos=f"0 {y} -0.065",
                    size="0.012 0.012 0.05",
                    rgba="0.55 0.61 0.67 1",
                )
            ET.SubElement(
                tool,
                "geom",
                name="peg",
                type="cylinder",
                pos="0 0 -0.15",
                size="0.01 0.05",
                rgba="1 0.76 0.31 1",
                **{"class": "contact"},
            )
            ET.SubElement(
                tool, "site", name="peg_tip", pos="0 0 -0.2", size="0.002", rgba="1 0.85 0.3 1"
            )
    return ET.tostring(root, encoding="unicode")


def load_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_scene())
