"""Prepare a relocatable, floating-base GL-165 asset; requires only Python's standard library."""

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def prepare(source: Path, destination: Path) -> Path:
    source = source.resolve()
    urdf = source if source.is_file() else source / "gl_165.urdf"
    tree = ET.parse(urdf)
    root = tree.getroot()
    ground = root.find("link[@name='ground_link']")
    joint = root.find("joint[@name='ground_joint']")
    if ground is not None:
        if (joint is None or joint.get("type") != "fixed"
                or joint.find("parent").get("link") != "ground_link"
                or joint.find("child").get("link") != "base_link"
                or len(ground)):
            raise ValueError("Unexpected ground reference; refusing to remove physical robot links.")
        root.remove(ground)
        root.remove(joint)
    links = {link.get("name") for link in root.findall("link")}
    children = {j.find("child").get("link") for j in root.findall("joint")}
    if links - children != {"base_link"}:
        raise ValueError("Expected base_link to be the only root link.")
    if len(root.findall("joint[@type='revolute']")) != 31:
        raise ValueError("Expected 31 actuated joints in GL-165.")

    meshes = {}
    for mesh in root.iter("mesh"):
        reference = Path(mesh.get("filename"))
        candidates = [urdf.parent / reference, urdf.parent / "meshes" / reference.name]
        found = next((p.resolve() for p in candidates if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(f"Missing mesh: {reference}")
        if reference.name in meshes and meshes[reference.name] != found:
            raise ValueError(f"Mesh filename collision: {reference.name}")
        meshes[reference.name] = found
        mesh.set("filename", f"../meshes/{reference.name}")

    destination = destination.resolve()
    output = destination / "urdf" / "gl_165.urdf"
    if output == urdf.resolve():
        raise ValueError("Choose a destination different from the source URDF.")
    output.parent.mkdir(parents=True, exist_ok=True)
    (destination / "meshes").mkdir(parents=True, exist_ok=True)
    for name, path in meshes.items():
        target = destination / "meshes" / name
        if target.resolve() != path:
            shutil.copy2(path, target)
    root.set("name", "gl_165")
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    mass = sum(float(m.get("value")) for m in root.findall("link/inertial/mass"))
    print(f"Prepared {output}: 31 joints, {len(meshes)} meshes, {mass:.3f} kg; root=base_link")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT.parent / "data" / "gl_165_URDF")
    parser.add_argument("--destination", type=Path, default=PROJECT_ROOT / "legged_lab/assets/gl_165")
    args = parser.parse_args()
    prepare(args.source, args.destination)
