#!/usr/bin/env python3
"""Sweep PME/ESP Coulomb force error for OpenMM benchmark systems.

The script compares Coulomb-only PME and ESP against a tighter double-precision
PME reference.  Lennard-Jones epsilons are set to zero, while OpenMM
NonbondedForce exceptions are preserved, so the measured force is:

    screened direct, exclusions filtered
  + reciprocal, all pairs
  - long-range part of excluded/exception pairs

It writes raw CSV/JSON files for audit and prints compact summary tables for
the cutoff/tolerance calibration.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import pathlib
import socket
import subprocess
import sys
import tarfile
import urllib.request


DEFAULT_SYSTEMS = ("pme", "apoa1pme", "amber20-cellulose", "amber20-stmv")
DEFAULT_CUTOFFS = tuple(round(0.40 + 0.05*i, 2) for i in range(23))
DEFAULT_TOLERANCES = (1e-3, 5e-4, 2e-4, 1e-4)
DEFAULT_METHODS = ("ESP", "PME")
SYSTEM_LABELS = {
    "pme": "DHFR",
    "apoa1pme": "ApoA1",
    "amber20-dhfr": "Amber20-DHFR",
    "amber20-cellulose": "Cellulose",
    "amber20-stmv": "STMV",
}
AMBER20_NAMES = {
    "amber20-dhfr": "JAC",
    "amber20-cellulose": "Cellulose",
    "amber20-stmv": "STMV",
}


def timestamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def source_root_from_script() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def git_revision(source_root: pathlib.Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_root, text=True).strip()
    except Exception:
        return "unknown"


def csv_items(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


def parse_floats(text: str, name: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in csv_items(text))
    if not values or any(value <= 0.0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated values")
    return values


def parse_methods(text: str) -> tuple[str, ...]:
    methods = tuple(item.upper() for item in csv_items(text))
    unknown = sorted(set(methods) - {"PME", "ESP"})
    if not methods or unknown:
        raise ValueError("methods must be PME, ESP, or both")
    return methods


def parse_reference_settings(text: str) -> dict[str, tuple[float, float]]:
    """Parse system-specific reference overrides: system:cutoff_nm:tolerance."""
    overrides = {}
    for item in csv_items(text):
        fields = item.split(":")
        if len(fields) != 3:
            raise ValueError(f"Invalid reference setting {item!r}; expected system:cutoff_nm:tolerance")
        overrides[fields[0]] = (float(fields[1]), float(fields[2]))
    return overrides


def estimate_esp_order(tol: float, cutoff: float) -> int:
    """Mirror pswf::estimateEspOrder(tol, cutoff) because OpenMM does not expose P."""
    setup_tol = tol*math.sqrt(cutoff)
    p = -math.log10(setup_tol)
    rounded = round(p)
    order = 2*int(rounded) - 2 if abs(p - rounded) < 0.2 else 2*math.ceil(p) - 3
    if tol <= 1e-4:
        order = max(order, 7 if cutoff < 0.7 else 6)
    elif tol <= 2e-4:
        order = max(order, 7 if cutoff < 0.5 else (6 if cutoff < 1.0 else 5))
    elif tol <= 5e-4:
        order = max(order, 6 if cutoff < 0.9 else 5)
    else:
        order = max(order, 6 if cutoff < 0.5 else (5 if cutoff < 0.9 else 4))
    return min(max(int(order), 4), 12)


def download_amber_suite(work_dir: pathlib.Path) -> pathlib.Path:
    suite = work_dir / "Amber20_Benchmark_Suite"
    if suite.exists():
        return suite
    url = "https://ambermd.org/Amber20_Benchmark_Suite.tar.gz"
    archive = work_dir / "Amber20_Benchmark_Suite.tar.gz"
    print(f"Downloading {url}", flush=True)
    urllib.request.urlretrieve(url, archive)
    print(f"Extracting {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as handle:
        root = suite.resolve()
        for member in handle.getmembers():
            target = (suite / member.name).resolve()
            if os.path.commonpath((root, target)) != str(root):
                raise RuntimeError(f"Refusing to extract unsafe tar member {member.name}")
        handle.extractall(path=suite)
    return suite


def load_benchmark_system(name: str, cutoff_nm: float, source_root: pathlib.Path, work_dir: pathlib.Path):
    import openmm.app as app
    import openmm.unit as unit

    bench_dir = source_root / "examples" / "benchmarks"
    if name == "pme":
        forcefield = app.ForceField("amber99sb.xml", "tip3p.xml")
        pdb = app.PDBFile(str(bench_dir / "5dfr_solv-cube_equil.pdb"))
        system = forcefield.createSystem(
            pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=cutoff_nm*unit.nanometer,
            constraints=app.HBonds, hydrogenMass=1.5*unit.amu)
        return system, pdb.positions
    if name == "apoa1pme":
        forcefield = app.ForceField("amber14/protein.ff14SB.xml", "amber14/lipid17.xml", "amber14/tip3p.xml")
        pdb = app.PDBFile(str(bench_dir / "apoa1.pdb"))
        system = forcefield.createSystem(
            pdb.topology, nonbondedMethod=app.PME, nonbondedCutoff=cutoff_nm*unit.nanometer,
            constraints=app.HBonds, hydrogenMass=1.5*unit.amu)
        return system, pdb.positions
    if name in AMBER20_NAMES:
        suite = download_amber_suite(work_dir)
        amber_name = AMBER20_NAMES[name]
        prmtop = app.AmberPrmtopFile(str(suite / f"PME/Topologies/{amber_name}.prmtop"))
        inpcrd = app.AmberInpcrdFile(str(suite / f"PME/Coordinates/{amber_name}.inpcrd"))
        system = prmtop.createSystem(
            nonbondedMethod=app.PME, nonbondedCutoff=cutoff_nm*unit.nanometer,
            constraints=app.HBonds)
        if inpcrd.boxVectors is not None:
            system.setDefaultPeriodicBoxVectors(*inpcrd.boxVectors)
        return system, inpcrd.positions
    raise ValueError(f"Unknown system {name!r}")


def make_coulomb_only(system, cutoff_nm: float, tolerance: float, method: str):
    import openmm as mm
    import openmm.unit as unit

    nonbonded = None
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            nonbonded = force
            force.setForceGroup(0)
        else:
            force.setForceGroup(1)
    if nonbonded is None:
        raise RuntimeError("System does not contain a NonbondedForce")

    nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(cutoff_nm*unit.nanometer)
    nonbonded.setEwaldErrorTolerance(tolerance)
    nonbonded.setUseDispersionCorrection(False)
    nonbonded.setReciprocalSpaceForceGroup(0)
    if hasattr(nonbonded, "setReciprocalSpaceKernelType"):
        kernel = mm.NonbondedForce.ESPKernel if method == "ESP" else mm.NonbondedForce.PMEKernel
        nonbonded.setReciprocalSpaceKernelType(kernel)

    for i in range(nonbonded.getNumParticles()):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        nonbonded.setParticleParameters(i, charge, sigma, 0.0*epsilon.unit)
    for i in range(nonbonded.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nonbonded.getExceptionParameters(i)
        nonbonded.setExceptionParameters(i, p1, p2, charge_prod, sigma, 0.0*epsilon.unit)
    return nonbonded


def make_double_platform(platform_name: str, device: str | None, use_cpu_pme: bool):
    import openmm as mm

    platform = mm.Platform.getPlatformByName(platform_name)
    properties = {}
    if platform_name in ("CUDA", "OpenCL", "HIP"):
        properties["Precision"] = "double"
    if platform_name in ("CUDA", "HIP"):
        properties["UseCpuPme"] = "true" if use_cpu_pme else "false"
    if device is not None and platform_name in ("CUDA", "OpenCL", "HIP"):
        properties["DeviceIndex"] = device
    return platform, properties


def evaluate(system, positions, platform, properties):
    import openmm as mm
    import openmm.unit as unit

    integrator = mm.VerletIntegrator(0.001*unit.picoseconds)
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True, groups={0})
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
    nonbonded = next(force for force in system.getForces() if isinstance(force, mm.NonbondedForce))
    alpha, nx, ny, nz = nonbonded.getPMEParametersInContext(context)
    del context, integrator
    return energy, forces, float(alpha), int(nx), int(ny), int(nz)


def force_metrics(test_forces, ref_forces, test_energy: float, ref_energy: float) -> dict[str, float]:
    diff2 = ref2 = sum_abs_diff = sum_abs_ref = sum_particle_rel = max_abs_diff = 0.0
    count = 0
    particle_rel_count = 0
    for f, r in zip(test_forces, ref_forces):
        dx, dy, dz = f[0] - r[0], f[1] - r[1], f[2] - r[2]
        local_diff2 = dx*dx + dy*dy + dz*dz
        local_ref2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2]
        abs_diff = math.sqrt(local_diff2)
        abs_ref = math.sqrt(local_ref2)
        diff2 += local_diff2
        ref2 += local_ref2
        sum_abs_diff += abs_diff
        sum_abs_ref += abs_ref
        if abs_ref > 0.0:
            sum_particle_rel += abs_diff/abs_ref
            particle_rel_count += 1
        max_abs_diff = max(max_abs_diff, abs_diff)
        count += 1
    return {
        "energy_rel": abs(test_energy - ref_energy)/max(abs(ref_energy), 1.0),
        "force_rel_l2": math.sqrt(diff2/ref2),
        "force_mean_rel": sum_abs_diff/sum_abs_ref,
        "force_mean_particle_rel": sum_particle_rel/particle_rel_count if particle_rel_count else 0.0,
        "force_rms_abs": math.sqrt(diff2/count),
        "force_mean_abs": sum_abs_diff/count,
        "force_max_abs": max_abs_diff,
    }


def make_reference(system_name: str, cutoff: float, tolerance: float, source_root: pathlib.Path,
                   work_dir: pathlib.Path, platform, properties) -> dict:
    system, positions = load_benchmark_system(system_name, cutoff, source_root, work_dir)
    make_coulomb_only(system, cutoff, tolerance, "PME")
    energy, forces, alpha, nx, ny, nz = evaluate(system, positions, platform, properties)
    return {
        "energy": energy,
        "forces": forces,
        "alpha": alpha,
        "grid": (nx, ny, nz),
        "num_particles": system.getNumParticles(),
    }


def run_case(system_name: str, method: str, cutoff: float, tolerance: float, reference: dict,
             source_root: pathlib.Path, work_dir: pathlib.Path, platform, properties) -> dict:
    system, positions = load_benchmark_system(system_name, cutoff, source_root, work_dir)
    make_coulomb_only(system, cutoff, tolerance, method)
    energy, forces, alpha, nx, ny, nz = evaluate(system, positions, platform, properties)
    row = {
        "system": system_name,
        "system_label": SYSTEM_LABELS.get(system_name, system_name),
        "method": method,
        "cutoff_nm": cutoff,
        "tolerance": tolerance,
        "alpha": alpha,
        "grid_x": nx,
        "grid_y": ny,
        "grid_z": nz,
        "num_particles": system.getNumParticles(),
        "energy_kj_mol": energy,
        "order": estimate_esp_order(tolerance, cutoff) if method == "ESP" else 5,
    }
    row.update(force_metrics(forces, reference["forces"], energy, reference["energy"]))
    row["force_ratio"] = row["force_rel_l2"]/tolerance
    row["force_mean_particle_ratio"] = row["force_mean_particle_rel"]/tolerance
    return row


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: pathlib.Path, data) -> None:
    def convert(obj):
        if isinstance(obj, pathlib.Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.write_text(json.dumps(data, indent=2, default=convert))


def fmt_tol(tol: float) -> str:
    return f"{tol:.0e}".replace("e-0", "e-").replace("e+0", "e")


def row_key(row: dict) -> tuple[float, float]:
    return row["cutoff_nm"], row["tolerance"]


def summarize(rows: list[dict], cutoffs: tuple[float, ...], tolerances: tuple[float, ...]) -> None:
    print("\nSummary:\n")
    print("  global L2 matches OpenMM's Ewald tolerance tests; mean particle is a small-force-sensitive diagnostic.\n")
    methods = tuple(method for method in DEFAULT_METHODS if any(row["method"] == method for row in rows))
    summary_metrics = (
        ("global L2", "force_ratio"),
        ("mean particle", "force_mean_particle_ratio"),
    )
    for label, metric in summary_metrics:
        print(f"  {label}:")
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            bad = sum(row[metric] > 1.5 for row in method_rows)
            worst = max(method_rows, key=lambda row: row[metric])
            print(f"    {method} rows {len(method_rows)}, rows > 1.5x tol: {bad}")
            print(
                f"    {method} worst: {worst['system_label']} rc={worst['cutoff_nm']:g} "
                f"tol={fmt_tol(worst['tolerance'])} ratio={worst[metric]:.6f}")
        print()

    by_method_key = {(row["method"],) + row_key(row): row for row in rows}
    if "ESP" in methods:
        print("P / Order Table\n")
        print("  rc    ESP:" + " ".join(f"{fmt_tol(tol):>4}" for tol in tolerances) + "    PME")
        for cutoff in cutoffs:
            esp_orders = [by_method_key[("ESP", cutoff, tol)]["order"] for tol in tolerances]
            print(f"  {cutoff:4.2f}   " + " ".join(f"{order:4d}" for order in esp_orders) + "        5")

    for method in methods:
        print(f"\nWorst Force Error Ratio, Global L2: {method}\n")
        print("  rc   " + " ".join(f"{fmt_tol(tol):>6}" for tol in tolerances))
        for cutoff in cutoffs:
            ratios = [by_method_key[(method, cutoff, tol)]["force_ratio"] for tol in tolerances]
            print(f"  {cutoff:4.2f} " + " ".join(f"{ratio:6.3f}" for ratio in ratios))

    for method in methods:
        print(f"\nWorst Force Error Ratio, Mean Particle: {method}\n")
        print("  rc   " + " ".join(f"{fmt_tol(tol):>6}" for tol in tolerances))
        for cutoff in cutoffs:
            ratios = [by_method_key[(method, cutoff, tol)]["force_mean_particle_ratio"] for tol in tolerances]
            print(f"  {cutoff:4.2f} " + " ".join(f"{ratio:6.3f}" for ratio in ratios))


def run(args) -> None:
    import openmm as mm

    source_root = pathlib.Path(args.source_root).resolve()
    out_dir = pathlib.Path(args.output_dir).resolve() if args.output_dir else (
        pathlib.Path.home() / "ceph" / "tmp" / "openmm-esp-accuracy-sweep" / timestamp())
    work_dir = pathlib.Path(args.work_dir).resolve() if args.work_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    systems = csv_items(args.systems)
    cutoffs = parse_floats(args.cutoffs, "cutoffs")
    tolerances = parse_floats(args.tolerances, "tolerances")
    methods = parse_methods(args.methods)
    reference_settings = parse_reference_settings(args.reference_settings)
    unknown_systems = sorted(set(systems) - set(SYSTEM_LABELS))
    if unknown_systems:
        raise ValueError(f"Unknown systems: {', '.join(unknown_systems)}")
    unused_references = sorted(set(reference_settings) - set(systems))
    if unused_references:
        raise ValueError(f"Unused reference settings: {', '.join(unused_references)}")

    platform, properties = make_double_platform(args.platform, args.device, args.use_cpu_pme)
    manifest = {
        "created_utc": timestamp(),
        "hostname": socket.gethostname(),
        "source_root": source_root,
        "git_revision": git_revision(source_root),
        "openmm_version": mm.version.version,
        "systems": systems,
        "cutoffs_nm": cutoffs,
        "tolerances": tolerances,
        "methods": methods,
        "precision": "double",
        "platform": args.platform,
        "platform_properties": properties,
        "reference_cutoff_nm_default": args.reference_cutoff,
        "reference_tolerance_default": args.reference_tolerance,
        "reference_settings": reference_settings,
    }
    write_json(out_dir / "manifest.json", manifest)

    references = {}
    reference_rows = []
    for system_name in systems:
        ref_cutoff, ref_tol = reference_settings.get(
            system_name, (args.reference_cutoff, args.reference_tolerance))
        print(f"Reference {system_name}: PME rc={ref_cutoff:g} tol={ref_tol:g}", flush=True)
        reference = make_reference(system_name, ref_cutoff, ref_tol, source_root, work_dir, platform, properties)
        references[system_name] = reference
        reference_rows.append({
            "system": system_name,
            "system_label": SYSTEM_LABELS.get(system_name, system_name),
            "cutoff_nm": ref_cutoff,
            "tolerance": ref_tol,
            "alpha": reference["alpha"],
            "grid_x": reference["grid"][0],
            "grid_y": reference["grid"][1],
            "grid_z": reference["grid"][2],
            "num_particles": reference["num_particles"],
            "energy_kj_mol": reference["energy"],
        })
    write_json(out_dir / "references.json", reference_rows)

    rows = []
    for system_name in systems:
        for cutoff in cutoffs:
            for tol in tolerances:
                for method in methods:
                    print(f"{system_name} {method} rc={cutoff:g} tol={tol:g}", flush=True)
                    row = run_case(system_name, method, cutoff, tol, references[system_name],
                                   source_root, work_dir, platform, properties)
                    rows.append(row)
                    write_csv(out_dir / "results.csv", rows)
                    write_json(out_dir / "results.json", rows)

    print(f"\nWrote {len(rows)} rows to {out_dir}", flush=True)
    summarize(rows, cutoffs, tolerances)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", default=str(source_root_from_script()), help="OpenMM source root")
    parser.add_argument("--output-dir", default=None, help="directory for results")
    parser.add_argument("--work-dir", default=None, help="directory for downloaded Amber20 inputs")
    parser.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS), help="comma-separated benchmark systems")
    parser.add_argument("--cutoffs", default=",".join(f"{x:.2f}" for x in DEFAULT_CUTOFFS),
                        help="comma-separated cutoffs in nm")
    parser.add_argument("--tolerances", default=",".join(str(x) for x in DEFAULT_TOLERANCES),
                        help="comma-separated tolerances")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="comma-separated methods: ESP,PME")
    parser.add_argument("--reference-cutoff", type=float, default=1.5, help="reference PME cutoff in nm")
    parser.add_argument("--reference-tolerance", type=float, default=5e-7, help="reference PME tolerance")
    parser.add_argument("--reference-settings", default="",
                        help="comma-separated system:cutoff_nm:tolerance reference overrides")
    parser.add_argument("--platform", default="CUDA", help="OpenMM platform")
    parser.add_argument("--device", default=None, help="CUDA/HIP/OpenCL device index")
    parser.add_argument("--use-cpu-pme", action="store_true", help="allow CPU PME on CUDA/HIP")
    args = parser.parse_args(argv)
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
