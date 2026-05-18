import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

import generate as standard_generator
import curved_generate as curved_generator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ASSETS_ROOT = os.path.join(BASE_DIR, "bin")

# Ensure both engines share the same output target so files land together.
standard_generator.OUTPUT_DIR = OUTPUT_DIR
curved_generator.OUTPUT_DIR = OUTPUT_DIR

OUTPUT_SUFFIXES = ("-1.png", "-2.png", "-3.png")


@dataclass
class RowJob:
    index: int
    row: pd.Series
    order: curved_generator.JerseyOrder
    team_folder: str
    coords: Dict
    use_curved: bool
    output_stub: str


def select_csv_path() -> Optional[str]:
    path = standard_generator.get_csv_path()
    if not path:
        print("[ERROR] No CSV file selected. Exiting.")
    return path


def read_input_csv(csv_path: str) -> pd.DataFrame:
    df = standard_generator.read_csv_with_fallback(csv_path)
    return df.dropna(how="all")


def prepare_output_dir() -> None:
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def requires_curved_pipeline(coords: Dict) -> bool:
    nameplate = coords.get("NamePlate") or {}
    curve_value = nameplate.get("curve") if isinstance(nameplate, dict) else None
    has_curve = curve_value is not None

    border_keys = ("NumberBorder", "FrontNumberBorder", "BackNumberBorder")
    has_number_border = any(coords.get(key) is not None for key in border_keys)

    return bool(has_curve and has_number_border)


def resolve_output_stub(row: pd.Series) -> str:
    raw_value = row.get("Name", "")
    if pd.isna(raw_value):
        raw_value = ""
    raw_name = str(raw_value)
    if raw_name:
        stub = curved_generator.sanitize_filename_component(raw_name)
    else:
        fallback = "-".join(
            str(row.get(label, "") or "").strip()
            for label in ("Team", "Color List", "Jersey Characters")
        )
        stub = curved_generator.sanitize_filename_component(fallback)
    return stub or "jersey"


def build_job(index: int, row: pd.Series) -> Optional[RowJob]:
    try:
        order = curved_generator.build_order(row)
    except ValueError as exc:
        print(f"[WARN] Skipping row {index}: {exc}")
        return None

    try:
        team_folder = curved_generator.locate_team_folder(order)
    except FileNotFoundError as exc:
        print(f"[WARN] Skipping row {index}: {exc}")
        return None

    try:
        coords = curved_generator.load_coords_json(team_folder)
    except Exception as exc:
        print(f"[WARN] Skipping row {index}: unable to load coords.json - {exc}")
        return None

    use_curved = requires_curved_pipeline(coords)
    output_stub = resolve_output_stub(row)
    return RowJob(
        index=index,
        row=row,
        order=order,
        team_folder=team_folder,
        coords=coords,
        use_curved=use_curved,
        output_stub=output_stub,
    )


def collect_jobs(df: pd.DataFrame) -> List[RowJob]:
    jobs: List[RowJob] = []
    for idx, row in df.iterrows():
        job = build_job(idx, row)
        if job:
            jobs.append(job)
    return jobs


def is_youth_row(row: pd.Series) -> bool:
    value = str(row.get("Mens or Youth", "")).strip().lower()
    return value == "youth"


def standardize_output_names(source_prefix: str, target_prefix: str) -> None:
    source_prefix = "" if source_prefix is None else str(source_prefix)
    target_prefix = "" if target_prefix is None else str(target_prefix)
    if source_prefix == target_prefix:
        return
    for suffix in OUTPUT_SUFFIXES:
        src = os.path.join(OUTPUT_DIR, f"{source_prefix}{suffix}")
        if not os.path.exists(src):
            continue
        dst = os.path.join(OUTPUT_DIR, f"{target_prefix}{suffix}")
        os.replace(src, dst)


def apply_youth_overlay_to_outputs(prefix: str, youth_overlay) -> None:
    if youth_overlay is None:
        return
    for suffix in OUTPUT_SUFFIXES:
        target_path = os.path.join(OUTPUT_DIR, f"{prefix}{suffix}")
        if os.path.exists(target_path):
            standard_generator.apply_overlay_to_file(target_path, youth_overlay)


def run_standard_job(job: RowJob, youth_overlay) -> None:
    name_value = job.row.get("Name", "")
    if pd.isna(name_value):
        name_value = ""
    source_prefix = str(name_value)
    standard_generator.process_front(job.row, job.team_folder, job.coords)
    standard_generator.process_back(job.row, job.team_folder, job.coords)
    front_path = os.path.join(OUTPUT_DIR, f"{source_prefix}-3.png")
    back_path = os.path.join(OUTPUT_DIR, f"{source_prefix}-2.png")
    standard_generator.process_combo(job.row, front_path, back_path)
    standardize_output_names(source_prefix, job.output_stub)
    if is_youth_row(job.row):
        apply_youth_overlay_to_outputs(job.output_stub, youth_overlay)
    print(f"✓ Row {job.index}: standard generator ({job.output_stub})")


def run_curved_job(job: RowJob, youth_overlay) -> None:
    source_prefix = job.order.file_stub
    curved_generator.process_order(job.order, youth_overlay)
    standardize_output_names(source_prefix, job.output_stub)
    print(f"✓ Row {job.index}: curved generator ({job.output_stub})")


def process_job(job: RowJob, youth_overlay) -> Tuple[str, bool]:
    try:
        if job.use_curved:
            run_curved_job(job, youth_overlay)
            return ("curved", True)
        run_standard_job(job, youth_overlay)
        return ("standard", True)
    except Exception as exc:
        pipeline = "curved" if job.use_curved else "standard"
        print(f"[ERROR] Row {job.index}: {pipeline} generator failed - {exc}")
        return (pipeline, False)


def main() -> None:
    csv_path = select_csv_path()
    if not csv_path:
        return

    df = read_input_csv(csv_path)
    jobs = collect_jobs(df)
    if not jobs:
        print("[ERROR] No valid rows to process. Exiting.")
        return

    prepare_output_dir()
    youth_overlay = curved_generator.load_youth_overlay()
    if youth_overlay is None:
        overlay_path = os.path.join(ASSETS_ROOT, "youth.png")
        if os.path.exists(overlay_path):
            try:
                from PIL import Image

                youth_overlay = Image.open(overlay_path).convert("RGBA")
            except Exception as exc:
                print(f"[WARN] Unable to load youth overlay from fallback path {overlay_path}: {exc}")

    max_workers = min(32, max(4, (os.cpu_count() or 4)))
    stats = {"standard": 0, "curved": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_job, job, youth_overlay): job for job in jobs}
        for future in as_completed(futures):
            pipeline, success = future.result()
            if success:
                stats[pipeline] += 1
            else:
                stats["error"] += 1

    print("\nRun complete:")
    print(f"  Standard generator rows: {stats['standard']}")
    print(f"  Curved generator rows:   {stats['curved']}")
    if stats["error"]:
        print(f"  Failed rows:             {stats['error']}")


if __name__ == "__main__":
    main()
