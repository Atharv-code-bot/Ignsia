import os, sys, argparse, subprocess

DIRS = {
    "geotiffs":  "./ward_outputs/geotiffs/",
    "stage2":    "./ward_outputs/stage2/",
    "stage3":    "./ward_outputs/stage3/",
    "stage4_5":  "./ward_outputs/stage4_5/",
    "cog":       "./ward_outputs/cog/",
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)


def run(cmd: str, label: str):
    print(f"\n{'─'*65}")
    print(f"  ▶  {label}")
    print(f"{'─'*65}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"\n[ERROR] Stage failed: {label}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="TEORA v2 Pipeline Runner")
    parser.add_argument("--ward", default=None)
    parser.add_argument("--all_wards", action="store_true")
    parser.add_argument("--start_from_stage", type=int, default=1)
    parser.add_argument("--start_date", default="2024-01-01")
    parser.add_argument("--end_date", default="2024-12-31")
    parser.add_argument("--t_max", type=int, default=5000)
    parser.add_argument("--use_segformer", action="store_true")
    parser.add_argument("--run_date", default="2024-12-31")
    args = parser.parse_args()

    scope = "--all_wards" if args.all_wards else f"--ward '{args.ward}'"

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  TEORA v2 — Pixel-Wise Pipeline (AUTO MODE)                 ║
║  Scope:      {scope:<46}║
║  Dates:      {args.start_date} → {args.end_date}                     ║
║  Start stage:{args.start_from_stage:<46}║
╚══════════════════════════════════════════════════════════════╝
""")

    if args.start_from_stage <= 1:
        run(
            f"python gee/gee_pixel_acquisition.py {scope} "
            f"--start_date {args.start_date} --end_date {args.end_date}",
            "Stage 1: GEE Pixel Export (async tasks submitted)"
        )

        print("\n[Stage 1] ⏳ GEE tasks submitted.")
        print("👉 Go to: https://code.earthengine.google.com/ → Tasks tab")
        print("👉 Wait until ALL tasks show COMPLETED")
        print(f"👉 Download all GeoTIFFs to: {DIRS['geotiffs']}")
        print("⚠️ After downloading, rerun pipeline with --start_from_stage 2\n")

        return  

    if args.start_from_stage <= 2:
        run(
            f"python local_processing/stage2_env_layers.py {scope} "
            f"--geotiff_dir {DIRS['geotiffs']} --out_dir {DIRS['stage2']}",
            "Stage 2: TsHARP LST Sharpening + Env Stack Assembly"
        )

    if args.start_from_stage <= 3:
        seg_flag = "--use_segformer" if args.use_segformer else ""
        run(
            f"python local_processing/stage3_segmentation.py {scope} "
            f"--stage2_dir {DIRS['stage2']} --out_dir {DIRS['stage3']} {seg_flag}",
            f"Stage 3: Land Cover Segmentation ({'SegFormer' if args.use_segformer else 'threshold MVP'})"
        )

    if args.start_from_stage <= 4:
        run(
            f"python local_processing/stage4_5_scoring_optimizer.py "
            f"--stage2_dir {DIRS['stage2']} --stage3_dir {DIRS['stage3']} "
            f"--aux_dir {DIRS['geotiffs']} --out_dir {DIRS['stage4_5']} "
            f"--t_max {args.t_max}",
            "Stage 4+5: TPIS Scoring + Knapsack Optimization"
        )

    print("\n[Stage 6] 🚫 Skipped (Database not configured)")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅ TEORA Pipeline Complete (File Mode)                     ║
╠══════════════════════════════════════════════════════════════╣
║  Outputs:                                                   ║
║    ward_tpis_scores.csv → {DIRS['stage4_5']:30} ║
║    seg_rgba_*.tif       → {DIRS['stage3']:30} ║
║    tpis_raster_*.tif    → {DIRS['stage4_5']:30} ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()