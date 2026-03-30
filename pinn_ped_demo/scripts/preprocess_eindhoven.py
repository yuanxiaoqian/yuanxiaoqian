import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False, default="data/raw")
    parser.add_argument("--output", required=False, default="data/processed/eindhoven_placeholder.txt")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("placeholder: add parquet preprocessing here\n")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
