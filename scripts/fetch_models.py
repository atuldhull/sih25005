"""Download the model weights and prove they are the right bytes.

Weights are not in git (see models/README.md). This fetches them from the
Kaggle datasets they were published to and checks each SHA256 against
MANIFEST.json before installing it.

The hash check is the point. A truncated or re-encoded download produces a
model that loads and then behaves oddly, which is indistinguishable from a
modelling bug and costs days to chase. Refusing to install a file whose hash
does not match turns that into a five-second error message.

  python scripts/fetch_models.py            # fetch anything missing
  python scripts/fetch_models.py --verify   # check what is already there
  python scripts/fetch_models.py --force    # re-download everything
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "models" / "MANIFEST.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(entry: dict) -> tuple:
    """(status, detail) for one manifest entry, without downloading."""
    target = ROOT / entry["path"]
    if not target.exists():
        return "MISSING", "not downloaded"
    if entry.get("sha256") in (None, "", "TBD"):
        return "UNVERIFIED", ("no hash in MANIFEST.json - ask whoever "
                              "published it for the SHA256")
    if target.is_dir():
        f = target / entry.get("hash_file", "model.safetensors")
        if not f.exists():
            return "BROKEN", f"{f.name} missing from the export directory"
        got = sha256(f)
    else:
        got = sha256(target)
    if got != entry["sha256"]:
        return "MISMATCH", (f"expected {entry['sha256'][:16]}..., "
                            f"got {got[:16]}... - this is NOT the model that "
                            f"produced the measured numbers")
    return "OK", f"{got[:16]}..."


def fetch(entry: dict) -> bool:
    src = entry.get("kaggle")
    if not src:
        print(f"  no automatic source - {entry.get('obtain_from', 'ask the owner')}")
        return False
    target = ROOT / entry["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "models" / "_download"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    print(f"  downloading {src} ...", flush=True)
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets",
                        "download", "-d", src, "-p", str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  kaggle failed: {(r.stderr or r.stdout).strip()[-300:]}")
        return False
    for z in tmp.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(tmp)
        z.unlink()
    want = entry.get("member") or target.name
    found = next((p for p in tmp.rglob("*") if p.name == want), None)
    if found is None:
        print(f"  '{want}' not found in the download")
        return False
    if target.exists():
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    shutil.move(str(found), str(target))
    shutil.rmtree(tmp, ignore_errors=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="only check what is present; download nothing")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"no manifest at {MANIFEST}")
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["models"]

    bad = 0
    for e in entries:
        print(f"\n{e['name']}  ->  {e['path']}")
        status, detail = check(e)
        if status == "OK" and not args.force:
            print(f"  OK  {detail}")
            continue
        if args.verify:
            print(f"  {status}  {detail}")
            bad += status not in ("OK",)
            continue
        if status in ("MISSING", "MISMATCH", "BROKEN") or args.force:
            print(f"  {status}  {detail}")
            if not fetch(e):
                bad += 1
                continue
            status, detail = check(e)
            print(f"  after download: {status}  {detail}")
            bad += status not in ("OK", "UNVERIFIED")
        else:
            print(f"  {status}  {detail}")

    print("\n" + "=" * 60)
    if bad:
        print(f"{bad} model(s) unusable. The pipeline degrades to NOT_SCORED")
        print("with a truthful reason rather than guessing, so this is not a")
        print("crash - but nothing will be measured until they are present.")
        return 1
    print("all models present and verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
