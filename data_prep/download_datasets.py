#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

MITDB_URL = "https://physionet.org/files/mitdb/1.0.0"
MITDB_DAT_BYTES = 1_950_000

GB = 1 << 30

DATASETS = {
    "ppg-dalia": ("https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip",
                  "ppg_dalia.zip", 2 * GB, False),
    "wesad": ("https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download",
              "wesad.zip", 2 * GB, True),
}

def _ssl_contexts(insecure=False):
    import ssl
    ctxs = []
    try:
        ctxs.append(ssl.create_default_context())
    except Exception:
        pass
    try:
        import certifi
        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    if insecure:
        ctx = ssl._create_unverified_context()
        print("  !! --insecure: HTTPS certificate verification DISABLED "
              "(file integrity is still CRC-checked after download)")
        ctxs.append(ctx)
    return ctxs or [None]

_SSL_CTXS = [None]
_SSL_INIT = False

def _get_ctx(i):
    global _SSL_CTXS, _SSL_INIT
    if not _SSL_INIT:
        _SSL_CTXS = _ssl_contexts(INSECURE)
        _SSL_INIT = True
    return _SSL_CTXS[i % len(_SSL_CTXS)]

INSECURE = False

def http_get(url, start_byte=0, timeout=60, ctx=None):
    req = urllib.request.Request(url, headers={
        "User-Agent": "pulseware-repro/1.3",
        **({"Range": f"bytes={start_byte}-"} if start_byte else {}),
    })
    return urllib.request.urlopen(req, timeout=timeout, context=ctx)

def curl_fallback(url, dest, min_bytes, resume=True):
    exe = shutil.which("curl")
    if not exe:
        return False
    print(f"  Python certificate paths failed - falling back to {exe} "
          f"(uses the Windows certificate store) ...")
    cmd = [exe, "-sSL", "--retry", "5", "--retry-delay", "3", "-o", dest, url]
    if resume and os.path.exists(dest):
        cmd.insert(1, "-C")
        cmd.insert(2, "-")
    r = subprocess.run(cmd)
    return os.path.exists(dest) and os.path.getsize(dest) >= min_bytes

def download_file(url, dest, min_bytes=1, tries=10, resume=True):
    ssl_failures = 0
    for attempt in range(1, tries + 1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have >= min_bytes:
            return True
        try:
            with http_get(url, start_byte=have, ctx=_get_ctx(attempt - 1)) as r:
                status = getattr(r, "status", 200)
                cl = r.headers.get("Content-Length")
                if status == 206 and have > 0:
                    mode, base = "ab", have
                    total = base + int(cl) if cl else -1
                else:
                    mode, base = "wb", 0
                    total = int(cl) if cl else -1
                with open(dest, mode) as f:
                    t0, last = time.time(), 0
                    while True:
                        chunk = r.read(1 << 18)
                        if not chunk:
                            break
                        f.write(chunk)
                        done = base + f.tell()
                        if f.tell() - last >= (1 << 25):
                            last = f.tell()
                            if total > 0:
                                print(f"\r  {os.path.basename(dest)}: {100*done/total:5.1f}% "
                                      f"({done//(1<<20)} / {total//(1<<20)} MB)", end="", flush=True)
                            else:
                                rate = f.tell() / max(1e-9, time.time() - t0) / (1 << 20)
                                print(f"\r  {os.path.basename(dest)}: {done//(1<<20)} MB "
                                      f"({rate:.1f} MB/s)", end="", flush=True)
            print()
            if os.path.getsize(dest) >= min_bytes:
                return True
        except Exception as e:
            msg = str(e)
            print(f"  attempt {attempt} failed ({e}); retrying in 3 s ...")
            if "CERTIFICATE" in msg.upper() or "SSL" in msg.upper():
                ssl_failures += 1
            time.sleep(3)

        if ssl_failures >= 2 and curl_fallback(url, dest, min_bytes, resume=resume):
            return True
    return curl_fallback(url, dest, min_bytes, resume=resume) or \
        (os.path.exists(dest) and os.path.getsize(dest) >= min_bytes)

def download_file_norange(url, dest, tries=10):
    part = dest + ".part"
    for attempt in range(1, tries + 1):
        try:
            with http_get(url) as r, open(part, "wb") as f:
                t0, last = time.time(), 0
                while True:
                    chunk = r.read(1 << 18)
                    if not chunk:
                        break
                    f.write(chunk)
                    if f.tell() - last >= (1 << 25):
                        last = f.tell()
                        rate = f.tell() / max(1e-9, time.time() - t0) / (1 << 20)
                        print(f"\r  {os.path.basename(dest)}: {f.tell()//(1<<20)} MB "
                              f"({rate:.1f} MB/s)", end="", flush=True)
            print()
            os.replace(part, dest)
            return True
        except Exception as e:
            print(f"  attempt {attempt} failed ({e}); restarting download in 3 s ...")
            time.sleep(3)
    if os.path.exists(part):
        os.remove(part)
    return False

def validate_zip(path, min_bytes=1):
    if not os.path.exists(path):
        return False, "file missing"
    if os.path.getsize(path) < min_bytes:
        return False, f"too small ({os.path.getsize(path)//(1<<20)} MB; expected >= {min_bytes//(1<<20)} MB)"
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad is not None:
                return False, f"corrupt member: {bad}"
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def download_mitdb(out_dir, workers=8):
    d = os.path.join(out_dir, "mitdb")
    os.makedirs(d, exist_ok=True)
    rec_path = os.path.join(d, "RECORDS")
    if not download_file(f"{MITDB_URL}/RECORDS", rec_path):
        sys.exit("Could not download RECORDS - check your internet connection.")
    records = [l.strip() for l in open(rec_path) if l.strip()]
    jobs = [f"{r}.{ext}" for r in records for ext in ("dat", "hea", "atr")]
    print(f"mitdb: {len(records)} records, {len(jobs)} files -> {d}")

    def get(name):
        dest = os.path.join(d, name)
        min_bytes = MITDB_DAT_BYTES if name.endswith(".dat") else 50
        return download_file(f"{MITDB_URL}/{name}", dest, min_bytes=min_bytes)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(get, jobs))
    bad = [j for j, r in zip(jobs, results) if not r]
    if bad:
        print(f"WARNING: {len(bad)} files incomplete after retries: {bad[:8]} ...")
        print("Re-run this script - it resumes where it left off.")
        return False
    bad = [f"{r}.dat" for r in records
           if os.path.getsize(os.path.join(d, f"{r}.dat")) != MITDB_DAT_BYTES]
    if bad:
        print(f"WARNING: wrong-size .dat files: {bad}")
        return False
    try:
        import wfdb
        for r in records[:5] + records[-5:]:
            wfdb.rdrecord(os.path.join(d, r), sampto=10)
        print("wfdb spot-check passed (10 records readable).")
    except ImportError:
        print("(pip install wfdb to enable read-validation; size checks passed.)")
    print(f"mitdb OK: {len(records)} records verified in {d}")
    return True

def download_big_dataset(key, out_dir, extract=False):
    url, fname, min_bytes, range_ok = DATASETS[key]
    dest = os.path.join(out_dir, fname)
    os.makedirs(out_dir, exist_ok=True)

    ok, why = validate_zip(dest, min_bytes)
    if ok:
        print(f"{key}: existing {fname} passed CRC + size check - skipping download.")
    else:
        if os.path.exists(dest):
            print(f"{key}: existing {fname} unusable ({why}) - deleting and re-downloading.")
            os.remove(dest)
        print(f"{key}: downloading {url}")
        if range_ok:
            print("  (resume supported - if the connection drops, re-run and it continues)")
            if not download_file(url, dest, min_bytes=min_bytes):
                print(f"WARNING: {key} download incomplete - run again to resume.")
                return False
        else:
            print("  (no resume support - a dropped connection restarts the file)")
            if not download_file_norange(url, dest):
                print(f"WARNING: {key} download failed after retries - run again to retry.")
                return False
        ok, why = validate_zip(dest, min_bytes)
        if not ok:
            print(f"WARNING: {key} zip failed validation ({why}) - deleted; run again.")
            os.remove(dest)
            return False

    if extract:
        print(f"  extracting {fname} ...")
        with zipfile.ZipFile(dest) as z:
            z.extractall(os.path.join(out_dir, key))
        inner = os.path.join(out_dir, key, "data.zip")
        if os.path.exists(inner):
            with zipfile.ZipFile(inner) as z:
                z.extractall(os.path.join(out_dir, key))
        print(f"  extracted -> {os.path.join(out_dir, key)}")
    print(f"{key} OK -> {dest}")
    return True

def verify_only(out_dir):
    allok = True
    for key, (_, fname, min_bytes, _) in DATASETS.items():
        dest = os.path.join(out_dir, fname)
        ok, why = validate_zip(dest, min_bytes)
        size = os.path.getsize(dest) if os.path.exists(dest) else 0
        print(f"  {fname}: {size//(1<<20)} MB -> {'OK' if ok else 'INVALID (' + why + ')'}")
        allok &= ok
    return allok

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["all", "mitdb", "ppg-dalia", "wesad", "verify"],
                    default="all")
    ap.add_argument("--out", default="../data", help="output directory (default ../data)")
    ap.add_argument("--workers", type=int, default=8, help="parallel mitdb downloads")
    ap.add_argument("--extract", action="store_true", help="extract zips after download")
    ap.add_argument("--insecure", action="store_true",
                    help="last resort: skip HTTPS certificate verification "
                         "(downloads are still CRC-validated)")
    args = ap.parse_args()

    global INSECURE
    INSECURE = args.insecure

    os.makedirs(args.out, exist_ok=True)
    if args.dataset == "verify":
        print("Validating existing zips ...")
        ok = verify_only(args.out)
        print("all zips valid" if ok else "INVALID zips found - re-run the dataset download")
        sys.exit(0 if ok else 1)

    results = {}
    if args.dataset in ("all", "mitdb"):
        results["mitdb"] = download_mitdb(args.out, args.workers)
    if args.dataset in ("all", "ppg-dalia"):
        results["ppg-dalia"] = download_big_dataset("ppg-dalia", args.out, args.extract)
    if args.dataset in ("all", "wesad"):
        results["wesad"] = download_big_dataset("wesad", args.out, args.extract)

    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k:10s} {'OK' if v else 'INCOMPLETE'}")
    if results.get("mitdb"):
        print("\nNext step (Tier 2 features):")
        print(f"  python build_mitbih_features.py --dl-dir {os.path.join(args.out, 'mitdb')} "
              f"--out {os.path.join(args.out, 'mitbih_features.csv')} --window 10 --stride 10")
    if not all(results.values()):
        sys.exit(1)

if __name__ == "__main__":
    main()
