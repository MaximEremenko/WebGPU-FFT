#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class HarnessHandler(SimpleHTTPRequestHandler):
    result_event = threading.Event()
    result_payload: dict | None = None

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/__autorun_result":
            self.send_error(404)
            return

        length_raw = self.headers.get("content-length", "0")
        try:
            length = int(length_raw)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            payload = {
                "kind": "unknown",
                "ok": False,
                "pass": 0,
                "fail": 1,
                "skip": 0,
                "total": 1,
                "ms": 0,
                "error": "invalid callback payload",
            }

        HarnessHandler.result_payload = payload
        HarnessHandler.result_event.set()

        self.send_response(204)
        self.end_headers()


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def reset_result() -> None:
    HarnessHandler.result_payload = None
    HarnessHandler.result_event.clear()


def resolve_chrome_path(hint: str | None) -> str | None:
    if hint:
        p = Path(hint)
        if p.is_file():
            return str(p)
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def wait_server(url: str, timeout_sec: float) -> bool:
    end = time.time() + timeout_sec
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if 200 <= resp.status < 500:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def parse_dom_marker(dom: str) -> dict | None:
    m = re.search(r'data-webgpufft-autorun="([^"]+)"', dom)
    if not m:
        return None
    try:
        payload = urllib.parse.unquote(m.group(1))
        return json.loads(payload)
    except Exception:
        return None


def chrome_common_args(user_data_dir: str) -> list[str]:
    return [
        "--no-sandbox",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-breakpad",
        "--disable-crash-reporter",
        "--enable-unsafe-webgpu",
        "--disable-gpu-sandbox",
        f"--user-data-dir={user_data_dir}",
    ]


def run_headless(chrome: str, url: str, timeout_sec: int, temp_root: Path) -> tuple[int, str]:
    with tempfile.TemporaryDirectory(
        prefix="webgpufft_chrome_headless_",
        dir=str(temp_root),
        ignore_cleanup_errors=True,
    ) as user_data_dir:
        cmd = [
            chrome,
            *chrome_common_args(user_data_dir),
            "--headless=new",
            f"--virtual-time-budget={timeout_sec * 1000}",
            "--dump-dom",
            url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec + 120)
        dom = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, dom


def _terminate_proc(proc: subprocess.Popen) -> tuple[int | None, str, str]:
    out = ""
    err = ""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                out, err = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, text=True)
                except Exception:
                    proc.kill()
                out, err = proc.communicate(timeout=5)
        else:
            out, err = proc.communicate(timeout=5)
    except Exception:
        pass
    return proc.returncode, out, err


def run_headed_wait_callback(chrome: str, url: str, timeout_sec: int, temp_root: Path) -> tuple[dict | None, int | None, str, str]:
    with tempfile.TemporaryDirectory(
        prefix="webgpufft_chrome_headed_",
        dir=str(temp_root),
        ignore_cleanup_errors=True,
    ) as user_data_dir:
        cmd = [
            chrome,
            *chrome_common_args(user_data_dir),
            f"--app={url}",
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        deadline = time.time() + timeout_sec
        payload = None
        last_progress_sec = -1
        while time.time() < deadline:
            if HarnessHandler.result_event.wait(0.25):
                payload = HarnessHandler.result_payload
                break

            elapsed = int(time.time() - (deadline - timeout_sec))
            if elapsed // 15 != last_progress_sec // 15:
                last_progress_sec = elapsed
                remain = max(0, int(deadline - time.time()))
                print(f"Waiting for autorun callback... elapsed={elapsed}s remaining={remain}s")

            # On Windows, the launcher process can exit while the actual browser child
            # keeps running. Keep waiting for callback on clean launcher exit.
            rc = proc.poll()
            if rc is not None and rc != 0 and not HarnessHandler.result_event.is_set():
                break

        code, out, err = _terminate_proc(proc)
        return payload, code, out, err


def print_result(result: dict) -> None:
    print(
        "AUTORUN {kind}: ok={ok} pass={pass_} fail={fail} skip={skip} total={total} ms={ms:.2f}".format(
            kind=result.get("kind", "?"),
            ok=result.get("ok", False),
            pass_=int(result.get("pass", 0)),
            fail=int(result.get("fail", 0)),
            skip=int(result.get("skip", 0)),
            total=int(result.get("total", 0)),
            ms=float(result.get("ms", 0.0)),
        )
    )
    adapter = result.get("adapter") or {}
    if adapter:
        print(
            "Adapter: selection={sel} limitsRequestMode={lrm} matchRtx5090={match} matchNvidiaBlackwell={blk} acceptedByPolicy={acc} infoAvailable={avail} summary={summary}".format(
                sel=adapter.get("selection", "?"),
                lrm=adapter.get("limitsRequestMode", "?"),
                match=adapter.get("isNvidia5090", "?"),
                blk=adapter.get("isNvidiaBlackwell", "?"),
                acc=adapter.get("acceptedByRtx5090Policy", "?"),
                avail=adapter.get("infoAvailable", "?"),
                summary=adapter.get("summary", "(unavailable)"),
            )
        )
    names = result.get("failureNames") or []
    if names:
        print("Failed tests:")
        for n in names:
            print(f"  - {n}")
    failures = result.get("failures") or []
    if failures:
        print("Failure details:")
        for f in failures:
            name = f.get("name", "?")
            msg = f.get("message", "")
            print(f"  - {name}: {msg}")
    benches = result.get("benches") or []
    if benches:
        print("Bench details:")
        for b in benches:
            name = b.get("name", "?")
            if b.get("skipped"):
                reason = b.get("reason", "")
                print(f"  - SKIP {name}: {reason}")
                continue
            parts: list[str] = []
            if "avgMs" in b:
                parts.append(f"avg={float(b['avgMs']):.2f} ms")
            if "totalMs" in b:
                parts.append(f"total={float(b['totalMs']):.2f} ms")
            if "gflops" in b:
                parts.append(f"gflops={float(b['gflops']):.2f}")
            if "transpose" in b:
                parts.append(f"transpose={bool(b['transpose'])}")
            if "mode" in b:
                parts.append(f"mode={b['mode']}")
            if "proxy" in b:
                parts.append(f"proxy={bool(b['proxy'])}")
            if "tileN" in b:
                parts.append(f"tile={int(b['tileN'])}")
            if "tiles" in b:
                parts.append(f"tiles={int(b['tiles'])}")
            if "sweeps" in b:
                parts.append(f"sweeps={int(b['sweeps'])}")
            if "requestedBytes" in b:
                parts.append(f"requestedBytes={int(b['requestedBytes'])}")
            if "maxBufferSize" in b:
                parts.append(f"maxBufferSize={int(b['maxBufferSize'])}")
            if "forcedMaxBind" in b:
                parts.append(f"forcedMaxBind={int(b['forcedMaxBind'])}")
            if "outOfCore" in b:
                parts.append(f"outOfCore={bool(b['outOfCore'])}")
            if "numAxisUploads" in b and b["numAxisUploads"] is not None:
                parts.append(f"numAxisUploads={int(b['numAxisUploads'])}")
            if "linesPerChunk" in b and b["linesPerChunk"] is not None:
                parts.append(f"linesPerChunk={int(b['linesPerChunk'])}")
            if "r2cUploads" in b and b["r2cUploads"] is not None:
                parts.append(f"r2cUploads={int(b['r2cUploads'])}")
            if "r2cChunkLines" in b and b["r2cChunkLines"] is not None:
                parts.append(f"r2cChunkLines={int(b['r2cChunkLines'])}")
            if "c2rUploads" in b and b["c2rUploads"] is not None:
                parts.append(f"c2rUploads={int(b['c2rUploads'])}")
            if "c2rChunkLines" in b and b["c2rChunkLines"] is not None:
                parts.append(f"c2rChunkLines={int(b['c2rChunkLines'])}")
            if "axis0ChunkLines" in b:
                parts.append(f"axis0ChunkLines={int(b['axis0ChunkLines'])}")
            if "axis0ChunkBytes" in b and b["axis0ChunkBytes"] is not None:
                parts.append(f"axis0ChunkBytes={int(b['axis0ChunkBytes'])}")
            if "axis2MaxZChunk" in b and b["axis2MaxZChunk"] is not None:
                parts.append(f"axis2MaxZChunk={int(b['axis2MaxZChunk'])}")
            if "axis2ChunkSpanBytes" in b and b["axis2ChunkSpanBytes"] is not None:
                parts.append(f"axis2ChunkSpanBytes={int(b['axis2ChunkSpanBytes'])}")
            if "segmentBytes" in b and b["segmentBytes"] is not None:
                parts.append(f"segmentBytes={int(b['segmentBytes'])}")
            if "segmentCount" in b and b["segmentCount"] is not None:
                parts.append(f"segmentCount={int(b['segmentCount'])}")
            if "ringDepth" in b and b["ringDepth"] is not None:
                parts.append(f"ringDepth={int(b['ringDepth'])}")
            if "axis0ChunkUtilization" in b and b["axis0ChunkUtilization"] is not None:
                parts.append(f"axis0ChunkUtil={float(b['axis0ChunkUtilization']):.3f}")
            if "axis2ChunkUtilization" in b and b["axis2ChunkUtilization"] is not None:
                parts.append(f"axis2ChunkUtil={float(b['axis2ChunkUtilization']):.3f}")
            if "largeMode" in b:
                parts.append(f"largeMode={bool(b['largeMode'])}")
            if "batchSliced" in b:
                parts.append(f"batchSliced={bool(b['batchSliced'])}")
            if "runs" in b:
                parts.append(f"runs={int(b['runs'])}")
            info = " ".join(parts)
            print(f"  - {name}{(': ' + info) if info else ''}")


def result_exit_code(result: dict) -> int:
    ok = bool(result.get("ok", False))
    fail = int(result.get("fail", 0))
    return 0 if ok and fail == 0 else 1


def effective_timeout_seconds(args: argparse.Namespace) -> int:
    timeout_sec = max(1, int(args.timeout))
    if args.mode in {"bench", "all"}:
        floor = 180
        if args.real_large_3d:
            n = max(1, int(args.real_large_3d_n))
            if n >= 1024 and args.real_large_3d_strict:
                floor = 1200
            elif n >= 1024:
                floor = 600
            elif n >= 512:
                floor = 300
        if timeout_sec < floor:
            print(f"Adjusting timeout from {timeout_sec}s to {floor}s for bench run")
            timeout_sec = floor
    return timeout_sec


def main() -> int:
    ap = argparse.ArgumentParser(description="Run webgpufft browser validation with automatic headless/headed fallback.")
    ap.add_argument("--mode", default="tests", choices=["tests", "bench", "all"])
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--timeout", type=int, default=120, help="Per-attempt timeout in seconds")
    ap.add_argument("--chrome-path", default="", help="Explicit path to chrome.exe")
    ap.add_argument("--headed-only", action="store_true", help="Skip headless attempt and use headed Chrome directly")
    ap.add_argument("--require-rtx5090", dest="require_rtx5090", action="store_true", help="Require benchmark adapter to match NVIDIA RTX 5090")
    ap.add_argument("--allow-any-adapter", dest="require_rtx5090", action="store_false", help="Do not enforce NVIDIA RTX 5090 adapter check")
    ap.add_argument("--real-large-3d", action="store_true", help="Enable true segmented full-volume 3D benchmark path for oversized cases")
    ap.add_argument("--real-large-3d-n", type=int, default=1024, help="N for real-large-3d full-volume case (default: 1024)")
    ap.add_argument("--real-large-3d-strict", dest="real_large_3d_strict", action="store_true", help="Do not fall back to tile-proxy for selected real-large-3d N")
    ap.add_argument("--allow-real-large-3d-fallback", dest="real_large_3d_strict", action="store_false", help="Allow tile-proxy fallback if real-large-3d run fails")
    ap.set_defaults(require_rtx5090=True, real_large_3d_strict=True)
    args = ap.parse_args()
    timeout_sec = effective_timeout_seconds(args)

    root = Path(__file__).resolve().parents[1]
    web_dir = root / "web"
    if not (web_dir / "index.html").is_file():
        print(f"ERROR: Missing {web_dir / 'index.html'}")
        return 2

    chrome = resolve_chrome_path(args.chrome_path or None)
    if not chrome:
        print("ERROR: Chrome not found. Use --chrome-path.")
        return 2

    temp_root = root / ".tmp_browser_runner"
    temp_root.mkdir(parents=True, exist_ok=True)

    build_script = web_dir / "build_bundle.py"
    if build_script.is_file():
        build = subprocess.run([sys.executable, str(build_script)], capture_output=True, text=True)
        if build.returncode != 0:
            print("ERROR: Failed to build web/bundle.js")
            print(build.stdout)
            print(build.stderr)
            return 2
        if build.stdout.strip():
            print(build.stdout.strip())

    old_cwd = Path.cwd()
    httpd: ThreadingHTTPServer | None = None
    try:
        os.chdir(root)
        httpd = ThreadingHTTPServer(("127.0.0.1", args.port), HarnessHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        base = f"http://localhost:{args.port}/web/"
        if not wait_server(base, 20.0):
            print(f"ERROR: Harness server did not become ready at {base}")
            return 2

        query = [("autorun", args.mode), ("machine", "1")]
        if args.mode in {"bench", "all"} and args.require_rtx5090:
            query.append(("require_rtx5090", "1"))
        if args.mode in {"bench", "all"} and args.real_large_3d:
            query.append(("real_large3d", "1"))
            query.append(("real_large3d_n", str(max(1, int(args.real_large_3d_n)))))
            query.append(("real_large3d_strict", "1" if args.real_large_3d_strict else "0"))
        url = f"{base}?{urllib.parse.urlencode(query)}"
        print(f"Using Chrome: {chrome}")
        print(f"Running autorun mode={args.mode} timeout={timeout_sec}s url={url}")

        if not args.headed_only:
            print("Attempt 1: headless Chrome...")
            reset_result()
            try:
                code, dom = run_headless(chrome, url, timeout_sec, temp_root)
            except subprocess.TimeoutExpired:
                code, dom = 124, ""

            payload = HarnessHandler.result_payload or parse_dom_marker(dom)
            if payload is not None:
                print_result(payload)
                return result_exit_code(payload)

            print("Headless run did not produce a result callback.")
            if code != 0:
                print(f"Headless Chrome exited with code {code}.")
            if "Status: initializing WebGPU..." in dom:
                print("Hint: WebGPU likely did not initialize in headless mode.")

        print("Attempt 2: headed Chrome app window (fallback)...")
        reset_result()
        payload, code, out, err = run_headed_wait_callback(chrome, url, timeout_sec, temp_root)
        if payload is None:
            print("ERROR: No autorun result received from headed fallback.")
            print(f"Hint: increase --timeout (current effective timeout: {timeout_sec}s) for long real-large-3d runs.")
            if code is not None:
                print(f"Headed Chrome exit code: {code}")
            if err:
                print(err[:3000])
            elif out:
                print(out[:3000])
            return 2

        print_result(payload)
        return result_exit_code(payload)

    finally:
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        os.chdir(old_cwd)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
