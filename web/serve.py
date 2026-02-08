import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class NoCacheHandler(SimpleHTTPRequestHandler):
    # Avoid confusing "works after hard refresh" issues while iterating on ESM modules.
    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> int:
    here = Path(__file__).resolve()
    root = here.parents[1]  # project root containing /src, /test, /web

    port = 8000
    if len(sys.argv) >= 2:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            return 2

    os.chdir(root)

    # Make failures obvious instead of serving confusing 404s.
    if not (root / "web" / "index.html").is_file():
      print("ERROR: Missing web/index.html under project root.")
      print(f"  root={root}")
      return 2
    if not (root / "src" / "index.js").is_file():
      print("ERROR: Missing src/index.js under project root.")
      print(f"  root={root}")
      return 2
    if not (root / "test" / "complete.suite.js").is_file():
      print("ERROR: Missing test/complete.suite.js under project root.")
      print(f"  root={root}")
      return 2

    httpd = ThreadingHTTPServer(("0.0.0.0", port), NoCacheHandler)
    print(f"Serving {root}")
    print(f"  Open: http://localhost:{port}/  (redirects to /web/)")
    print(f"  Or:   http://localhost:{port}/web/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

