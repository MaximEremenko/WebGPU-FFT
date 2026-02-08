from __future__ import annotations

import re
from pathlib import Path
import posixpath

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST = ROOT / "test"
OUT = ROOT / "web" / "bundle.js"


def to_posix(s: str) -> str:
    return s.replace("\\", "/")


def resolve_spec(from_id: str, spec: str) -> str:
    spec = spec.strip()
    if spec.startswith("."):
        base = posixpath.dirname(from_id)
        return posixpath.normpath(posixpath.join(base, spec))
    return spec


def parse_named_list(text: str):
    parts = [p.strip() for p in text.replace("\n", " ").split(",") if p.strip()]
    pairs = []
    for p in parts:
        m = re.match(r"^([A-Za-z_$][\w$]*)\s+as\s+([A-Za-z_$][\w$]*)$", p)
        if m:
            pairs.append((m.group(1), m.group(2)))
        else:
            pairs.append((p, p))
    items = []
    for exp, loc in pairs:
        items.append(exp if exp == loc else f"{exp}: {loc}")
    return "{ " + ", ".join(items) + " }", pairs


def transform_module(module_id: str, code: str) -> str:
    # Some edited files may carry a UTF-8 BOM; strip it so import/export
    # matching works and raw ESM syntax is not emitted into bundle.js.
    if code.startswith("\ufeff"):
        code = code[1:]
    out_lines = []
    exports = []
    reexport_i = 0

    lines = code.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip("\ufeff").lstrip()

        if stripped.startswith("import "):
            stmt = stripped
            while ";" not in stmt and i + 1 < len(lines):
                i += 1
                stmt += "\n" + lines[i]
            s = stmt.strip()

            m = re.match(r"^import\s+\*\s+as\s+([A-Za-z_$][\w$]*)\s+from\s+['\"](.+?)['\"]\s*;\s*$", s, re.S)
            if m:
                name = m.group(1)
                spec = resolve_spec(module_id, m.group(2))
                out_lines.append(f"const {name} = require({spec!r});")
                i += 1
                continue

            m = re.match(r"^import\s+\{([\s\S]*?)\}\s+from\s+['\"](.+?)['\"]\s*;\s*$", s, re.S)
            if m:
                destruct, _pairs = parse_named_list(m.group(1))
                spec = resolve_spec(module_id, m.group(2))
                out_lines.append(f"const {destruct} = require({spec!r});")
                i += 1
                continue

            raise RuntimeError(f"Unsupported import syntax in {module_id}: {s}")

        if stripped.startswith("export "):
            m = re.match(r"^export\s+\{([\s\S]*?)\}\s+from\s+['\"](.+?)['\"]\s*;\s*$", stripped, re.S)
            if m:
                _destruct, pairs = parse_named_list(m.group(1))
                spec = resolve_spec(module_id, m.group(2))
                reexport_i += 1
                tmp = f"__reexport_{reexport_i}"
                out_lines.append(f"const {tmp} = require({spec!r});")
                for exp, loc in pairs:
                    out_lines.append(f"exports[{loc!r}] = {tmp}[{exp!r}];")
                i += 1
                continue

            m = re.match(r"^export\s+\{([\s\S]*?)\}\s*;\s*$", stripped, re.S)
            if m:
                _destruct, pairs = parse_named_list(m.group(1))
                for exp, loc in pairs:
                    out_lines.append(f"exports[{loc!r}] = {exp};")
                i += 1
                continue

            m = re.match(r"^export\s+(async\s+)?function\s+([A-Za-z_$][\w$]*)\b", stripped)
            if m:
                name = m.group(2)
                out_lines.append(line.replace("export ", "", 1))
                exports.append(name)
                i += 1
                continue

            m = re.match(r"^export\s+class\s+([A-Za-z_$][\w$]*)\b", stripped)
            if m:
                name = m.group(1)
                out_lines.append(line.replace("export ", "", 1))
                exports.append(name)
                i += 1
                continue

            m = re.match(r"^export\s+(const|let|var)\s+([A-Za-z_$][\w$]*)\b", stripped)
            if m:
                name = m.group(2)
                out_lines.append(line.replace("export ", "", 1))
                exports.append(name)
                i += 1
                continue

            raise RuntimeError(f"Unsupported export syntax in {module_id}: {stripped}")

        out_lines.append(line)
        i += 1

    if exports:
        out_lines.append("")
        for name in exports:
            out_lines.append(f"exports[{name!r}] = {name};")

    return "\n".join(out_lines) + "\n"


def main() -> None:
    modules = []
    for p in sorted(SRC.rglob("*.js")):
        rel = to_posix(str(p.relative_to(ROOT)))
        modules.append((rel, p))

    # Include browser suites so the file:// harness can run without module imports.
    suite = TEST / "complete.suite.js"
    if suite.exists():
        modules.append((to_posix(str(suite.relative_to(ROOT))), suite))

    out = []
    out.append("/* webgpufft browser bundle (file:// compatible) */\n")
    out.append("(function(){\n  'use strict';\n")
    out.append("  const __modules = Object.create(null);\n  const __cache = Object.create(null);\n")
    out.append("  function __define(id, fn){ __modules[id]=fn; }\n")
    out.append("  function __require(id){\n    if(__cache[id]) return __cache[id].exports;\n    const fn=__modules[id];\n    if(!fn) throw new Error('Missing module: '+id);\n    const module={exports:{}};\n    __cache[id]=module;\n    fn(__require, module.exports, module);\n    return module.exports;\n  }\n\n")

    for module_id, path in modules:
        code = path.read_text(encoding="utf-8-sig")
        transformed = transform_module(module_id, code)
        out.append(f"  __define({module_id!r}, function(require, exports, module){{\n")
        for ln in transformed.splitlines():
            out.append("    " + ln + "\n")
        out.append("  });\n\n")

    out.append("  const lib = __require('src/index.js');\n")
    out.append("  const math = __require('src/utils/math.js');\n\n")

    # Embed harness code from a literal file to keep generator small.
    harness = (ROOT / 'web' / 'harness_inline.js').read_text(encoding='utf-8-sig')
    out.append(harness)
    out.append("\n})();\n")

    OUT.write_text("".join(out), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == '__main__':
    main()

