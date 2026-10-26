"""Monkey-patch hashline.line_hash: MD5-hex[:2] -> letter-only 2-char hash.

原版 line_hash 输出 hex(0-9a-f), 约 1/3 概率纯数字(如 "58"), 弱模型
(Qwen3.5-35B-A3B 等)在生成 JSON 工具调用时会把 "58" 推断为 int,
触发 pydantic 的 string_type 校验失败, 进而退化为不带
end_line/end_hash 的 partial edit, 静默破坏文件结构.

换成 ZPMQVRWSNKTXJBYH 这套全字母 alphabet 后, hash 永远是 2 个字母
(如 "MQ"), JSON 输出只能是 string; 同时不与 hex 字面量、英文单词、
视觉相似字符冲突.

console.py 是懒导入 hashline 模块, 所以这里 patch 一次,
format_hashline_output(读端)和 apply_hashline_edit_with_summary
(写端)都会同步生效.

副作用模块: ``import`` 即生效, 无需调用任何函数. 必须在创建任何
deep agent / 调用 hashline 工具之前 import.
"""

from __future__ import annotations

import xxhash

from pydantic_ai_backends import hashline as _hashline
from pydantic_ai_backends.toolsets import console as _console


_HASHLINE_ALPHABET = "ZPMQVRWSNKTXJBYH"  # 16 字母, 无数字/元音/D/G/I/L/O


def _patched_line_hash(content: str) -> str:
    seed = content if any(c.isalnum() for c in content) else (content or " ")
    h = xxhash.xxh32(seed.encode("utf-8")).intdigest()
    return _HASHLINE_ALPHABET[(h >> 4) & 0x0F] + _HASHLINE_ALPHABET[h & 0x0F]


_hashline.line_hash = _patched_line_hash


_console.HASHLINE_READ_FILE_DESCRIPTION = """\
Read file content with hashline tags. ALWAYS read a file before editing it.

Every ``read_file`` call MUST include ``path``. The tool does not remember the
previous file. Do not call it with only ``offset``/``limit`` when paginating.

Each line is tagged with a content hash: ``{line_number}:{hash}|{content}``.
Use the ``line:hash`` pair when calling ``hashline_edit``.

Usage:
- For large files (>200 lines), use pagination: first scan with \
``limit=100`` to understand structure, then read targeted sections.
- For small files, read the whole file by not providing offset/limit.
- You can read multiple files in parallel for maximum efficiency.
- Read existing files before modifying them — understand the codebase first."""


_console.HASHLINE_EDIT_DESCRIPTION = """\
Edit a file by referencing lines with their content hashes. \
This is the preferred way to modify existing files.

You MUST ``read_file`` first to see the ``line:hash`` tags.

Every ``hashline_edit`` call MUST include ``path``. Use the exact same \
file path you passed to ``read_file``.

Required arguments for every call:
- ``path``
- ``start_line``
- ``start_hash``
- ``new_content``

For range replacement, also provide:
- ``end_line``
- ``end_hash``

For this project, solver edits usually target ``path="./workdir/solver.py"``.
Do not omit ``path``, even when editing the same file that was just read.

Operations:
- **Replace single line**: set ``path`` + ``start_line`` + ``start_hash`` + ``new_content``
- **Replace range**: set ``path`` + ``start_line`` + ``start_hash`` + ``end_line`` + ``end_hash`` + ``new_content``
- **Insert after**: set ``path`` + ``start_line`` + ``start_hash`` + ``insert_after=True`` + ``new_content``
- **Delete**: set ``path`` + ``start_line`` + ``start_hash`` + ``new_content=""``

If the hash doesn't match, the file changed since your last read — \
re-read it first. When making multiple edits, work **bottom-to-top** so \
line numbers don't shift.

After editing, re-read before making subsequent edits to the same file — \
auto-formatters or pre-commit hooks may have changed content on disk.
When making substitutions, change ONLY the exact tokens specified. \
Do NOT adjust surrounding text (articles, grammar, punctuation)."""
