#!/usr/bin/env python
#------------------------------------------------------------------------------
# Copyright 2013-2014 Avik Partners, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------
import json
import os
import re
import sys
from collections import Counter
from optparse import OptionParser

import requests

from grokcli.exceptions import GrokCLIError


if __name__ == "__main__":
  subCommand = "%prog"
else:
  subCommand = "%%prog %s" % __name__.rpartition('.')[2]


USAGE = """%s --query "QUESTION" --corpus PATH [options]

Retrieve context from local files and ask Grok with grounded context.
""".strip() % subCommand

parser = OptionParser(usage=USAGE)
parser.add_option(
  "-q",
  "--query",
  dest="query",
  help="Question to ask")
parser.add_option(
  "-c",
  "--corpus",
  dest="corpus_paths",
  action="append",
  default=[],
  help="File or directory to index; can be used multiple times")
parser.add_option(
  "--ext",
  dest="extensions",
  default=".md,.txt,.rst,.json,.yaml,.yml,.py,.js,.ts,.tsx,.jsx,.java,.go,.rs,.c,.cpp,.h,.hpp,.html,.css,.csv",
  help="Comma-separated file extensions to include")
parser.add_option(
  "--top-k",
  dest="top_k",
  type="int",
  default=5,
  help="Number of retrieved chunks to include (default: 5)")
parser.add_option(
  "--chunk-size",
  dest="chunk_size",
  type="int",
  default=1200,
  help="Chunk size in characters (default: 1200)")
parser.add_option(
  "--chunk-overlap",
  dest="chunk_overlap",
  type="int",
  default=150,
  help="Chunk overlap in characters (default: 150)")
parser.add_option(
  "--api-url",
  dest="api_url",
  default="https://api.x.ai/v1/chat/completions",
  help="Chat completion endpoint")
parser.add_option(
  "--model",
  dest="model",
  default="grok-2-latest",
  help="Model name to use (default: grok-2-latest)")
parser.add_option(
  "--api-key",
  dest="api_key",
  default=None,
  help="API key. If omitted, uses XAI_API_KEY env var")
parser.add_option(
  "--temperature",
  dest="temperature",
  type="float",
  default=0.2,
  help="Sampling temperature (default: 0.2)")
parser.add_option(
  "--max-tokens",
  dest="max_tokens",
  type="int",
  default=800,
  help="Max tokens to generate (default: 800)")
parser.add_option(
  "--timeout",
  dest="timeout",
  type="int",
  default=60,
  help="HTTP timeout seconds (default: 60)")
parser.add_option(
  "--show-context",
  dest="show_context",
  default=False,
  action="store_true",
  help="Print retrieved chunks before model output")


def _tokenize(text):
  return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _iter_files(path, allowed_exts):
  if os.path.isfile(path):
    _, ext = os.path.splitext(path)
    if ext.lower() in allowed_exts:
      yield path
    return

  for root, _, files in os.walk(path):
    for name in files:
      _, ext = os.path.splitext(name)
      if ext.lower() in allowed_exts:
        yield os.path.join(root, name)


def _split_text(text, chunk_size, chunk_overlap):
  if chunk_size <= 0:
    raise GrokCLIError("chunk-size must be > 0")
  if chunk_overlap < 0:
    raise GrokCLIError("chunk-overlap must be >= 0")
  if chunk_overlap >= chunk_size:
    raise GrokCLIError("chunk-overlap must be smaller than chunk-size")

  chunks = []
  start = 0
  step = chunk_size - chunk_overlap
  text_len = len(text)
  while start < text_len:
    end = min(start + chunk_size, text_len)
    chunk = text[start:end].strip()
    if chunk:
      chunks.append(chunk)
    if end >= text_len:
      break
    start += step
  return chunks


def _build_index(paths, allowed_exts, chunk_size, chunk_overlap):
  entries = []
  for path in paths:
    for file_path in _iter_files(path, allowed_exts):
      try:
        data = open(file_path, "rb").read()
        text = data.decode("utf-8", "ignore")
      except Exception:
        continue

      chunks = _split_text(text, chunk_size, chunk_overlap)
      for idx, chunk in enumerate(chunks):
        terms = _tokenize(chunk)
        if not terms:
          continue
        entries.append({
          "path": file_path,
          "chunk_id": idx,
          "text": chunk,
          "terms": Counter(terms),
        })
  return entries


def _score(query_terms, entry):
  score = 0.0
  for term, q_count in query_terms.items():
    d_count = entry["terms"].get(term, 0)
    if d_count:
      score += min(float(q_count), float(d_count))
  return score


def _retrieve(index_entries, query, top_k):
  query_terms = Counter(_tokenize(query))
  if not query_terms:
    raise GrokCLIError("query must include at least one alphanumeric token")

  scored = []
  for entry in index_entries:
    s = _score(query_terms, entry)
    if s > 0:
      scored.append((s, entry))

  scored.sort(key=lambda x: x[0], reverse=True)
  return scored[:top_k]


def _build_messages(query, retrieved):
  blocks = []
  for score, entry in retrieved:
    blocks.append(
      "[Source: %s | Chunk: %s | Score: %.2f]\n%s" % (
        entry["path"],
        entry["chunk_id"],
        score,
        entry["text"],
      )
    )

  context = "\n\n".join(blocks)

  system = (
    "You are a grounded assistant. Use the provided context first. "
    "If context is insufficient, say so clearly and then provide the best-effort answer. "
    "Always cite source paths in your answer when you use context."
  )
  user = (
    "Question:\n%s\n\n"
    "Context:\n%s\n\n"
    "Answer with concise bullets and include citations like [path/to/file]."
  ) % (query, context)

  return [
    {"role": "system", "content": system},
    {"role": "user", "content": user},
  ]


def _request_completion(options, messages):
  api_key = options.api_key or os.environ.get("XAI_API_KEY")
  if not api_key:
    raise GrokCLIError("Missing API key. Pass --api-key or set XAI_API_KEY")

  payload = {
    "model": options.model,
    "messages": messages,
    "temperature": options.temperature,
    "max_tokens": options.max_tokens,
  }

  headers = {
    "Authorization": "Bearer %s" % api_key,
    "Content-Type": "application/json",
  }

  response = requests.post(
    options.api_url,
    headers=headers,
    data=json.dumps(payload),
    timeout=options.timeout,
  )

  if response.status_code < 200 or response.status_code >= 300:
    raise GrokCLIError("API request failed (%s): %s" % (response.status_code, response.text))

  data = response.json()
  try:
    return data["choices"][0]["message"]["content"]
  except Exception:
    raise GrokCLIError("Unexpected API response format: %s" % json.dumps(data))


def handle(options, args):
  if args:
    parser.print_help(sys.stderr)
    sys.exit(1)

  if not options.query:
    raise GrokCLIError("--query is required")
  if not options.corpus_paths:
    raise GrokCLIError("At least one --corpus path is required")

  allowed_exts = set()
  for ext in options.extensions.split(","):
    ext = ext.strip()
    if not ext:
      continue
    if not ext.startswith("."):
      ext = "." + ext
    allowed_exts.add(ext.lower())

  index_entries = _build_index(
    options.corpus_paths,
    allowed_exts,
    options.chunk_size,
    options.chunk_overlap,
  )
  if not index_entries:
    raise GrokCLIError("No indexable documents found in corpus paths")

  retrieved = _retrieve(index_entries, options.query, options.top_k)
  if not retrieved:
    raise GrokCLIError("No relevant context found. Try broader corpus or different query")

  if options.show_context:
    print "Retrieved context:"
    for score, entry in retrieved:
      print "- %s#%s (score=%.2f)" % (entry["path"], entry["chunk_id"], score)
    print

  messages = _build_messages(options.query, retrieved)
  answer = _request_completion(options, messages)
  print answer


if __name__ == "__main__":
  handle(*parser.parse_args())