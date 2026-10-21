# Repository guidance for Codex

## Response style

- Always follow the installed Caveman skill.
- Default mode: full.
- Keep responses concise and direct.
- Preserve technical details, code, commands, paths, numbers, and exact errors.
- Use full sentences for warnings and ambiguous multi-step instructions.
- Stop Caveman style only when the user says "stop caveman" or "normal mode".
- Write persisted documentation, code comments, and other project artifacts in normal prose, as the Caveman skill requires.

## Project overview

This repository contains a Python CLI that turns a narration file into a narrated educational video by default. Explicit topic mode also generates the narration from research. It uses provider APIs for generation, local files for job state, and FFmpeg for deterministic rendering. The MVP runs one worker per jobs directory, with sequential image generation and resumable checkpoints. There is no web application or database.

Runtime: Python 3.11+, Pydantic 2, Pillow, httpx, and jsonschema. Tests use pytest. FFmpeg and ffprobe must be installed, with `libx264`, `loudnorm`, and `sidechaincompress` available.

Read `README.md` for usage, `IMPLEMENTATION_STATUS.md` for recorded verification and deferred acceptance, and `video_automation_plan.md` for the original design. Prefer current code and tests when the original plan differs; text generation now defaults to OpenRouter rather than direct OpenAI.

## Structure and ownership

| Path | Responsibility |
| --- | --- |
| `video_automation/__main__.py` | Entry point for `python -m video_automation`. |
| `video_automation/cli.py` | Argument parsing, local credentials, setup checks, job lookup, and worker locking. |
| `video_automation/models.py` | Strict Pydantic configuration, script, timing, cue, stage, and manifest records. |
| `video_automation/pipeline.py` | Research, narration gates, source audit, speech decoding, visual planning, images, and orchestration. |
| `video_automation/providers.py` | Provider HTTP calls, prompts, JSON schemas, research provenance, and fixed visual style. |
| `video_automation/store.py` | Atomic writes, hashes, checkpoints, retries, usage reservations, path bounds, and worker lock. |
| `video_automation/timing.py` | Narration normalization, sentence/word alignment, cue segmentation, absolute frame boundaries, and SRT. |
| `video_automation/media.py` | Image checks, overlays, FFmpeg rendering, audio finishing, and final validation. |
| `video_automation/assets.py` | Synthetic renderer fixtures and legacy asset import. |
| `video_automation/kdenlive.py` | Optional editable project export from a supplied Kdenlive template. |
| `tests/test_pipeline.py` | Offline provider contracts, pipeline invariants, recovery, and real FFmpeg integration. |
| `tests/test_openrouter.py`, `tests/test_openrouter_images.py` | Text/image provider selection, HTTP contracts, references, schemas, citations, credentials, retries, and generation cache invalidation. |
| `pyproject.toml` | Packaging, dependencies, test extra, and pytest discovery. |
| `config.example.json`, `.env.example` | Tracked setup templates without credentials. |

`jobs/`, `.venv/`, `.test-tmp/`, `.env`, and `config.local.json` are ignored local state. Do not treat them as source files. The installed console command `video-automation` calls the same CLI as the module entry point.

## Setup and configuration

Run commands from the repository root. These PowerShell commands preserve existing configuration and credentials:

```powershell
if (-not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\python.exe -m pip install -e '.[test]'
if (-not (Test-Path config.local.json)) { & .\.venv\Scripts\python.exe -m video_automation init }
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Configure `voice_id` locally. The code defaults to OpenRouter text model `openai/gpt-oss-120b`, ElevenLabs speech model `eleven_multilingual_v2`, and OpenRouter image model `google/gemini-3.1-flash-lite-image`. Direct OpenAI text generation is optional through `text_provider: "openai"` and `openai_model`. Direct Gemini images are optional through `image_provider: "gemini"` and `gemini_model`. There is no automatic application-level provider fallback.

The selected text provider requires `OPENROUTER_API_KEY` or `OPENAI_API_KEY`. The selected image provider requires `OPENROUTER_API_KEY` or `GEMINI_API_KEY`. The default text/image routes share one OpenRouter key; speech requires `ELEVENLABS_API_KEY`. Nonempty process environment values take precedence over `.env`. Never print, commit, or copy credential values into reports, manifests, or logs.

The CLI automatically loads `config.local.json` from the working directory. Put the global `--config` option before the subcommand to select another file. Relative jobs, music, font, and character reference paths resolve against the configuration directory; `.env` loads from that directory too. `Config.load()` checks `D:/kenlive/bin` for FFmpeg tools when default executable names are unavailable on PATH. Avoid adding machine-specific paths elsewhere.

```powershell
& .\.venv\Scripts\python.exe -m video_automation --help
& .\.venv\Scripts\python.exe -m video_automation --config .\config.local.json doctor
```

`doctor` checks local executables, credential presence, and configured inputs without making generation requests. It does not verify live account access or model compatibility.

## Production workflow and invariants

The default `input_mode` is `script`: `run narration.txt` or `run --script narration.txt` supplies final UTF-8 narration and skips research, drafting, revision, and source auditing. Snapshot the normalized input in `input_script.txt` with its hash so resume does not depend on the external file. `run --mode topic TOPIC` keeps the original workflow, recorded as `production` in manifests. Existing jobs retain their saved mode when resumed.

1. In topic mode, research the topic with bounded searches and retrieved source URLs. Generate a structured narration draft. In script mode, read the supplied narration instead.
2. Normalize CRLF/CR to LF and trim outer whitespace. Require exactly 4,500–5,000 Python Unicode characters in both modes, including internal spaces and punctuation. Topic mode permits only one automatic length revision; script mode never rewrites supplied text.
3. In topic mode, audit every sentence against research. Reject unsupported claims, missing coverage, and references outside retrieved sources before paid speech or image generation. Script mode skips this audit explicitly.
4. Send the exact normalized narration to ElevenLabs. Decode audio to WAV and derive word/sentence timing from original character alignment. Reject text mismatches instead of inventing timestamps.
5. Build dynamic cues from aligned words. Plan visuals in batches of 24 while preserving every cue ID and its order.
6. Generate images sequentially. Reuse the shared character reference and the first approved image within each scene group. Render exact labels with Pillow; overlays require stationary scenes.
7. Render clips, assemble `rough.mp4`, finish audio, write separate SRT subtitles, and validate `final.mp4`. Mark the job completed only after validation passes.

Preserve these additional constraints:

- Audio determines duration. Never hardcode cue counts, video length, image counts, or values from the existing imported video.
- Derive clip lengths from adjacent absolute frame boundaries. Start at frame zero and hold the final image through the recording. Merge collapsed boundaries without losing text.
- Support only `hold` and the centered 1.5% `slow_push` motion unless the task explicitly changes that scope.
- Keep production, script, fixture, and imported modes distinct. Script mode still requires exact fresh speech alignment and the narration-length gate. Fixture/imported rendering does not prove live topic-to-video acceptance or fresh speech alignment.
- Keep generation independent of browser interfaces. Retain the single-worker architecture unless the task explicitly requires a change.

## Jobs, recovery, and costs

Default job folders use `jobs/video_<input-hash>/`; script identity includes normalized narration and mode, while topic identity retains the original topic hash. `--job-id` supplies an administrative name and rejects changed inputs or modes. `manifest.json` holds state, attempts, cues, usage, and artifact hashes. `execution.jsonl` records events. Key outputs include `input_script.txt` in script mode, `sources.json` and `source_audit.json` in topic mode, and shared `script.txt`, `script.json`, `alignment.json`, `transcript.json`, image/overlay folders, `rough.mp4`, `final.mp4`, `subtitles.en.srt`, and `quality_report.json`.

Use `Store.task()` for reusable work and list all output files in its result. Completed stages are reusable only when input hashes and output hashes match. Include changed prompts, models, settings, reference hashes, or an explicit version in affected cache inputs. A code change alone does not automatically invalidate cached stages. Write artifacts with the existing atomic helpers and resolve job-relative paths through `Store.path()`.

`TemporaryError` receives bounded exponential-backoff retries. Other failures stop the stage. Attempts persist across restarts; `resume --retry-failed` explicitly resets failed/running attempts for another bounded batch. Preserve completed speech, images, clips, and raw provider responses during recovery.

```powershell
& .\.venv\Scripts\python.exe -m video_automation run --script .\narration.txt
& .\.venv\Scripts\python.exe -m video_automation run --mode topic 'Your educational topic'
& .\.venv\Scripts\python.exe -m video_automation resume JOB_ID
& .\.venv\Scripts\python.exe -m video_automation resume JOB_ID --retry-failed
& .\.venv\Scripts\python.exe -m video_automation validate JOB_ID
```

`resume` and `validate` also accept a job folder or manifest path. Inspect the manifest and latest quality report for current status; `failure_report.json` may describe an earlier run. Raw responses contain source/narration content and remain local. Do not delete existing jobs to resolve a failed stage.

Provider attempts reserve optional operation cost caps before requests. If `spending_ceiling_usd` is configured, every requested operation needs a conservative cap. Reservations remain counted after failures/timeouts. This ledger is not a provider-enforced billing limit. Use offline providers for routine verification; live generation can incur charges.

## Verification workflow

Run focused tests for changed behavior, then the full suite when changing pipeline, provider, timing, persistence, or media behavior:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q tests/test_openrouter.py
& .\.venv\Scripts\python.exe -m pytest -q tests/test_pipeline.py
& .\.venv\Scripts\python.exe -m pytest -q
```

Provider tests are offline. Several pipeline tests run real FFmpeg, including rendering, music ducking, overlays, validation, and cached reruns. Reuse `FakeProviders` and HTTP mocks; do not introduce live requests into automated tests. No formatter, linter, or type-check command is currently configured in `pyproject.toml`.

If Windows cannot create pytest's default temporary directory, use disposable workspace storage. Pytest clears its `--basetemp` directory, so never point it at jobs or other retained artifacts:

```powershell
New-Item -ItemType Directory -Force .test-tmp | Out-Null
& .\.venv\Scripts\python.exe -m pytest -q --basetemp="$PWD\.test-tmp\pytest"
```

For a manual renderer smoke test without credentials, use a new fixture job ID:

```powershell
& .\.venv\Scripts\python.exe -m video_automation fixture --job-id codex_renderer_fixture
& .\.venv\Scripts\python.exe -m video_automation resume codex_renderer_fixture
```

`fixture` renders immediately and rejects an existing job; use `resume` for subsequent runs. The synthetic fixture exercises three scenes and an exact `38%` overlay. Use `import-assets` for legacy files and `export-kdenlive JOB_ID --template PATH` for optional editor export; preserve original source assets. See `README.md` for arguments.

The recorded live topic-to-video acceptance remains deferred in `IMPLEMENTATION_STATUS.md`. Offline tests and imported exports do not verify provider account access, actual costs, generated factual quality, or artistic continuity. State exactly which checks ran and whether any live acceptance occurred.

## Change guidance

- Make changes in the responsible module; keep provider HTTP behavior out of CLI and rendering code.
- Preserve strict Pydantic records and locally validated provider JSON schemas. Keep schema changes compatible with saved manifests or explicitly handle migration.
- Add regression coverage for changed invariants, retries, cache inputs, or timing. Documentation-only changes need command/reference checks rather than new tests.
- Update `README.md` and setup templates when public commands or configuration change. Update `IMPLEMENTATION_STATUS.md` only with observed evidence.
- Keep secrets, generated media, virtual environments, and local configuration untracked. Preserve user edits and existing job artifacts.
