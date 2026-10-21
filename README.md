# Educational video automation

A Python CLI turns a supplied narration file into timed speech, visual plans, sequential images, and a validated video. This branch experiments with text-only image generation: each cue receives a complete standalone prompt, and each image request receives that prompt without context or reference images. Script-file input is the default. An explicit topic mode also performs bounded research, script generation, and source auditing as described in `video_automation_plan.md`. Jobs use atomic local manifests. One worker owns the jobs directory at a time. Generation uses provider APIs without browser steps.

## Setup

Use Python 3.11 or later and an FFmpeg distribution with `ffmpeg`, `ffprobe`, `libx264`, `loudnorm`, and `sidechaincompress`.

```powershell
python -m pip install -e '.[test]'
python -m video_automation init
Copy-Item .env.example .env
```

Fill `.env` locally with `OPENROUTER_API_KEY` and `ELEVENLABS_API_KEY`. OpenRouter is the default provider for both text (`openai/gpt-oss-120b`) and images (`google/gemini-3.1-flash-lite-image`). One OpenRouter key covers both stages. Add Gemini keys to enable automatic image fallback when OpenRouter credits are exhausted; an empty pool does not prevent OpenRouter setup. Set `voice_id` in `config.local.json` to an ElevenLabs voice available to your account. Existing process environment variables take precedence over `.env`. The loader supports simple `KEY=value` lines, optionally quoted; it does not expand shell expressions. Keys are never stored in the manifest or execution log.

`init` detects the existing `D:/kenlive/bin` FFmpeg installation when the tools are not on PATH. All other installations can set executable paths in configuration. Relative music, font, and jobs paths resolve against the configuration file's directory. The legacy `character_reference` setting remains accepted but is ignored on this branch, including by setup checks. `config.local.json` loads automatically when running from this directory; `--config` selects another setup file. `config.example.json` documents the main settings.

```powershell
python -m video_automation doctor
python -m video_automation run .\narration.txt
```

The narration file is the only per-video content input in the default mode. Model IDs, voice, resolution, retry limits, and optional music are configured once. The doodle style is fixed in code; the visual planner chooses characters appropriate to the narration. The selected model IDs are explicit defaults; account access must be checked during setup. The configured speech models all accept the complete 4,500–5,000-character script in one request.

## Input modes

Save the final narration as a UTF-8 plain text file, with or without a BOM. Both commands select script mode with the default configuration:

```powershell
python -m video_automation run .\narration.txt
python -m video_automation run --script .\narration.txt
```

Script mode skips research, drafting, length revision, and source auditing. You are responsible for reviewing the supplied narration. The file is normalized by converting CRLF/CR to LF and trimming outer whitespace. It must contain 4,500–5,000 Python Unicode characters after normalization; invalid files stop before generation requests. The normalized text is sent unchanged to ElevenLabs. Exact character alignment, timed transcripts, visual planning, sequential image generation, rendering, and validation still run. The text provider is still required for visual planning.

Each script job stores an `input_script.txt` snapshot and its hash. `resume JOB_ID` reads that snapshot, so moving, editing, or deleting the original file does not change an existing job. Running a changed script creates a different default job. Reusing an explicit `--job-id` with different content or a different mode is rejected. If the saved snapshot is missing or corrupt, rerun with the original narration file and the same job ID to restore it. Input-file paths resolve against the working directory.

For the original research and script workflow, select topic mode explicitly:

```powershell
python -m video_automation run --mode topic 'Who Watched the Baby While Ancient Humans Hunted?'
```

`input_mode` defaults to `"script"`, including when an existing configuration omits the field. Set it to `"topic"` to make positional inputs topics again. `--mode` overrides the configured default; `--script` always selects script mode and cannot be combined with a positional input or `--mode topic`. Existing manifests retain their recorded mode when resumed; topic jobs remain labeled `production`, and supplied-script jobs use `script`.

## Text provider and model

The active setup in `config.local.json` uses:

```json
{
  "text_provider": "openrouter",
  "openrouter_model": "openai/gpt-oss-120b"
}
```

Add the actual key locally to the existing `.env` file:

```dotenv
OPENROUTER_API_KEY=your_openrouter_key
```

Change `openrouter_model` to select a different compatible OpenRouter text model. Requests use OpenRouter's Responses endpoint with strict JSON schemas and `provider.require_parameters=true`; responses also undergo local JSON Schema validation. Research uses the `openrouter:web_search` server tool with Exa, at most three searches, five results per search, and 15 total results. Retrieved citations remain mandatory. Model and search charges use your OpenRouter account. Speech generation still uses ElevenLabs.

Some routed search responses contain prose despite the requested JSON format. When cited prose is returned, the adapter makes one additional request to the same model to format the existing evidence as JSON, without search tools. The formatted result must pass the original schema and use only URLs retrieved by the search response. Refusals, empty responses, and missing source evidence stop immediately. Both requests remain in the usage ledger and raw response files; each reserves the configured `research` request cost cap when a spending ceiling is set. Only one formatting pass is allowed per search response; normal bounded stage retries still apply.

OpenAI remains an optional direct text provider. Set `text_provider` to `openai`, configure `openai_model` (default `gpt-4.1`), and fill `OPENAI_API_KEY`. Setup checks require the selected text and image providers' keys plus the ElevenLabs key, deduplicating a shared OpenRouter key. They print both active providers and models without revealing credentials. Text generation has no automatic provider fallback; image credit fallback is described below.

Provider and model are included in generation cache keys. Changing either invalidates the affected text stages and their dependent artifacts; existing imported-video exports remain available. A compatible schema/tool route and valid account access are required. Offline tests verify the adapter contract; a live request is still required to verify account-specific availability.

## Image provider and model

The active setup in `config.local.json` uses:

```json
{
  "image_provider": "openrouter",
  "openrouter_image_model": "google/gemini-3.1-flash-lite-image"
}
```

Image requests use OpenRouter's dedicated `POST /api/v1/images` API and the same `OPENROUTER_API_KEY` as text generation. Each request asks for one 1K image at 16:9. The request contains only a text prompt and generation settings; `input_references` is omitted. The direct Gemini route also sends only text parts. Returned base64 raster images pass the existing file/dimension checks before being checkpointed. Usage and available cost data are retained in the manifest. Missing, empty, malformed, or invalid images receive bounded retries; successful images and voiceover remain cached.

Change `openrouter_image_model` to choose another text-to-image model supporting the configured resolution/aspect ratio. Image settings have their own cache keys, including provider, model, Gemini fallback model, and text-only strategy version. Changing them regenerates affected images without invalidating unchanged research, narration, speech, or visual planning. Existing primary image checkpoints are adopted when adding the fallback setting, retaining artifact checks and persisted retry attempts. Changing credential values does not invalidate successful images. Each image depends on its exact prompt rather than earlier image files. Existing experimental media artifacts remain in their job folders until replaced by their regenerated outputs.

The visual planner writes complete scene descriptions in batches of 24 cues. Each description specifies visible characters and their appearance, actions, setting, objects, composition, and colors. Recurring characters repeat their appearance details. Previous scene descriptions may provide text-only context to the planner; the image model receives only the current prompt. The pipeline appends the fixed doodle style exactly once and checkpoints the complete ordered list in `image_prompts.json` before the first image request. Each record has `cue_id` and `prompt`; the saved prompt is the exact text sent to the image provider. There is no additional prompt-generation request or review pause. Full visual consistency remains a live-quality acceptance check.

Keep `image_provider` set to `openrouter` to use OpenRouter first. On an image HTTP 402 credit error, the current invocation switches to direct Gemini for that image and subsequent images. Temporary OpenRouter in-flight budget errors retain bounded retries. Authentication failures, rate limits, moderation errors, malformed images, and server or transport failures do not trigger provider fallback. Each new invocation tries OpenRouter first for uncached images, allowing a replenished account to resume primary generation. Text generation continues to use the configured text provider and has no Gemini fallback.

Configure the ordered Gemini key pool in the configuration directory's local `.env` file:

```dotenv
GEMINI_API_KEY=["your-first-key", "your-second-key"]
```

Use one line. A legacy single key or comma-separated list also works; duplicate keys are tried only once. `[]` disables the pool. Existing nonempty process environment values take precedence. `doctor` reports the fallback model and configured key count without printing keys. For direct Gemini (`image_provider: "gemini"`), an empty pool fails setup.

Gemini uses `gemini_model` for both fallback and direct generation, with the current code default `gemini-3.1-flash-lite-image`. An older local setup may retain another model and should be checked before enabling fallback. Requests send the same saved text-only prompt, with no reference images. The pool starts at its first key and retains a working key for subsequent images. HTTP 402 or 429 advances to the next key; other failures retain the existing permanent/temporary error policy. Pool exhaustion stops or uses the stage's bounded backoff retries for rate limits. A new invocation starts the pool again. [Gemini rate limits apply per project](https://ai.google.dev/gemini-api/docs/rate-limits), so multiple keys from one project share quota. Live model access, account credits, and image quality still require authenticated verification.

## Artifacts

Each input receives a stable experimental job directory, `jobs/video_text_only_<input-hash>/`. Topic mode hashes the topic; script mode hashes the normalized narration and its mode. `--job-id` provides an optional administrative name. New generated jobs record `image_strategy: "text_only"` in their manifest. Missing legacy strategy fields resolve to `"reference"`. This branch rejects legacy generated jobs before writing their files, including explicit job IDs, resume, validation, and editor export. Continue those jobs on `main`; the distinct default prefix keeps experimental results separate. Fixture and imported modes retain their existing rendering workflow.

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Job state, image strategy, cues, timing, attempts, input hashes, artifact hashes, errors, usage |
| `input_script.txt` | Normalized input snapshot for supplied-script jobs; retained for resume |
| `sources.json` | Retrieved source URLs and supported claims; topic mode only |
| `draft.json`, optional `draft_revision.json` | Original script draft and the single permitted length revision; topic mode only |
| `script.txt`, `script.json` | Exact normalized narration and stable paragraph/sentence IDs |
| `source_audit.json` | Separate model audit of every sentence's source support; topic mode only |
| `audio/narration.wav`, `alignment.json` | Final decoded audio and character, word, sentence timing |
| `transcript.json`, `visuals/` | Permanent cue IDs and scene plans |
| `image_prompts.json` | Ordered `{cue_id, prompt}` list containing the exact complete text for each image request |
| `images/`, `overlays/` | One validated image per cue and optional exact labels; no character-reference image |
| `render/prepared/`, `render/clips/`, `render/clips.txt` | Prepared image plates, encoded cue clips, and concatenation order |
| `rough.mp4`, `render_metadata.json` | Synchronized rough cut and absolute frame boundaries |
| `final.mp4`, `subtitles.en.srt` | H.264/AAC MP4 and separate subtitles |
| `quality_report.json`, `audio_measurements.json` | Validation and audio measurements |
| `execution.jsonl`, `responses/` | Execution events and raw provider responses |
| `progress.json` | Latest workflow phase, completed phase count, task/attempt, batch/image/clip counts, cache reuse, timestamps, and terminal status |
| `failure_report.json` | Most recent failed run, when present |

Raw responses and intermediates are retained for recovery and diagnosis. They may contain the narration and source content. A previous failure report is historical; the manifest and newest quality report identify current status.

HTTP failures save selected, redacted provider error fields and any available request ID in `responses/*_error.json`. The CLI includes the provider message and nested upstream reason when available. Request-header values and configured provider keys are redacted before fields are limited to 1,000 characters; non-JSON response bodies are not echoed. This diagnostic output does not change which HTTP failures receive automatic retries.

## Workflow progress

`run`, `resume`, `fixture`, and `import-assets` show live progress on stderr. Stdout retains the final JSON result. Generated jobs have seven phases: narration preparation, speech, visual planning, image generation, clip rendering, audio finishing, and validation. Fixture and imported jobs have the final three phases. Research, drafting, optional revision, and source auditing appear as tasks within narration preparation.

Progress shows the active task, attempt limit, cached tasks, and completed batches, images, or clips when their totals are known. Cue-derived totals remain dynamic. Counts describe completed work rather than estimated time remaining; phases have different costs and durations. Clip counts do not mark rendering complete until rough-cut assembly also finishes. Completion is reported only after validation passes.

Each phase transition and checkpoint event updates the job's atomic `progress.json` snapshot. Inspect it from another terminal while the worker runs:

```powershell
python -m video_automation status video_text_only_JOB_HASH
```

`status` also accepts a job folder or manifest path. It reads saved progress and the authoritative manifest status without acquiring the worker lock, making generation requests, or modifying files. Jobs predating progress tracking return their manifest status with `progress_available: false`. Snapshots include an update timestamp and elapsed seconds for the current invocation. During a long provider or FFmpeg call, the last reported task remains active. Abrupt process termination may leave a running status until the next resume.

On resume, cached stages and cue checkpoints are verified and reported as reused work; the phase counters describe that invocation. Failure and interruption preserve the last phase and completed counts. If saving a progress snapshot fails, the CLI warns once and continues using normal workflow checkpoints. A disconnected progress stream does not cause generation retries.

## Resume and retries

```powershell
python -m video_automation resume video_text_only_JOB_HASH
python -m video_automation resume video_text_only_JOB_HASH --retry-failed
python -m video_automation validate video_text_only_JOB_HASH
```

`resume` accepts a job name, job folder, or manifest path. Generated jobs must carry the text-only strategy marker on this branch. Completed stages are reused only when their input hashes and output file hashes match. A missing or corrupt output is rebuilt. Changed prompts and settings invalidate dependent work. Image generation and clip rendering checkpoint each cue individually. Successful images and speech remain available if a later image fails.

Temporary HTTP failures, transport errors, missing image responses, and invalid images receive bounded exponential-backoff retries. Authentication, unsupported content, failed source audits, and invalid narration fail clearly. The retry counter persists across restarts. `--retry-failed` explicitly starts one new bounded batch for failed or interrupted stages; successful stages remain cached. An interrupted request can still be billable if the provider completed it before the client lost the response.

Visual batches must contain every requested cue ID exactly once in order, with valid scene descriptions, character IDs, and motion settings. The requested output schema constrains the cue count and allowed IDs for each batch. Malformed visual output is rejected before a checkpoint is marked completed and receives the existing bounded retries; refusals still stop immediately. Cached batches from older runs are revalidated before reuse. Invalid cached batches are retried within their remaining attempt budget, while valid earlier batches and completed speech remain cached. Later batches are regenerated when their continuity inputs change. Raw provider responses remain available for diagnosis.

The character count uses Python Unicode characters after CRLF/CR normalization and outer whitespace trimming. Only `script.txt` is counted and spoken. Topic mode allows exactly one automatic length revision; a second failure stops before speech or image generation. Script mode rejects out-of-range input without rewriting it. Original ElevenLabs alignment must match the exact narration in both modes. A mismatch fails clearly rather than inventing timestamps. Timing is frozen after audio decoding; later speech changes require regeneration and alignment.

Visual cues follow aligned words and sentence boundaries. Their count is dynamic. Frame durations derive from adjacent absolute boundaries. Boundaries that collapse onto one frame are merged without dropping text. The first visual starts at frame zero; the last image holds until the final recording ends. Output may pad the final audio by less than one frame to match the encoded video.

The doodle style lives in code. Characters and appearance descriptions are chosen from the narration and repeated in standalone prompts. This branch neither creates nor reads character-reference or scene-reference images. Only `hold` and a centered 1.5% `slow_push` are supported. Overlay scenes remain stationary. Image checks cover file validity, dimensions, and coverage; artistic correctness is outside this MVP.

Text overlays are RGBA PNGs containing only red text. Every pixel outside the text is fully transparent; no background panel, shadow, or border is drawn. Overlay checkpoints include a renderer version. Resuming a job refreshes older label PNGs and their dependent video clips without regenerating unchanged scene images or speech. Renderer fixtures and Kdenlive exports refresh labels through the same checkpoint path. Already encoded MP4 files change only when the job is rendered again.

Audio finishing uses two-pass loudness normalization. Optional local music loops, fades, and ducks under speech. Validation decodes the final file and checks image coverage, text coverage, timeline continuity, frame count, format, duration agreement within one frame, caption bounds, overlay bounds, peaks, and long silences in both narration and final audio. Failed validation writes a report and marks the job failed.

## Cost controls

Every attempted provider request records status, available provider usage, response location, and request ID. No price is invented for unknown usage. By default `spending_ceiling_usd` is `null`; provider billing limits should be configured on the provider accounts.

An optional local budget uses conservative, operator-supplied **maximum request costs**, keyed by operation: `research`, `script`, `script_revision`, `source_audit`, `speech`, `visuals`, and `image`. If a ceiling is set, every requested operation must have a cap. Before each request, the pipeline reserves that cap and refuses requests that would exceed the job ceiling. Each OpenRouter attempt, Gemini fallback request, and Gemini key change reserves a separate `image` cap; configure that cap to cover both image providers. Reservations remain charged against the local allowance even after timeouts or failed responses. This bounds the local ledger only to the extent that the supplied caps truly cover provider charges. It is not a provider-enforced billing limit. Model and tool prices must be checked when configuring those caps.

## Renderer fixtures and existing assets

```powershell
python -m video_automation fixture
python -m video_automation resume renderer_fixture
```

This portable three-scene fixture uses synthetic tone audio, verifies stationary/moving scenes and an exact `38%` overlay, and requires no API credentials. It is labeled `fixture`; the production narration-length requirement is explicitly excluded from its report.

The original workspace's images, narration and corrected subtitles can be imported without changing the originals:

```powershell
python -m video_automation import-assets `
  --manifest 'D:/duc/startUp/yt/who_watched_the_baby_images_20261020/manifest.json' `
  --audio 'D:/duc/startUp/yt/ElevenLabs_2026-10-20T14_30_14_Eve - Narrative & Audiobooks_pvc_sp95_s71_sb37_se0_b_m2.mp3' `
  --subtitles 'D:/duc/startUp/yt/who_watched_the_baby_video/who_watched_the_baby.en.srt' `
  --job-id existing_full
```

Add `--limit-cues 3` and a different job ID to make a short fixture. Without a limit, the complete source recording is rendered. Cue counts and audio duration are read from assets. Existing motion decisions are reused from `edit_decisions.csv` beside the subtitles when available; other imported scenes stay stationary. Imported jobs use supplied cue timings and do not claim generated word alignment or topic-to-video acceptance.

## Optional Kdenlive export

```powershell
python -m video_automation export-kdenlive existing_full `
  --template 'D:/duc/startUp/yt/who_watched_the_baby_video/kdenlive_template.kdenlive'
```

This writes `project.kdenlive` from the job manifest using the existing workspace template. Image count, narration path, profile, scene durations, and selected camera movement are dynamic. Overlays are baked into the editor's prepared image plates. The existing builder and original Kdenlive project remain available. The MP4 pipeline does not require an editor.

## Verification

```powershell
python -m pytest -q
```

If a restricted Windows environment cannot create its default temporary directory, create `.test-tmp` in the workspace and use an absolute `--basetemp` path under it. Tests use offline provider contracts for narration gates, audits, exact speech text, word coverage, caching, retries, budget checks, and interrupted-image recovery. Real FFmpeg integration covers music ducking, overlays, encoding, validation, and cached reruns. Offline provider tests do not prove live account access or generated content quality.

The final acceptance step remains an unattended **live** topic-to-video run with configured provider credentials and voice ID. See `IMPLEMENTATION_STATUS.md` for the evidence collected in this workspace.

## Provider references

- [OpenRouter Responses API](https://openrouter.ai/docs/api_reference/responses/overview)
- [OpenRouter web search server tool](https://openrouter.ai/docs/guides/features/server-tools/web-search)
- [OpenRouter structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs)
- [Requested default: openai/gpt-oss-120b](https://openrouter.ai/openai/gpt-oss-120b)
- [OpenRouter Image API](https://openrouter.ai/docs/guides/overview/multimodal/image-generation)
- [Requested image model: google/gemini-3.1-flash-lite-image](https://openrouter.ai/google/gemini-3.1-flash-lite-image)
- [OpenAI Docs: structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
- [OpenAI Docs: web search](https://developers.openai.com/api/docs/guides/tools-web-search)
- [ElevenLabs speech with timing](https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps/)
- [ElevenLabs model input limits](https://elevenlabs.io/docs/overview/models)
- [Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation)
- [Gemini generateContent API](https://ai.google.dev/api/generate-content)
- [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html)
