# Educational video automation

A Python CLI turns a supplied narration file into timed speech, visual plans, sequential images, and a validated video. Script-file input is the default. An explicit topic mode also performs bounded research, script generation, and source auditing as described in `video_automation_plan.md`. Jobs use atomic local manifests. One worker owns the jobs directory at a time. Generation uses provider APIs without browser steps.

## Setup

Use Python 3.11 or later and an FFmpeg distribution with `ffmpeg`, `ffprobe`, `libx264`, `loudnorm`, and `sidechaincompress`.

```powershell
python -m pip install -e '.[test]'
python -m video_automation init
Copy-Item .env.example .env
```

Fill `.env` locally with `OPENROUTER_API_KEY` and `ELEVENLABS_API_KEY`. OpenRouter is the default provider for both text (`openai/gpt-oss-120b`) and images (`google/gemini-3.1-flash-lite-image`). One OpenRouter key covers both stages; no `GEMINI_API_KEY` is required with these defaults. Set `voice_id` in `config.local.json` to an ElevenLabs voice available to your account. Existing process environment variables take precedence over `.env`. The loader supports simple `KEY=value` lines, optionally quoted; it does not expand shell expressions. Keys are never stored in the manifest or execution log.

`init` detects the existing `D:/kenlive/bin` FFmpeg installation when the tools are not on PATH. All other installations can set executable paths in configuration. Relative music, font, reference, and jobs paths resolve against the configuration file's directory. `config.local.json` loads automatically when running from this directory; `--config` selects another setup file. `config.example.json` documents the main settings.

```powershell
python -m video_automation doctor
python -m video_automation run .\narration.txt
```

The narration file is the only per-video content input in the default mode. Model IDs, voice, style, resolution, retry limits, optional music, and optional character reference are configured once. The selected model IDs are explicit defaults; account access must be checked during setup. The configured speech models all accept the complete 4,500–5,000-character script in one request.

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

OpenAI remains an optional direct text provider. Set `text_provider` to `openai`, configure `openai_model` (default `gpt-4.1`), and fill `OPENAI_API_KEY`. Setup checks require the selected text and image providers' keys plus the ElevenLabs key, deduplicating a shared OpenRouter key. They print both active providers and models without revealing credentials. There is no automatic application-level provider fallback.

Provider and model are included in generation cache keys. Changing either invalidates the affected text stages and their dependent artifacts; existing imported-video exports remain available. A compatible schema/tool route and valid account access are required. Offline tests verify the adapter contract; a live request is still required to verify account-specific availability.

## Image provider and model

The active setup in `config.local.json` uses:

```json
{
  "image_provider": "openrouter",
  "openrouter_image_model": "google/gemini-3.1-flash-lite-image"
}
```

Image requests use OpenRouter's dedicated `POST /api/v1/images` API and the same `OPENROUTER_API_KEY` as text generation. Each request asks for one 1K image at 16:9. Character and scene references are sent as base64 data URLs in `input_references`. Returned base64 raster images pass the existing file/dimension checks before being checkpointed. Usage and available cost data are retained in the manifest. Missing, empty, malformed, or invalid images receive bounded retries; successful images and voiceover remain cached.

Change `openrouter_image_model` to choose another image model that supports reference images and the configured resolution/aspect ratio. Image settings have their own cache keys, including provider and model. Changing them regenerates affected images and references without invalidating unchanged research, narration, speech, or visual planning. Existing media artifacts remain in their job folders until replaced by their regenerated outputs.

Direct Gemini remains optional: set `image_provider` to `gemini`, set `gemini_model` to an available Google image model, and supply `GEMINI_API_KEY`. Its current code default is `gemini-3.1-flash-lite-image`; an older local setup may retain a previous `gemini_model` value and should update it before selecting direct Gemini. OpenRouter image generation does not use that field or require a separate Google account/key. Live model access and available credits must still be verified with your OpenRouter account.

## Artifacts

Each input receives a stable job directory, `jobs/video_<input-hash>/`. Topic mode hashes the topic; script mode hashes the normalized narration and its mode. `--job-id` provides an optional administrative name.

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Job state, cues, timing, attempts, input hashes, artifact hashes, errors, usage |
| `input_script.txt` | Normalized input snapshot for supplied-script jobs; retained for resume |
| `sources.json` | Retrieved source URLs and supported claims; topic mode only |
| `draft.json`, optional `draft_revision.json` | Original script draft and the single permitted length revision; topic mode only |
| `script.txt`, `script.json` | Exact normalized narration and stable paragraph/sentence IDs |
| `source_audit.json` | Separate model audit of every sentence's source support; topic mode only |
| `audio/narration.wav`, `alignment.json` | Final decoded audio and character, word, sentence timing |
| `transcript.json`, `visuals/` | Permanent cue IDs and scene plans |
| `images/`, `overlays/` | One validated image per cue, shared reference, exact labels |
| `rough.mp4`, `render_metadata.json` | Synchronized rough cut and absolute frame boundaries |
| `final.mp4`, `subtitles.en.srt` | H.264/AAC MP4 and separate subtitles |
| `quality_report.json`, `audio_measurements.json` | Validation and audio measurements |
| `execution.jsonl`, `responses/` | Execution events and raw provider responses |
| `failure_report.json` | Most recent failed run, when present |

Raw responses and intermediates are retained for recovery and diagnosis. They may contain the narration and source content. A previous failure report is historical; the manifest and newest quality report identify current status.

## Resume and retries

```powershell
python -m video_automation resume video_JOB_HASH
python -m video_automation resume video_JOB_HASH --retry-failed
python -m video_automation validate video_JOB_HASH
```

`resume` accepts a job name, job folder, or manifest path. Completed stages are reused only when their input hashes and output file hashes match. A missing or corrupt output is rebuilt. Changed prompts, settings, and references invalidate dependent work. Image generation and clip rendering checkpoint each cue individually. Successful images and speech remain available if a later image fails.

Temporary HTTP failures, transport errors, missing image responses, and invalid images receive bounded exponential-backoff retries. Authentication, unsupported content, failed source audits, and invalid narration fail clearly. The retry counter persists across restarts. `--retry-failed` explicitly starts one new bounded batch for failed or interrupted stages; successful stages remain cached. An interrupted request can still be billable if the provider completed it before the client lost the response.

The character count uses Python Unicode characters after CRLF/CR normalization and outer whitespace trimming. Only `script.txt` is counted and spoken. Topic mode allows exactly one automatic length revision; a second failure stops before speech or image generation. Script mode rejects out-of-range input without rewriting it. Original ElevenLabs alignment must match the exact narration in both modes. A mismatch fails clearly rather than inventing timestamps. Timing is frozen after audio decoding; later speech changes require regeneration and alignment.

Visual cues follow aligned words and sentence boundaries. Their count is dynamic. Frame durations derive from adjacent absolute boundaries. Boundaries that collapse onto one frame are merged without dropping text. The first visual starts at frame zero; the last image holds until the final recording ends. Output may pad the final audio by less than one frame to match the encoded video.

The style and character design live in code. A generated character reference is reused across jobs under `jobs/references/`. Consecutive cues in a scene use the first approved image for that scene as an additional reference. Only `hold` and a centered 1.5% `slow_push` are supported. Overlay scenes remain stationary. Image checks cover file validity, dimensions, and coverage; artistic correctness is outside this MVP.

Audio finishing uses two-pass loudness normalization. Optional local music loops, fades, and ducks under speech. Validation decodes the final file and checks image coverage, text coverage, timeline continuity, frame count, format, duration agreement within one frame, caption bounds, overlay bounds, peaks, and long silences in both narration and final audio. Failed validation writes a report and marks the job failed.

## Cost controls

Every attempted provider request records status, available provider usage, response location, and request ID. No price is invented for unknown usage. By default `spending_ceiling_usd` is `null`; provider billing limits should be configured on the provider accounts.

An optional local budget uses conservative, operator-supplied **maximum request costs**, keyed by operation: `research`, `script`, `script_revision`, `source_audit`, `speech`, `visuals`, and `image`. If a ceiling is set, every requested operation must have a cap. Before each request, the pipeline reserves that cap and refuses requests that would exceed the job ceiling. Reservations remain charged against the local allowance even after timeouts or failed responses. This bounds the local ledger only to the extent that the supplied caps truly cover provider charges. It is not a provider-enforced billing limit. Model and tool prices must be checked when configuring those caps.

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
- [OpenRouter Image API and reference images](https://openrouter.ai/docs/guides/overview/multimodal/image-generation)
- [Requested image model: google/gemini-3.1-flash-lite-image](https://openrouter.ai/google/gemini-3.1-flash-lite-image)
- [OpenAI Docs: structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
- [OpenAI Docs: web search](https://developers.openai.com/api/docs/guides/tools-web-search)
- [ElevenLabs speech with timing](https://elevenlabs.io/docs/api-reference/text-to-speech/convert-with-timestamps/)
- [ElevenLabs model input limits](https://elevenlabs.io/docs/overview/models)
- [Gemini image generation](https://ai.google.dev/gemini-api/docs/image-generation)
- [Gemini generateContent API](https://ai.google.dev/api/generate-content)
- [FFmpeg filters](https://ffmpeg.org/ffmpeg-filters.html)
