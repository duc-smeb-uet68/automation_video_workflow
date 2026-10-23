# Educational video automation

A Python CLI turns a supplied narration file into a video title and description, timed speech, visual plans, images, and a validated video. External-image mode is the default: the CLI prepares narration, metadata, audio, alignment, subtitles, and standalone image prompts, then waits for you to generate the images elsewhere and copy them into the job's `images/` directory. Resume validates and adopts the supplied files before rendering. Optional API-image mode generates images automatically. Branch `codex/image-native-text` requests any needed short text directly inside the image; the planner specifies exact quoted wording, and the image AI chooses suitable placement. Script-file input is the default. Explicit topic mode also performs bounded research, script generation, and source auditing. Jobs use atomic local manifests, and one worker owns the jobs directory at a time.

## Setup

Use Python 3.11 or later and an FFmpeg distribution with `ffmpeg`, `ffprobe`, `libx264`, `loudnorm`, and `sidechaincompress`.

```powershell
python -m pip install -e '.[test]'
python -m video_automation init
Copy-Item .env.example .env
```

Fill `.env` locally with `OPENROUTER_API_KEY` and `ELEVENLABS_API_KEY`. OpenRouter remains the default text provider (`openai/gpt-oss-120b`), including visual prompt planning, so external-image mode still needs its text key. External-image mode requires no image-provider credentials and makes no image API calls or image cost reservations. With `text_provider: "openai"`, only `OPENAI_API_KEY` and `ELEVENLABS_API_KEY` are needed for preparation. API-image mode defaults to OpenRouter images (`google/gemini-3.1-flash-lite-image`); optional Gemini keys enable image credit fallback. Set `voice_id` to an ElevenLabs voice available to your account. Nonempty process environment variables take precedence over `.env`. Keys are never stored in manifests or execution logs.

`init` detects the existing `D:/kenlive/bin` FFmpeg installation when the tools are not on PATH. All other installations can set executable paths in configuration. Relative music and jobs paths resolve against the configuration file's directory. Legacy `character_reference` and `font_path` settings remain accepted but are ignored by setup checks and generation, and omitted from newly initialized configuration. `config.local.json` loads automatically when running from this directory; `--config` selects another setup file. `config.example.json` documents the main settings.

```powershell
python -m video_automation doctor
python -m video_automation run .\narration.txt
```

The narration file is the initial per-video content input. Model IDs, voice, resolution, retry limits, and optional music are configured once. The doodle style is fixed in code; the visual planner chooses characters appropriate to the narration. The project accepts any nonempty normalized script without a minimum or maximum character count. Speech generation still uses one timestamped request and checks the selected speech model's input limit before sending it.

## External images and API images

`image_mode` defaults to `"external"`, including when existing configuration omits it. It is independent of the script/topic `input_mode`.

```powershell
python -m video_automation run .\narration.txt
```

The command prepares `script.txt`, `video_metadata.json`, `audio/narration.wav`, `alignment.json`, `transcript.json`, `subtitles.en.srt`, and `image_prompts.json`. If images are missing or invalid, it returns success with `status: "waiting_for_images"`, the job ID, metadata path, prompt path, image directory, and report path. It does not mark the job completed, create placeholder images, or render a partial video. `external_images.json` lists every expected filename and any missing or invalid files.

Generate one image per saved prompt in your external AI, then copy the results as `images/cue_0001.png`, `images/cue_0002.png`, and so on inside that job. Filenames must match cue IDs. Images must be valid raster files, approximately 16:9 and at least 320×180; 1920×1080 matches the default output. External-image mode never overwrites these supplied files. Then run:

```powershell
python -m video_automation resume JOB_ID
```

Resume reuses valid preparation checkpoints, validates and records file hashes for supplied images, and renders when all expected images are ready. Partial or invalid copies leave the job waiting without consuming image attempts. Repeated resumes do not call image APIs. Replacing an image and resuming adopts the replacement and rebuilds affected clips and final output. Image-provider model/key settings do not invalidate external-image checkpoints. Text and speech can still incur charges if their own checkpoints need regeneration.

Existing jobs saved before this feature retain API-image mode when resumed. To explicitly use external images for an existing schema-version-2 native-text job:

```powershell
python -m video_automation resume JOB_ID --image-mode external
```

To generate images through provider APIs instead, select it explicitly; this can incur image-generation charges:

```powershell
python -m video_automation run .\narration.txt --image-mode api
python -m video_automation resume JOB_ID --image-mode api
python -m video_automation doctor --image-mode api
```

Image mode is saved in each manifest. Resume uses the saved mode unless explicitly overridden. Default job identities distinguish external-image jobs from API-image jobs for the same input, preserving older API job IDs. Reusing an explicit job ID with a different image mode is rejected; use the explicit resume override to change it.

## Input modes

Save the final narration as a UTF-8 plain text file, with or without a BOM. Both commands select script mode with the default configuration:

```powershell
python -m video_automation run .\narration.txt
python -m video_automation run --script .\narration.txt
```

Script mode skips research, drafting, and source auditing. You are responsible for reviewing the supplied narration. The file is normalized by converting CRLF/CR to LF and trimming outer whitespace. Empty normalized files stop before generation requests. There is no project-level narration-length requirement or automatic length revision in either input mode. The normalized text is sent unchanged to ElevenLabs, subject to the selected speech model's request limit. Exact character alignment, timed transcripts, visual planning, image preparation, rendering, and validation still run. The text provider is still required for metadata and visual planning.

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

In `image_mode: "api"`, image requests use OpenRouter's dedicated `POST /api/v1/images` API and the same `OPENROUTER_API_KEY` as text generation. Each request asks for one 1K image at 16:9. The request contains only a text prompt and generation settings; `input_references` is omitted. The direct Gemini route also sends only text parts. Returned base64 raster images pass the existing file/dimension checks before being checkpointed. Usage and available cost data are retained in the manifest. Missing, empty, malformed, or invalid images receive bounded retries; successful images and voiceover remain cached.

Change `openrouter_image_model` to choose another text-to-image model supporting the configured resolution/aspect ratio. Image settings have their own cache keys, including provider, model, Gemini fallback model, and native-text strategy version. Changing them regenerates affected images without invalidating unchanged research, narration, speech, or visual planning. Existing primary image checkpoints are adopted when adding the fallback setting, retaining artifact checks and persisted retry attempts. Changing credential values does not invalidate successful images. Each image depends on its exact prompt rather than earlier image files. Changes rebuild affected outputs within native-text jobs; older jobs remain untouched.

The visual planner writes complete scene descriptions in batches of 24 cues. Each description specifies visible characters and their appearance, actions, setting, objects, composition, and colors. Recurring characters repeat their appearance details. Previous scene descriptions may provide text-only context to the planner; the image model receives only the current prompt. The pipeline appends the fixed doodle style exactly once and checkpoints the complete ordered list in `image_prompts.json` before the first image request. Each record has `cue_id` and `prompt`; the saved prompt is the exact text sent to the image provider. There is no additional prompt-generation request. External-image mode pauses for the supplied files; API-image mode proceeds immediately. Full visual consistency remains a live-quality acceptance check.

In API-image mode, keep `image_provider` set to `openrouter` to use OpenRouter first. On an image HTTP 402 credit error, the current invocation switches to direct Gemini for that image and subsequent images. Temporary OpenRouter in-flight budget errors retain bounded retries. Authentication failures, rate limits, moderation errors, malformed images, and server or transport failures do not trigger provider fallback. Each new invocation tries OpenRouter first for uncached images, allowing a replenished account to resume primary generation. Text generation continues to use the configured text provider and has no Gemini fallback.

Configure the ordered Gemini key pool in the configuration directory's local `.env` file:

```dotenv
GEMINI_API_KEY=["your-first-key", "your-second-key"]
```

Use one line. A legacy single key or comma-separated list also works; duplicate keys are tried only once. `[]` disables the pool. Existing nonempty process environment values take precedence. `doctor` reports the fallback model and configured key count without printing keys. For direct Gemini in API-image mode (`image_provider: "gemini"`), an empty pool fails setup. External-image mode does not inspect or require the Gemini pool.

Gemini uses `gemini_model` for both fallback and direct generation, with the current code default `gemini-3.1-flash-lite-image`. An older local setup may retain another model and should be checked before enabling fallback. Requests send the same saved text-only prompt, with no reference images. The pool starts at its first key and retains a working key for subsequent images. HTTP 402 or 429 advances to the next key; other failures retain the existing permanent/temporary error policy. Pool exhaustion stops or uses the stage's bounded backoff retries for rate limits. A new invocation starts the pool again. [Gemini rate limits apply per project](https://ai.google.dev/gemini-api/docs/rate-limits), so multiple keys from one project share quota. Live model access, account credits, and image quality still require authenticated verification.

## Video title and description

Every script-mode and topic-mode run generates a title and description from the finalized, normalized narration using the configured text provider and model. This separate `metadata` phase runs after narration preparation and before speech, so its output remains available if speech fails or the job later waits for external images. It does not change the narration, speech request, image prompts, or subtitles.

The job's `video_metadata.json` contains exactly `title` and `description`. The title must be nonblank, occupy one line, and contain at most 100 characters. The description must be nonblank and contain at most 5,000 characters. The prompt requests the narration's language and a concise description of one to three short paragraphs grounded only in that narration. The normal CLI result includes the absolute artifact path as `metadata`, both at the external-image handoff and after completion. Metadata is saved locally; the workflow does not publish or upload it.

Metadata has its own checkpoint, keyed by narration, text provider/model, prompt, and schema. Resume reuses valid metadata. Missing or corrupt output is regenerated without invalidating unchanged speech, images, or renders. Resuming an existing compatible script/topic job without metadata adds this text-provider request while preserving valid downstream checkpoints. Fixture and imported jobs retain their three rendering phases and do not generate metadata. If `spending_ceiling_usd` is set, configure a conservative `request_cost_caps_usd.metadata` value before running or resuming.

## Artifacts

Each input receives a stable job directory, `jobs/video_native_text_<input-hash>/`. API-image jobs retain the original topic/script hashes; external-image jobs additionally hash their image mode to keep their default directories distinct. `--job-id` provides an optional administrative name. New manifests use `schema_version: 2`; generated jobs record `image_strategy: "native_text"`. Cue records have no `overlay_text` or `overlay_path` fields.

This workflow applies only to new jobs. Older manifests and incompatible generated strategies are rejected before strict cue parsing or any job-file writes, including explicit job IDs, resume, validation, and editor export. Continue schema-version-1 text-only overlay jobs on `codex/text-only-image-prompts`, and reference-based jobs on `main`. Existing fixtures and imported jobs also remain on their original branch. New fixtures and imports use schema version 2. Read-only `status` remains available for older jobs. No existing job is migrated or regenerated automatically.

| Artifact | Contents |
| --- | --- |
| `manifest.json` | Job state, image strategy, cues, timing, attempts, input hashes, artifact hashes, errors, usage |
| `input_script.txt` | Normalized input snapshot for supplied-script jobs; retained for resume |
| `sources.json` | Retrieved source URLs and supported claims; topic mode only |
| `draft.json` | Generated narration draft; topic mode only |
| `script.txt`, `script.json` | Exact normalized narration and stable paragraph/sentence IDs |
| `source_audit.json` | Separate model audit of every sentence's source support; topic mode only |
| `video_metadata.json` | Generated title and description based on the finalized narration; script and topic modes |
| `audio/narration.wav`, `alignment.json` | Final decoded audio and character, word, sentence timing |
| `transcript.json`, `visuals/` | Permanent cue IDs and scene plans |
| `external_images.json` | Expected image paths, valid image count, and missing/invalid file details in external-image mode |
| `image_prompts.json` | Ordered `{cue_id, prompt}` list containing the exact complete text for each image request |
| `images/` | One validated image per cue, with any requested text drawn by the image AI |
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

`run`, `resume`, `fixture`, and `import-assets` show live progress on stderr. Stdout retains the final JSON result. Generated jobs have eight phases: narration preparation, metadata generation, speech, visual planning, image generation or external-image adoption, clip rendering, audio finishing, and validation. Fixture and imported jobs have the final three phases. Research, drafting, and source auditing appear as tasks within narration preparation.

Progress shows the active task, attempt limit, cached tasks, and completed batches, images, or clips when their totals are known. Cue-derived totals remain dynamic. Counts describe completed work rather than estimated time remaining; phases have different costs and durations. Clip counts do not mark rendering complete until rough-cut assembly also finishes. Completion is reported only after validation passes.

Each phase transition and checkpoint event updates the job's atomic `progress.json` snapshot. Inspect it from another terminal while the worker runs:

```powershell
python -m video_automation status video_native_text_JOB_HASH
```

`status` also accepts a job folder or manifest path. It reads saved progress and the authoritative manifest status without acquiring the worker lock, making generation requests, or modifying files. Jobs predating progress tracking return their manifest status with `progress_available: false`. Snapshots include an update timestamp and elapsed seconds for the current invocation. During a long provider or FFmpeg call, the last reported task remains active. Abrupt process termination may leave a running status until the next resume.

On resume, cached stages and cue checkpoints are verified and reported as reused work; the phase counters describe that invocation. Failure and interruption preserve the last phase and completed counts. If saving a progress snapshot fails, the CLI warns once and continues using normal workflow checkpoints. A disconnected progress stream does not cause generation retries.

## Resume and retries

```powershell
python -m video_automation resume video_native_text_JOB_HASH
python -m video_automation resume video_native_text_JOB_HASH --retry-failed
python -m video_automation validate video_native_text_JOB_HASH
```

`resume` accepts a job name, job folder, or manifest path. Generated jobs must carry schema version 2 and the native-text strategy marker on this branch. Completed stages are reused only when their input hashes and output file hashes match. A missing or corrupt output is rebuilt. Changed prompts and settings invalidate dependent work. Image generation and clip rendering checkpoint each cue individually. Successful images and speech remain available if a later image fails.

Temporary HTTP failures, transport errors, missing image responses, and invalid images receive bounded exponential-backoff retries. Authentication, unsupported content, failed source audits, and invalid narration fail clearly. The retry counter persists across restarts. `--retry-failed` explicitly starts one new bounded batch for failed or interrupted stages; successful stages remain cached. An interrupted request can still be billable if the provider completed it before the client lost the response.

Visual batches must contain every requested cue ID exactly once in order, with valid scene descriptions, character IDs, and motion settings. The requested output schema constrains the cue count and allowed IDs for each batch. Malformed visual output is rejected before a checkpoint is marked completed and receives the existing bounded retries. Visual responses with `status: "incomplete"` and `incomplete_details.reason: "max_output_tokens"` also receive those retries, and the error reports the token-limit reason. Partial output is never approved. Refusals and other incomplete-response reasons still stop immediately. Cached batches from older runs are revalidated before reuse. Invalid cached batches are retried within their remaining attempt budget, while valid earlier batches and completed speech remain cached. Later batches are regenerated when their continuity inputs change. Raw provider responses remain available for diagnosis.

The recorded character count uses Python Unicode characters after CRLF/CR normalization and outer whitespace trimming. Only `script.txt` is counted and spoken. Narration must be nonempty; no character-range gate or automatic length revision applies. Final validation retains the historical `narration_length` report key to check nonempty text, normalization, and the recorded character count, with `enforced: false` indicating that length bounds are disabled. Original ElevenLabs alignment must match the exact narration in both modes. A mismatch fails clearly rather than inventing timestamps. Timing is frozen after audio decoding; later speech changes require regeneration and alignment.

Visual cues follow aligned words and sentence boundaries. Their count is dynamic. Frame durations derive from adjacent absolute boundaries. Boundaries that collapse onto one frame are merged without dropping text. The first visual starts at frame zero; the last image holds until the final recording ends. Output may pad the final audio by less than one frame to match the encoded video.

The doodle style lives in code. Characters and appearance descriptions are chosen from the narration and repeated in standalone prompts. This branch neither creates nor reads character-reference or scene-reference images. Only `hold` and a centered 1.5% `slow_push` are supported. The planner selects `hold` for text-heavy and diagram scenes. Image checks cover file validity, dimensions, and coverage; artistic correctness is outside this MVP.

The visual planner requests text only when short names, percentages, important numbers, or labels help explain a scene. It places exact quoted wording in `scene_description`, and the image AI draws the requested text with readable size and contrast in a suitable location that avoids important scene content and frame edges. Scenes without useful text omit text instructions. The complete image prompt, including these instructions, is saved before generation and sent unchanged to OpenRouter or Gemini. There are no separate label PNGs, overlay checkpoints, or text-compositing steps. Text is part of the source image in both the MP4 and Kdenlive export. Generated spelling, numbers, readability, and placement are best effort and require visual review; there is no OCR verification or deterministic text guarantee. Narration subtitles remain a separate SRT file.

Audio finishing uses two-pass loudness normalization. Optional local music loops, fades, and ducks under speech. Validation decodes the final file and checks image coverage, spoken-text coverage, timeline continuity, frame count, format, duration agreement within one frame, caption bounds, peaks, and long silences in both narration and final audio. Failed validation writes a report and marks the job failed. Missing or invalid external images instead leave preparation in `waiting_for_images` before rendering or validation begins.

## Cost controls

Every attempted provider request records status, available provider usage, response location, and request ID. No price is invented for unknown usage. By default `spending_ceiling_usd` is `null`; provider billing limits should be configured on the provider accounts.

An optional local budget uses conservative, operator-supplied **maximum request costs**, keyed by operation: `research`, `script`, `source_audit`, `metadata`, `speech`, `visuals`, and `image`. If a ceiling is set, every requested operation must have a cap. Before each request, the pipeline reserves that cap and refuses requests that would exceed the job ceiling. Each OpenRouter attempt, Gemini fallback request, and Gemini key change reserves a separate `image` cap; configure that cap to cover both image providers. Reservations remain charged against the local allowance even after timeouts or failed responses. This bounds the local ledger only to the extent that the supplied caps truly cover provider charges. It is not a provider-enforced billing limit. Model and tool prices must be checked when configuring those caps.

## Renderer fixtures and existing assets

```powershell
python -m video_automation fixture
python -m video_automation resume renderer_fixture
```

This portable three-scene fixture uses synthetic tone audio, verifies stationary/moving scenes and a `38%` label already drawn inside a source image, and requires no API credentials. It does not test AI spelling or placement. It is labeled `fixture`; the production narration-length requirement is explicitly excluded from its report. Use a new job ID if the default fixture already exists from an older branch.

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

This writes `project.kdenlive` from the job manifest using the existing workspace template. Image count, narration path, profile, scene durations, and selected camera movement are dynamic. Prepared image plates preserve any text already present in the source image without adding text. The existing builder and original Kdenlive project remain available. The MP4 pipeline does not require an editor.

## Verification

```powershell
python -m pytest -q
```

If a restricted Windows environment cannot create its default temporary directory, create `.test-tmp` in the workspace and use an absolute `--basetemp` path under it. Tests use offline provider contracts for nonempty normalized narration, audits, exact speech text, word coverage, optional native text, job isolation, caching, retries, budget checks, and interrupted-image recovery. Real FFmpeg integration covers short and long narration, music ducking, preservation of source-image text, encoding, validation, and cached reruns. Offline provider tests do not prove live account access or generated content quality.

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
