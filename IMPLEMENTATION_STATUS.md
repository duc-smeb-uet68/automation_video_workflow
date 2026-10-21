# Implementation and verification status

Verified on 2026-10-21. The user has attempted live research and completed live script-mode speech plus 63 cue images; full live video acceptance remains incomplete.

## Transparent text overlays

The label renderer previously painted a cream rounded rectangle with alpha 245/255 behind text. That panel caused the background color mismatch shown in the user's screenshot. Labels now contain only red text on a fully transparent RGBA canvas, retaining the existing font, wrapping, positioning, and antialiased glyph edges.

Overlay checkpoints now include renderer version 2. Image generation, video rendering, and Kdenlive export share the same label-refresh path. Resuming a job refreshes older overlay PNGs and affected video outputs while reusing unchanged scene images, narration, and unaffected clips. Existing encoded videos require a rerender to apply the change.

Six transparency/compositing regression cases failed before the fix and passed afterward. Focused overlay/editor checks passed **14 tests** in **4.35 seconds**. The full offline suite, including added 1920×1080 coverage, passes **146 tests** in **64.28 seconds**, run with `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --tb=short --basetemp=D:/duc/startUp/demo/.test-tmp/transparent-full`. Checks cover glyph-only alpha at 640×360 and 1920×1080, long wrapped labels, exact background preservation across three colors, cache upgrades without new image/speech requests, Kdenlive prepared images, and real FFmpeg rerendering. A decoded final-video frame preserves a background sample within four RGB levels, and a subsequent cached rerun keeps the final MP4 hash unchanged. The decoded fixture frame was visually inspected and shows text without a panel. No live provider requests were made, and no retained user jobs were rerendered during verification.

## Text-only image generation experiment (initial verification)

Branch `codex/text-only-image-prompts` replaces reference-based generation with standalone text prompts. Visual planning retains the doodle style while selecting characters appropriate to the narration and repeating appearance details in scene descriptions. The pipeline checkpoints the complete ordered `{cue_id, prompt}` list in `image_prompts.json` before the first image request. Mocked requests confirm that OpenRouter and direct Gemini receive exactly the saved prompt text without image attachments. The pipeline no longer creates or reads character-reference images. Legacy `character_reference` configuration remains accepted and is ignored by setup and generation.

New generated jobs use `video_text_only_<input-hash>` IDs and record `image_strategy: "text_only"`. Legacy generated manifests default to the reference strategy and are rejected before job-file writes. Sixteen CLI regression cases cover script/topic jobs, absent/explicit legacy markers, and run, resume, validate, and editor-export commands; file contents and modification times remain unchanged. Fixture and imported rendering retain their existing workflow.

The full offline suite passes **135 tests**, run with `.venv/Scripts/python.exe -m pytest -q -p no:cacheprovider --tb=short --basetemp=D:/duc/startUp/demo/.test-tmp/text-only-full` in **54.09 seconds**. Focused image, visual, job-isolation, and configuration checks passed **72 tests** in **4.59 seconds**. Coverage includes text-only payloads for both image providers, exact prompt persistence before requests, dynamic characters, invalid prompt input rejection, cached prompt revalidation, independent image cache invalidation, retries, interrupted recovery, legacy preservation, and real FFmpeg integration. No authenticated generation requests were made for this experiment. Live image quality and character continuity remain unverified.

The remaining sections retain historical evidence from the reference-based workflow on `main`.

## Delivered before the text-only experiment

- Python CLI, Pydantic records, setup configuration, local credential loading, and a single-worker lock.
- Default script-file input (`run narration.txt` or `run --script narration.txt`) bypasses research, drafting, revision, and source auditing. It retains the 4,500–5,000-character gate, exact speech alignment, timed transcripts, and all downstream generation and validation. A hashed `input_script.txt` snapshot supports resume without the external file. Explicit `--mode topic` retains the original workflow; existing manifests keep their saved mode.
- Atomic manifests with per-stage input/output hashes, bounded retries, error history, usage records, and resumable image/clip checkpoints.
- Responses adapters for OpenRouter (default) and OpenAI (optional), covering bounded research, structured narration, one length revision, a separate sentence support audit, and batched visual planning. OpenRouter uses the requested `openai/gpt-oss-120b` model, managed Exa search with fixed search/result limits, and mandatory retrieved citations. Structured output is validated locally as well as requested from the provider.
- Exact 4,500–5,000-character gate before ElevenLabs speech generation, WAV conversion, and original-character alignment converted to words and sentences.
- Dynamic transcript cues, sequential OpenRouter images using the requested `google/gemini-3.1-flash-lite-image` model, an optional direct Gemini route, a reusable character reference, scene references, image checks, and Pillow labels. OpenRouter images use the dedicated Image API with one 1K/16:9 output and base64 reference images. Text and image generation share one OpenRouter key by default.
- FFmpeg rendering with absolute frame boundaries, stationary scenes, restrained 1.5% camera movement, two-pass narration normalization, optional ducked music, and separate SRT subtitles.
- Validation reports covering narration, images, text/timeline coverage, alignment/audio identity for production, cue timing, output format, frame count, duration, decoding, labels, clipping, and long silence.
- Legacy asset import and an optional manifest-driven Kdenlive export using the existing workspace template.
- An editable installation in `.venv`; `python -m video_automation` also works from the workspace.

## Saved verification artifacts

| Job | Result |
| --- | --- |
| `jobs/renderer_fixture/` | Portable synthetic fixture; 3.117-second input, three images, stationary/moving scenes, exact `38%` overlay; validation passes |
| `jobs/existing_assets_fixture/` | First three existing images with eight seconds of original narration; validation passes |
| `jobs/existing_full/` | Complete existing video: 288 seconds, 1920×1080, 30 fps, 8,640 frames, 81 images, 18 original approved motion clips; validation passes |

The full export contains `final.mp4`, `subtitles.en.srt`, `manifest.json`, `quality_report.json`, `audio_measurements.json`, `render_metadata.json`, and `project.kdenlive`. The MP4 is 39,157,710 bytes. Audio, video, and source narration are each 288 seconds. Final decoded peak is approximately −1.396 dBFS. No silence exceeding the configured three-second threshold was detected. The imported narration contains 4,710 normalized characters; imported mode does not claim fresh research, generated speech, or new word alignment.

A complete cached rerun kept the final MP4 hash unchanged and reused all 81 clip checkpoints. Kdenlive XML has 81 timeline clips and a 1920×1080/30 fps profile. Export structure is checked; interactive editor playback was not tested. Sample frames from both fixture and full export were visually inspected.

## Automated checks before the text-only experiment

The complete suite passes: **109 tests**, run with `D:/python/python.exe -m pytest -q -p no:cacheprovider --tb=short --basetemp=D:/duc/startUp/demo/.test-tmp/image-error-full` (58.06 seconds). It covers Unicode counting and boundaries, one successful or failed length revision, blocked downstream generation, exact narration/alignment identity, dynamic cue coverage, collapsed frame boundaries, output corruption, persisted retry limits, failed atomic replacement, budget reservations, provider HTTP handling, bounded research/source provenance, invalid-image retry, interrupted-image recovery, source-audit failures, full pipeline orchestration, local credential loading, setup failure reporting, worker locking, path bounds, labels, and Kdenlive structure. The OpenRouter tests additionally verify exact text/image model selection, endpoint/key selection for all text/image provider combinations, strict local schema checks, bounded Exa research, flat/nested citation provenance, missing citation rejection, setup checks, credential loading, provider/model cache invalidation, Image API HTTP contracts, ordered reference-image data URLs, usage attribution, malformed-image retries, direct Gemini compatibility, and image recovery without replacing successful images or voiceover. Research recovery tests cover cited prose, a single tool-free formatting pass, original-search provenance, rejection of invented URLs and invalid schemas, refusal handling, both search output record types, request cost reservations, and preservation of both raw responses.

Script-mode coverage verifies the default for configurations without `input_mode`, positional and `--script` inputs, explicit mode selection, invalid and ambiguous inputs, BOM/newline normalization, both accepted length boundaries, rejection before job creation, content-based job identity, mode/content collision rejection, snapshot corruption and restoration, transcript timestamps, resume after deleting the original file and derived script files without repeating successful provider calls, and compatibility with older topic manifests. The focused checks passed 17 tests. CLI help and the example configuration were also checked. No live generation requests were made for this change.

Real FFmpeg tests include both topic and script pipelines with offline provider contracts and 4,500–5,000-character narration, plus optional music, labels, final MP4 export, subtitles, validation, and cached reruns. Script-mode checks also reject missing speech alignment and invalid narration length during final validation. The offline speech contract uses synthetic audio and does not count as a live provider acceptance run.

## Visual-plan recovery verification

The user's script job `video_a603d46da62b` completed live ElevenLabs speech and decoding, producing 292.246354 seconds of narration and 75 derived cues. All four visual requests returned HTTP 200, but batch `visuals.002` returned two placeholder IDs instead of the requested 24 IDs (`cue_0049` through `cue_0072`). The previous pipeline saved that malformed batch as completed and checked cue coverage only after all batches, so `--retry-failed` could not repair the completed bad checkpoint.

Visual requests now constrain each batch's count, allowed cue IDs, and character IDs in the output schema. Generated plans are checked for schema, exact cue order, scene descriptions, and character IDs before checkpoint completion. Malformed output receives bounded retries. Completed cached plans are revalidated against the same contract; invalid cached plans are marked failed and retried without resetting persisted attempts. Valid plans remain reusable, and later batches regenerate when continuity inputs change. Provider refusals remain non-retryable.

Nine regression cases reproduced the old failure before the fix and passed afterward. Focused visual/provider checks passed 21 tests. They cover missing, duplicate, unexpected, and reordered IDs, blank descriptions, unknown characters, malformed JSON/schema output, retry exhaustion across resume, invalid older checkpoints, valid cache reuse, output schemas for both text providers, and non-retryable refusals.

An offline replay in a disposable copy of the actual job reused both speech checkpoints and recovered all 75 cue IDs using mocked visual responses. Only batches starting at `cue_0049` and `cue_0073` were requested again; the latter depends on the repaired batch's continuity data. Hashes of 16 original job files, including speech, alignment, checkpoints, and raw responses, remained unchanged. No live generation request was made while implementing or verifying this fix. The original job remains ready for the user to retry with `python -m video_automation resume video_a603d46da62b --retry-failed`; full live video acceptance is still pending.

## Image-request diagnostics

After the visual-plan recovery fix, the user's job `video_a603d46da62b` completed all 75 visual plans, its character reference, and cue images `cue_0001` through `cue_0063`. The request for `image.cue_0064` returned HTTP 400. The previous HTTP adapter discarded unsuccessful response bodies and request IDs, leaving only the status and a generic credentials/model/quota hint. The exact upstream reason for that historical failure cannot be recovered from the saved artifacts; successful earlier requests do not establish why the later request failed.

The HTTP adapter now saves selected, redacted error fields and available request IDs to `responses/*_error.json`, records the response path in usage, and includes provider and nested upstream reasons in CLI failures. It handles OpenRouter metadata, Gemini errors, and ElevenLabs detail records. Request-header values and configured provider keys are redacted before diagnostic fields are bounded to 1,000 characters. Non-JSON response bodies are not echoed. The retry policy is unchanged: HTTP 400 still stops rather than blindly repeating generation.

Six new regression cases failed before the diagnostic change and passed afterward. Focused error/redaction/recovery checks passed eight tests. Coverage includes nested upstream reasons, request IDs, credential redaction, truncation, non-JSON errors, and unchanged permanent/temporary error handling. Hash checks confirmed that all 67 completed speech and image artifacts, including the 63 cue images, match the existing manifest. No job files, generation settings, or successful checkpoints were changed, and no live generation requests were made. The image-generation failure itself remains unconfirmed until a new request succeeds or returns the detailed error. Resume continues with the saved speech and images using `python -m video_automation resume video_a603d46da62b --retry-failed`.

## Deferred live acceptance

Initial implementation verification sent no live generation requests. Setup checks originally found no configured generation credentials or `voice_id`, and a production invocation stopped before API calls. The subsequent OpenRouter integrations select `openai/gpt-oss-120b` for text and `google/gemini-3.1-flash-lite-image` for images in the local configuration. Both use `OPENROUTER_API_KEY`; a separate Gemini key is unnecessary for these defaults. Existing credential values are preserved.

The user's subsequent production attempt reached OpenRouter research and received HTTP 200 with `status: completed`, retrieved source citations, and Markdown prose instead of the requested JSON. Job `video_62d7bd255c16` stopped with `research: missing or invalid structured output (possible refusal)` before speech or images. Its saved usage reports one search and a provider-reported cost of USD 0.00744769. The raw response and failed stage are preserved. The adapter now attempts one tool-free formatting request for cited research prose and validates the result against the original schema and retrieved URLs. This repair was verified offline; no new authenticated generation request was made while implementing it.

An offline replay of the actual saved response with mocked formatting retained all five retrieved URLs and left the original response hash unchanged. The latest `doctor` run reports `ready_for_live_generation: true`, including credential presence and `voice_id`; it does not verify live speech/image access or credits.

Phases 1–4 are implemented and locally exercised. Phase 5 has local rendering/recovery evidence; the unattended live topic-to-video acceptance run remains deferred. Provider access, request compatibility on the user's accounts, actual generation costs, artistic continuity, and generated factual quality need that run. The source audit is model-based and is not an independent guarantee of factual accuracy.

To check setup and retry the existing job with the repaired adapter, run:

```powershell
python -m video_automation doctor
python -m video_automation resume video_62d7bd255c16 --retry-failed
```

The pending production job has no completed generation stages. Retrying may incur new provider charges. A successful full live run is still required to verify the remaining generation and acceptance conditions.
