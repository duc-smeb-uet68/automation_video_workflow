# Implementation and verification status

Verified on 2026-10-21. The user explicitly deferred provider credentials and the live generation run.

## Delivered

- Python CLI, Pydantic records, setup configuration, local credential loading, and a single-worker lock.
- Atomic manifests with per-stage input/output hashes, bounded retries, error history, usage records, and resumable image/clip checkpoints.
- Responses adapters for OpenRouter (default) and OpenAI (optional), covering bounded research, structured narration, one length revision, a separate sentence support audit, and batched visual planning. OpenRouter uses the requested `openai/gpt-oss-120b` model, managed Exa search with fixed search/result limits, and mandatory retrieved citations. Structured output is validated locally as well as requested from the provider.
- Exact 4,500–5,000-character gate before ElevenLabs speech generation, WAV conversion, and original-character alignment converted to words and sentences.
- Dynamic transcript cues, sequential Gemini images, a reusable character reference, scene references, image checks, and Pillow labels.
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

## Automated checks

The complete suite passes: **38 tests**. It covers Unicode counting and boundaries, one successful or failed length revision, blocked downstream generation, exact narration/alignment identity, dynamic cue coverage, collapsed frame boundaries, output corruption, persisted retry limits, failed atomic replacement, budget reservations, provider HTTP handling, bounded research/source provenance, invalid-image retry, interrupted-image recovery, source-audit failures, full pipeline orchestration, local credential loading, setup failure reporting, worker locking, path bounds, labels, and Kdenlive structure. The OpenRouter tests additionally verify exact default model selection, endpoint/key selection for both text providers, strict local schema checks, bounded Exa research, flat/nested citation provenance, missing citation rejection, setup checks, credential loading, and provider/model cache invalidation.

Real FFmpeg tests include the full pipeline with offline provider contracts and 4,500–5,000-character narration, plus optional music, labels, final MP4 export, subtitles, validation, and cached reruns. The offline speech contract uses synthetic audio and does not count as a live provider acceptance run.

## Deferred live acceptance

No live generation request was sent. Initial setup checks found no configured generation credentials or `voice_id`. A production invocation stopped before API calls and saved a clear setup failure report. The user chose to supply credentials later. The subsequent OpenRouter integration adds `OPENROUTER_API_KEY` to the local credential loader and `.env` template, selects `openai/gpt-oss-120b` in the local configuration, and requires only the selected text provider's key. Existing credential values are preserved.

Phases 1–4 are implemented and locally exercised. Phase 5 has local rendering/recovery evidence; the unattended live topic-to-video acceptance run remains deferred. Provider access, request compatibility on the user's accounts, actual generation costs, artistic continuity, and generated factual quality need that run. The source audit is model-based and is not an independent guarantee of factual accuracy.

To continue, fill `.env` locally, set `voice_id` in `config.local.json`, then run:

```powershell
python -m video_automation doctor
python -m video_automation run 'Who Watched the Baby While Ancient Humans Hunted?'
```

The pending production job has no completed generation stages and no recorded provider usage. It can start normally after setup.
