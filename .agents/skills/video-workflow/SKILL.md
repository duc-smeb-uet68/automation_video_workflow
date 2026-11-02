---
name: video-workflow
description: Create or resume videos in this repository using Codex built-in image generation to supply external scene images, then render and validate with the existing Python CLI.
---

# Video workflow with Codex images

Use the repository's Python pipeline for narration, metadata, speech, timing, visual planning, rendering, and validation. Use Codex's built-in image generation for the external-image handoff. Run this workflow when the user requests a video, not merely an explanation or skill installation.

## Inputs and setup

- Read the repository's `AGENTS.md` and relevant README instructions. Run commands from the repository root. Prefer `.venv/Scripts/python.exe` on Windows; use the project's configured interpreter elsewhere.
- Accept a narration file, an explicitly requested topic, or an existing job ID/folder/manifest. If the target cannot be determined from the request and context, ask for it before generation. Do not select an arbitrary job or invent narration.
- Retain the user's configuration, speech provider, optional music, and thumbnail selection. Put any supplied global `--config PATH` before the subcommand in every command. Resolve jobs from CLI output and configuration; do not assume they live under the default `jobs/` path.
- Complete the image-tool discovery below before starting new preparation that may incur provider charges. Follow the installed `imagegen` skill in built-in mode. Do not use its CLI fallback, OpenAI image APIs, OpenRouter images, Gemini images, Runway, or browser automation for this workflow.
- Built-in images count toward Codex usage limits and do not need `OPENAI_API_KEY`. Text planning and speech still use the configured providers and may incur charges. The Python spending ledger does not track Codex image usage. Do not promise a free or unlimited run.

## Discover the built-in image tool

Read the installed `imagegen` skill before deciding that generation is unavailable. Check directly exposed tools and any tool registry or deferred-tool discovery provided by the current session. The built-in namespace can be `image_gen.imagegen`; in a `functions.exec` environment its normalized name can be `image_gen__imagegen` on the `tools` object. A visible Runway plugin is not evidence that built-in ImageGen is absent.

When `functions.exec` exposes `ALL_TOOLS` and `tools`, run this metadata-only check there, not in PowerShell or a separate JavaScript runtime:

```javascript
text(ALL_TOOLS
  .filter(({ name }) => /image_gen|imagegen/i.test(name))
  .map(({ name }) => ({ name, callable: typeof tools[name] === "function" })));
```

If the built-in tool is exposed, use its actual schema and the host's tool-call instructions for the image phase. Tool discovery makes no generation request and does not prove that account access or generation will succeed. If the session uses a different discovery mechanism, use that mechanism instead of assuming these JavaScript globals exist.

Only report the tool as unavailable after checking the surfaces the session actually exposes. Name the checks and missing tool in the report. An installed skill supplies instructions; it cannot enable a tool withheld by the host. If the tool is genuinely absent, stop before paid preparation, suggest a new app chat with both `$imagegen` and `$video-workflow`, and explain that this may clarify routing but does not guarantee tool availability. Do not install an API client or request an API key to repair a missing built-in tool.

## Prepare or resume the job

For a new supplied-script job:

```powershell
& .\.venv\Scripts\python.exe -m video_automation run --script .\narration.txt --image-mode external
```

Use `run --mode topic 'TOPIC' --image-mode external` only when the user explicitly requests topic mode. Preserve the supplied script; do not rewrite it or bypass narration/alignment gates.

For an existing job, inspect its manifest and `status` first. Generated jobs must use schema version 2, `image_strategy: "native_text"`, and external-image mode. Never manually edit the manifest to force compatibility. A historical schema-version-2 manifest without `image_mode` retains API mode. If an API job is selected, obtain an explicit instruction to switch it before using `resume JOB_ID --image-mode external`. Do not resume an API job through its saved image provider. Fixture/imported jobs do not need this generation workflow.

For a compatible external-image job, use:

```powershell
& .\.venv\Scripts\python.exe -m video_automation resume JOB_ID
```

Read the command's exit code and final JSON on stdout; progress is on stderr. `waiting_for_images` with exit code 0 is the expected handoff. A failed preparation must be resolved before generating scenes. Preserve all completed checkpoints. Use `--retry-failed` only when the user authorizes another bounded retry batch; do not repeatedly reset attempts.

## Supply the images

1. Read the complete ordered `image_prompts.json` and `external_images.json` from this job. Match every `cue_id` to the report's expected path, normally `images/<cue_id>.png`. The prompt count determines the number of scene images; never hardcode it.
2. Validate existing images with `video_automation.media.check_image(Path(...))` or the CLI's refreshed adoption report. Preserve every valid image unless the user requests replacement. Do not trust a stale report after files have changed. Keep an invalid existing file as a non-overwriting backup before replacing it.
3. Generate missing or invalid scenes sequentially, with one built-in image call per cue. Use the saved prompt verbatim, adding only the 16:9 framing requirement if needed. Preserve quoted labels, appearance details, and the fixed style. Do not add reference images, combine scenes into a grid, create placeholders, or substitute programmatic drawings.
4. Retrieve each actual generated file from the tool's returned artifact or documented output location, then copy it into this job under its exact cue filename. Built-in output may initially be under `$CODEX_HOME/generated_images/`; do not assume a destination argument or leave project assets only there. If the output cannot be retrieved, stop and report that limitation.
5. Save a real PNG, not another format renamed to `.png`. Request approximately 16:9, preferably 1920x1080; repository validation requires at least 320x180. Use `media.check_image` on the saved destination. Inspect the generated scene for obvious content or label errors. Do not claim OCR verification. Stop on an invalid output rather than repeatedly consuming generation attempts without user direction.
6. Save and validate each image before starting the next. Do not change prompts, cues, timing, stage records, or hashes by hand. Let `resume` adopt files and invalidate affected renders.

If thumbnails are enabled and `thumbnail_prompt.json` exists, use its complete `prompt` to generate a missing/invalid `thumbnail.png` through the same built-in path. Preserve a valid thumbnail. A thumbnail failure or missing plan does not block a video whose scene images are ready. Do not enable thumbnails for a job where they are disabled.

## Finish and report

After all scene files are saved, run `resume JOB_ID` with the same configuration. The pipeline validates/adopts images, renders, finishes audio, and validates the final video. If the job was already completed and unchanged, inspect its current outputs without unnecessary regeneration.

Report completion only when `manifest.json` says `completed`, the latest `quality_report.json` has `passed: true`, and `final.mp4` exists. Report absolute paths to the final video, subtitles, metadata, and any ready thumbnail, together with the job ID and image counts. Identify built-in Codex image generation as the image source and the saved prompt file as the prompt set. State which validation ran; do not describe this as live topic-mode acceptance unless that mode actually completed.

On image-tool failure, usage limits, interruption, or unavailable output files, preserve saved assets and report the job ID, current status, exact remaining cue IDs/filenames, and the next action. Do not switch to a paid image API, regenerate valid images, or claim completion. Retry transient tool failures only within an explicit tool-provided retry instruction; otherwise stop at the first failed image call.
