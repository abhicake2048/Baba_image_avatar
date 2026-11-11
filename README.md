# Private Portrait Studio

A localhost-only Gradio application that creates reference-conditioned images of one consenting adult through
OpenRouter. Users can choose between Nano Banana Pro and OpenAI GPT Image, review and edit a fixed multi-view prompt
set, and retain a private local history with prompt and generation metadata. Results are shown one at a time in a
previous/next carousel, with a download action for the currently visible image.

## Multi-view workflow

1. Enter the shared scene, wardrobe, expression, environment, and lighting request.
2. Click **Draft the five view prompts**. The text model creates a front waist-up prompt, left and right
   three-quarter prompts, a front full-body standing prompt, and a front full-body seated prompt.
3. Review or edit each prompt in the UI. Uncheck any view that should not be generated.
4. Click **Generate selected views**. The app submits one independent image request per selected prompt and labels
   each saved image by view.

The product UI groups these into four pose categories; the three-quarter category has one prompt for each side, so
selecting every category generates five images. Drafting prompts does not make a paid image request. Image generation
begins only after the reviewed prompts are submitted.

## Privacy and scope

- The server binds to `127.0.0.1` and Gradio sharing is disabled.
- Reference photos remain in the local `references` directory, but normalized in-memory copies are sent to the
  selected OpenRouter image provider for each generation.
- The text prompt enhancer receives the scene request and non-biometric profile notes only. It never receives photos.
- This application is for an adult who has explicitly authorized use of their likeness. It is not a face-swap,
  identity-document, sexual deepfake, or public-figure impersonation tool.
- Reference conditioning improves consistency but cannot guarantee a perfectly identical likeness in every output.

## Requirements

- Python 3.11 or newer
- Windows PowerShell for the included convenience scripts
- An OpenRouter API key with sufficient credit
- Four to ten approved JPEG, PNG, or WebP reference photographs

Choose clear, solo-subject photographs across front, profile, and three-quarter angles, with varied lighting and
expressions. Avoid filters, heavy occlusion, group photos, and low-resolution screenshots.

## Setup

From PowerShell in this directory:

```powershell
.\scripts\setup.ps1
```

Then:

1. Open `.env` and replace the placeholder `OPENROUTER_API_KEY`.
2. Review `profile.yaml`. Change both consent declarations to `true` only when they are accurate.
3. Add 4-10 approved images to `references`.
4. Start the app:

```powershell
.\scripts\run.ps1
```

The browser opens at `http://127.0.0.1:7860`. Do not use a Gradio public share link. The app validates the current
OpenRouter image catalog at startup and only displays configured models that advertise reference-image support.

## Quality presets

| Preset | Nano Banana Pro | GPT Image 2.6 Sunburst |
|---|---:|---:|
| Draft | 1K | `low` |
| Standard | 2K | `high` |
| High | 4K | `max` |

Each requested image is a separate `n=1` API call. At most two calls run concurrently. Paid image requests are not
automatically retried after a timeout, preventing accidental duplicate charges.

## Files and configuration

- `.env`: API key and model/path overrides; never committed.
- `profile.yaml`: consent assertion and non-biometric generation rules; never committed by default.
- `references/`: private source photographs; ignored by Git.
- `outputs/`: one folder per request containing images, metadata, and a download ZIP; ignored by Git.
- `logs/app.log`: rotating diagnostic log containing request settings, original and enhanced prompts, failures,
  and local output paths; ignored by Git. API keys and base64 image payloads are never logged.

Supported environment settings are documented in `.env.example`. Model defaults are:

- `google/gemini-3-pro-image`
- `openai/gpt-image-2.6-sunburst`
- `openai/gpt-4.1-mini` for prompt enhancement

OpenRouter's Image API returns base64 image bytes rather than durable hosted URLs. The app validates and stores those
bytes locally, then Gradio exposes a localhost-only download URL for the image currently visible in the carousel.

## Diagnostics

Open the **Diagnostics** tab in the app to inspect the latest backend log lines, or read `logs/app.log` directly.
The file rotates at 5 MB and keeps five backups. Set `LOG_LEVEL=DEBUG`, `INFO`, `WARNING`, or `ERROR` in `.env`, then
restart the app to change verbosity. Prompts are intentionally logged for debugging, so treat the log directory as
private data.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests use mocked OpenRouter responses and never require a real API key or upload real photographs. For manual visual
acceptance, use `docs/manual-quality-suite.md`.

## API references

- [OpenRouter unified Image API](https://openrouter.ai/blog/tutorials/image-generation/)
- [Nano Banana Pro on OpenRouter](https://openrouter.ai/google/gemini-3-pro-image)
- [GPT Image 2.6 Sunburst on OpenRouter](https://openrouter.ai/openai/gpt-image-2.6-sunburst)
- [Official OpenAI image-generation documentation](https://developers.openai.com/api/docs/guides/image-generation)

