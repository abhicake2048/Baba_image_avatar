from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import gradio as gr

from .logging_config import get_logger, tail_log
from .models import ViewPrompt
from .runtime import AppRuntime


THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="violet",
    neutral_hue="slate",
    radius_size="lg",
)

CSS = """
body { background: var(--body-background-fill); }
.gradio-container { background: transparent !important; }
.app-shell { max-width: 1240px; margin: 0 auto; padding: 24px 18px 48px; gap: 20px !important; }
.hero-panel {
  position: relative; overflow: hidden; padding: 34px 36px; border: 1px solid var(--border-color-primary);
  border-radius: 26px; background: linear-gradient(135deg, color-mix(in srgb, var(--primary-500) 15%, var(--block-background-fill)), color-mix(in srgb, var(--secondary-500) 12%, var(--block-background-fill)));
  box-shadow: var(--shadow-drop-lg);
}
.hero-panel::after { content: ""; position: absolute; width: 260px; height: 260px; right: -90px; top: -130px; border-radius: 999px; background: color-mix(in srgb, var(--primary-400) 18%, transparent); filter: blur(2px); }
.hero-copy { position: relative; z-index: 1; max-width: 760px; }
.eyebrow { margin: 0 0 8px; color: var(--primary-600); font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: 12px; }
.hero-title { margin: 0; color: var(--body-text-color); font-size: clamp(30px, 5vw, 52px); line-height: 1.03; letter-spacing: -.035em; }
.hero-subtitle { margin: 14px 0 0; max-width: 680px; color: var(--body-text-color-subdued); font-size: 16px; line-height: 1.65; }
.hero-badges { display: flex; gap: 9px; flex-wrap: wrap; margin-top: 18px; }
.hero-badge { display: inline-flex; align-items: center; gap: 6px; padding: 7px 11px; border-radius: 999px; background: color-mix(in srgb, var(--block-background-fill) 78%, transparent); border: 1px solid var(--border-color-primary); color: var(--body-text-color); font-size: 12px; font-weight: 600; }
.section-heading h2 { margin-bottom: 2px !important; letter-spacing: -.02em; }
.section-heading p { color: var(--body-text-color-subdued); }
.studio-panel { border: 1px solid var(--border-color-primary) !important; border-radius: 22px !important; padding: 20px !important; background: var(--block-background-fill) !important; box-shadow: var(--shadow-drop); }
#prompt-box textarea { min-height: 184px !important; font-size: 16px !important; line-height: 1.65 !important; }
#generate-button { min-height: 52px; font-weight: 700; font-size: 16px; box-shadow: 0 10px 24px color-mix(in srgb, var(--primary-500) 24%, transparent); }
#status-card { padding: 12px 16px; border-radius: 16px; border: 1px solid var(--border-color-primary); background: color-mix(in srgb, var(--primary-500) 6%, var(--block-background-fill)); min-height: 52px; }
.output-panel { border: 1px solid var(--border-color-primary) !important; border-radius: 24px !important; padding: 18px !important; background: var(--block-background-fill) !important; box-shadow: var(--shadow-drop-lg); }
#carousel-image { border-radius: 18px; overflow: hidden; background: color-mix(in srgb, var(--neutral-800) 8%, var(--block-background-fill)); }
#carousel-image img { object-fit: contain !important; max-height: 680px !important; }
.carousel-controls { align-items: center !important; gap: 10px !important; }
.carousel-counter { min-width: 100px; text-align: center; padding-top: 10px; color: var(--body-text-color-subdued); font-weight: 700; }
.download-current { min-width: 190px; }
.privacy-note { padding: 12px 15px; border-left: 3px solid var(--primary-500); border-radius: 12px; background: color-mix(in srgb, var(--primary-500) 7%, var(--block-background-fill)); color: var(--body-text-color-subdued); }
.prompt-workbench { gap: 14px !important; }
.prompt-card { border: 1px solid var(--border-color-primary) !important; border-radius: 18px !important; padding: 15px !important; background: color-mix(in srgb, var(--primary-500) 3%, var(--block-background-fill)) !important; }
.prompt-card textarea { line-height: 1.5 !important; }
.prompt-step { padding: 10px 14px; border-radius: 14px; background: color-mix(in srgb, var(--secondary-500) 6%, var(--block-background-fill)); border: 1px solid var(--border-color-primary); }
.bottom-tabs { margin-top: 6px; }
.diagnostics-note { color: var(--body-text-color-subdued); font-size: 13px; }
#log-viewer textarea { font-family: var(--font-mono); font-size: 12px !important; line-height: 1.55 !important; }
@media (max-width: 720px) {
  .app-shell { padding: 12px 8px 32px; }
  .hero-panel { padding: 25px 22px; border-radius: 20px; }
  .studio-panel, .output-panel { padding: 13px !important; border-radius: 18px !important; }
  #carousel-image img { max-height: 520px !important; }
}
"""


def _history_payload(runtime: AppRuntime) -> tuple[list[tuple[str, str]], list[str]]:
    items = runtime.storage.list_history()
    gallery = [(str(item.thumbnail), item.caption) for item in items]
    paths = [str(item.request_dir) for item in items]
    return gallery, paths


_VIEW_LABELS = {
    "front-headshot": "Front headshot / waist-up",
    "three-quarter-left": "Left three-quarter view",
    "three-quarter-right": "Right three-quarter view",
    "full-body-front": "Full-body front standing",
    "full-body-seated": "Full-body front seated",
}


def _view_label_from_path(path: str) -> str | None:
    stem = Path(path).stem
    for key, label in _VIEW_LABELS.items():
        if stem.endswith(key):
            return label
    return None


def carousel_view(paths: list[str] | None, index: int) -> tuple[str | None, int, str, str | None]:
    if not paths:
        return None, 0, "No images yet", None
    safe_index = index % len(paths)
    current = paths[safe_index]
    view_label = _view_label_from_path(current)
    prefix = f"**{view_label}** · " if view_label else ""
    return current, safe_index, f"{prefix}Image {safe_index + 1} of {len(paths)}", current


def build_app(runtime: AppRuntime) -> gr.Blocks:
    settings = runtime.settings
    model_names = list(runtime.available_models)
    log_path = settings.log_dir / "app.log"
    logger = get_logger()
    safe_app_title = html.escape(settings.app_title)

    async def draft_view_prompts(
        prompt: str,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[Any, ...]:
        progress(0.15, desc="Creating the fixed view prompts")
        try:
            prompt_set = await runtime.generation.prepare_pose_prompts(prompt)
        except Exception as exc:
            logger.exception("ui.prompt_set_failed error=%s", exc)
            raise gr.Error(str(exc)) from exc
        progress(1.0, desc="Prompts ready for review")
        fallback_note = " A deterministic fallback was used." if prompt_set.used_fallback else ""
        return (
            prompt_set.front_headshot,
            prompt_set.three_quarter_left,
            prompt_set.three_quarter_right,
            prompt_set.full_body_front,
            prompt_set.full_body_seated,
            f"**Five prompts ready.** Review or edit every selected view before generation.{fallback_note}",
            tail_log(log_path),
        )

    async def generate_images(
        base_prompt: str,
        model_name: str,
        aspect_ratio: str,
        quality: str,
        use_front: bool,
        front_prompt: str,
        use_left: bool,
        left_prompt: str,
        use_right: bool,
        right_prompt: str,
        use_standing: bool,
        standing_prompt: str,
        use_seated: bool,
        seated_prompt: str,
        progress: gr.Progress = gr.Progress(),
    ) -> tuple[Any, ...]:
        model_id = runtime.available_models.get(model_name)
        if not model_id:
            raise gr.Error("That model is not currently available with reference-image support.")
        candidates = [
            (use_front, "front-headshot", "Front headshot / waist-up", front_prompt),
            (use_left, "three-quarter-left", "Left three-quarter view", left_prompt),
            (use_right, "three-quarter-right", "Right three-quarter view", right_prompt),
            (use_standing, "full-body-front", "Full-body front standing", standing_prompt),
            (use_seated, "full-body-seated", "Full-body front seated", seated_prompt),
        ]
        selected: list[ViewPrompt] = []
        for enabled, key, label, view_prompt in candidates:
            if not enabled:
                continue
            if not view_prompt or len(view_prompt.strip()) < 20:
                raise gr.Error(f"The enabled prompt for {label} must contain at least 20 characters.")
            selected.append(ViewPrompt(key=key, label=label, prompt=view_prompt.strip()))
        if not selected:
            raise gr.Error("Select at least one view to generate.")
        progress(0.08, desc=f"Generating {len(selected)} selected view(s)")
        try:
            result = await runtime.generation.generate_pose_views(
                base_prompt=base_prompt,
                model=model_id,
                aspect_ratio=aspect_ratio,
                quality_preset=quality,
                views=selected,
            )
        except Exception as exc:
            logger.exception(
                "ui.pose_generation_failed model=%s selected_views=%s error=%s",
                model_id,
                [view.key for view in selected],
                exc,
            )
            raise gr.Error(str(exc)) from exc

        progress(1.0, desc="Complete")
        paths = [str(image.path) for image in result.images]
        current, current_index, counter, download = carousel_view(paths, 0)
        logger.info(
            "ui.output_shown %s",
            json.dumps(
                {
                    "request_id": result.request_id,
                    "visible_output": current,
                    "selected_views": [view.key for view in selected],
                    "all_outputs": paths,
                },
                ensure_ascii=False,
            ),
        )
        if result.images and result.failures:
            status = (
                f"**Partial success** · Generated {len(result.images)} of {len(selected)} selected views. "
                + " ".join(result.failures)
            )
        elif result.images:
            status = (
                f"**Generation complete** · {len(result.images)} selected view(s) · "
                f"Request `{result.request_id}`"
            )
        else:
            status = "**Generation failed** · " + " ".join(result.failures)
        history_gallery, history_paths = _history_payload(runtime)
        return (
            current,
            paths,
            current_index,
            counter,
            download,
            result.enhanced_prompt,
            status,
            history_gallery,
            history_paths,
            tail_log(log_path),
        )

    def move_carousel(paths: list[str] | None, index: int, step: int) -> tuple[Any, ...]:
        current, new_index, counter, download = carousel_view(paths, int(index) + step)
        if current:
            logger.info("ui.carousel_display index=%s output_path=%s", new_index, current)
        return current, new_index, counter, download

    def previous_image(paths: list[str] | None, index: int) -> tuple[Any, ...]:
        return move_carousel(paths, index, -1)

    def next_image(paths: list[str] | None, index: int) -> tuple[Any, ...]:
        return move_carousel(paths, index, 1)

    def refresh_history() -> tuple[list[tuple[str, str]], list[str]]:
        return _history_payload(runtime)

    def refresh_logs() -> str:
        return tail_log(log_path)

    def select_history(paths: list[str] | None, evt: gr.SelectData) -> tuple[Any, ...]:
        if not paths or evt.index is None:
            return "", "No history item selected.", None, [], 0, "No images yet", None
        index = evt.index[0] if isinstance(evt.index, tuple) else int(evt.index)
        if index < 0 or index >= len(paths):
            return "", "The selected history item is no longer available.", None, [], 0, "No images yet", None
        directory = Path(paths[index])
        files = [str(path) for path in sorted(directory.glob("image-*.*"))]
        current, carousel_index, counter, download = carousel_view(files, 0)
        logger.info("ui.history_loaded request_dir=%s outputs=%s", directory, files)
        return (
            str(directory),
            f"Loaded `{directory.name}` into the carousel above.",
            current,
            files,
            carousel_index,
            counter,
            download,
        )

    def delete_selected(path: str) -> tuple[Any, ...]:
        empty_carousel = (None, [], 0, "No images yet", None)
        if not path:
            return *refresh_history(), "", "Select a history item first.", *empty_carousel
        try:
            runtime.storage.delete_request(path)
            logger.info("history.deleted request_dir=%s", path)
        except Exception as exc:
            logger.exception("history.delete_failed request_dir=%s error=%s", path, exc)
            return *refresh_history(), "", f"Could not delete the selection: {exc}", *empty_carousel
        return *refresh_history(), "", "Selected request deleted permanently.", *empty_carousel

    def delete_all(confirmed: bool) -> tuple[Any, ...]:
        empty_carousel = (None, [], 0, "No images yet", None)
        if not confirmed:
            return (
                *refresh_history(),
                False,
                "",
                "Tick the confirmation box before deleting all history.",
                *empty_carousel,
            )
        deleted_count = runtime.storage.delete_all()
        logger.info("history.deleted_all count=%s", deleted_count)
        return (
            *refresh_history(),
            False,
            "",
            f"Deleted {deleted_count} request folder(s) permanently.",
            *empty_carousel,
        )

    with gr.Blocks(title=settings.app_title) as demo:
        carousel_paths = gr.State([])
        carousel_index = gr.State(0)
        history_state = gr.State([])
        selected_history = gr.State("")

        with gr.Column(elem_classes="app-shell"):
            gr.HTML(
                f"""
                <section class="hero-panel">
                  <div class="hero-copy">
                    <p class="eyebrow">Private · Local · Reference conditioned</p>
                    <h1 class="hero-title">{safe_app_title}</h1>
                    <p class="hero-subtitle">Describe one scene, review five fixed camera-view prompts, edit any detail, and generate a consistent multi-view portrait set.</p>
                    <div class="hero-badges">
                      <span class="hero-badge">● Localhost only</span>
                      <span class="hero-badge">◆ Five fixed views</span>
                      <span class="hero-badge">↓ One-click downloads</span>
                    </div>
                  </div>
                </section>
                """
            )

            gr.Markdown("## Step 1 · Describe the portrait set\nYour scene details will be preserved across every fixed camera view.", elem_classes="section-heading")
            with gr.Row(equal_height=True):
                with gr.Column(scale=7, elem_classes="studio-panel"):
                    prompt = gr.Textbox(
                        label="Scene direction",
                        placeholder="Seated in a modern library, wearing a red shirt, soft window light, editorial photography...",
                        lines=7,
                        max_lines=14,
                        elem_id="prompt-box",
                    )
                with gr.Column(scale=5, elem_classes="studio-panel"):
                    model = gr.Radio(model_names, value=model_names[0], label="Image model")
                    with gr.Row():
                        aspect = gr.Dropdown(
                            ["Auto", "1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16", "21:9"],
                            value="Auto",
                            label="Aspect ratio",
                        )
                    quality = gr.Radio(["Draft", "Standard", "High"], value="Standard", label="Quality")
                    gr.Markdown(
                        "Reference photos stay out of the prompt-enhancement call. They are sent only to the selected image model.",
                        elem_classes="privacy-note",
                    )

            draft_prompts = gr.Button("Draft the five view prompts", variant="primary", size="lg")
            prompt_status = gr.Markdown(
                "**Step 1:** Draft the prompts. **Step 2:** Review, edit, or deselect views. **Step 3:** Generate.",
                elem_classes="prompt-step",
            )

            gr.Markdown(
                "## Step 2 · Review and edit the fixed views\nAll prompts remain editable. The three-quarter category includes one prompt for each side.",
                elem_classes="section-heading",
            )
            with gr.Column(elem_classes="prompt-workbench"):
                with gr.Column(elem_classes="prompt-card"):
                    use_front = gr.Checkbox(value=True, label="Generate category 1 · Front headshot / waist-up")
                    front_prompt = gr.Textbox(
                        label="Front headshot / waist-up prompt",
                        lines=5,
                        placeholder="Draft the prompts to populate this view, then edit if needed.",
                    )
                with gr.Column(elem_classes="prompt-card"):
                    gr.Markdown("### Category 2 · Three-quarter angles")
                    with gr.Row():
                        with gr.Column():
                            use_left = gr.Checkbox(value=True, label="Generate left three-quarter view")
                            left_prompt = gr.Textbox(
                                label="Left three-quarter prompt",
                                lines=5,
                                placeholder="The subject's left side will be more visible.",
                            )
                        with gr.Column():
                            use_right = gr.Checkbox(value=True, label="Generate right three-quarter view")
                            right_prompt = gr.Textbox(
                                label="Right three-quarter prompt",
                                lines=5,
                                placeholder="The subject's right side will be more visible.",
                            )
                with gr.Column(elem_classes="prompt-card"):
                    use_standing = gr.Checkbox(value=True, label="Generate category 3 · Full-body front standing")
                    standing_prompt = gr.Textbox(
                        label="Full-body front standing prompt",
                        lines=5,
                        placeholder="The complete standing body will remain inside the frame.",
                    )
                with gr.Column(elem_classes="prompt-card"):
                    use_seated = gr.Checkbox(value=True, label="Generate category 4 · Full-body front seated")
                    seated_prompt = gr.Textbox(
                        label="Full-body front seated prompt",
                        lines=5,
                        placeholder="The complete seated body, hands, seat, legs, and feet will remain visible.",
                    )

            generate = gr.Button("Generate selected views", variant="primary", size="lg", elem_id="generate-button")
            status = gr.Markdown("Draft and review the prompts before generation.", elem_id="status-card")

            gr.Markdown("## Results\nOne image at a time, with simple navigation and a direct download.", elem_classes="section-heading")
            with gr.Column(elem_classes="output-panel"):
                carousel_image = gr.Image(
                    label="Generated portrait",
                    type="filepath",
                    height=680,
                    interactive=False,
                    buttons=["download", "fullscreen"],
                    placeholder="Your generated portrait will appear here.",
                    elem_id="carousel-image",
                )
                with gr.Row(elem_classes="carousel-controls"):
                    previous = gr.Button("← Previous", variant="secondary")
                    counter = gr.Markdown("No images yet", elem_classes="carousel-counter")
                    following = gr.Button("Next →", variant="secondary")
                    download_current = gr.DownloadButton(
                        "Download current image",
                        value=None,
                        variant="primary",
                        elem_classes="download-current",
                    )

            with gr.Accordion("Prompts sent in the last generation", open=False):
                enhanced_prompt = gr.Textbox(label="Selected view prompts", lines=12, interactive=False)

            with gr.Tabs(elem_classes="bottom-tabs"):
                with gr.Tab("History"):
                    history_status = gr.Markdown("Select a previous request to load it into the carousel.")
                    history_gallery = gr.Gallery(
                        label="Recent successful requests",
                        columns=4,
                        object_fit="cover",
                        height="auto",
                        buttons=["download", "fullscreen"],
                    )
                    with gr.Row():
                        refresh = gr.Button("Refresh history")
                        delete_selection = gr.Button("Delete selected", variant="secondary")
                    with gr.Row():
                        confirm_all = gr.Checkbox(label="I understand that deleting all history is permanent")
                        delete_everything = gr.Button("Delete all history", variant="stop")

                with gr.Tab("Diagnostics"):
                    gr.Markdown(
                        f"Logs are written to `{log_path}` and rotate at 5 MB. They include prompts and local output paths, but never the API key or image payloads.",
                        elem_classes="diagnostics-note",
                    )
                    refresh_log = gr.Button("Refresh logs")
                    log_viewer = gr.Textbox(
                        value=tail_log(log_path),
                        label="Latest backend activity",
                        lines=22,
                        max_lines=30,
                        interactive=False,
                        elem_id="log-viewer",
                    )

        draft_prompts.click(
            draft_view_prompts,
            inputs=[prompt],
            outputs=[
                front_prompt,
                left_prompt,
                right_prompt,
                standing_prompt,
                seated_prompt,
                prompt_status,
                log_viewer,
            ],
        )
        generate.click(
            generate_images,
            inputs=[
                prompt,
                model,
                aspect,
                quality,
                use_front,
                front_prompt,
                use_left,
                left_prompt,
                use_right,
                right_prompt,
                use_standing,
                standing_prompt,
                use_seated,
                seated_prompt,
            ],
            outputs=[
                carousel_image,
                carousel_paths,
                carousel_index,
                counter,
                download_current,
                enhanced_prompt,
                status,
                history_gallery,
                history_state,
                log_viewer,
            ],
        )
        previous.click(
            previous_image,
            inputs=[carousel_paths, carousel_index],
            outputs=[carousel_image, carousel_index, counter, download_current],
        )
        following.click(
            next_image,
            inputs=[carousel_paths, carousel_index],
            outputs=[carousel_image, carousel_index, counter, download_current],
        )
        refresh.click(refresh_history, outputs=[history_gallery, history_state])
        refresh_log.click(refresh_logs, outputs=[log_viewer])
        demo.load(refresh_history, outputs=[history_gallery, history_state])
        demo.load(refresh_logs, outputs=[log_viewer])
        history_gallery.select(
            select_history,
            inputs=[history_state],
            outputs=[
                selected_history,
                history_status,
                carousel_image,
                carousel_paths,
                carousel_index,
                counter,
                download_current,
            ],
        )
        delete_selection.click(
            delete_selected,
            inputs=[selected_history],
            outputs=[
                history_gallery,
                history_state,
                selected_history,
                history_status,
                carousel_image,
                carousel_paths,
                carousel_index,
                counter,
                download_current,
            ],
        )
        delete_everything.click(
            delete_all,
            inputs=[confirm_all],
            outputs=[
                history_gallery,
                history_state,
                confirm_all,
                selected_history,
                history_status,
                carousel_image,
                carousel_paths,
                carousel_index,
                counter,
                download_current,
            ],
        )

    return demo

