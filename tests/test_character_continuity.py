from copy import deepcopy

import pytest

from test_pipeline import FakeProviders, character_profile, config, image_bytes, store, visual_words
from video_automation.pipeline import ImagesPending, Pipeline
from video_automation.providers import STYLE, validate_visual_context
from video_automation.store import Store, TemporaryError, file_hash, read_json


def daniel():
    return {"id": "daniel", "age_bracket": "8-year-old boy",
            "head_shape": "simple round white head",
            "hair_style": "three spiky hair strands pointing right", "hair_color": "jet-black",
            "clothing": [{"style": "short-sleeve boxy shirt", "color": "royal-blue"},
                         {"style": "knee-length shorts", "color": "dark-navy"}],
            "signature_anchor": {"item": "flashlight", "color": "silver", "placement": "held in left mitten hand"},
            "proportions": "child-sized torso and short stick limbs",
            "relative_height": "half adult height, head below a standard tabletop"}


def test_complete_identity_profile_is_accepted():
    validate_visual_context({"summary": "Daniel explores his surroundings.", "characters": [daniel()]})


def test_uniform_drawing_rules_are_in_every_style_block():
    assert "uniform bold black outlines" in STYLE
    assert "two black dot eyes" in STYLE
    assert "no shading" in STYLE.lower()


def test_copy_identity_to_every_relevant_prompt_across_batches(store):
    store.manifest.image_mode = "external"
    provider = FakeProviders()
    shared = {"summary": "Daniel explores, pauses for a diagram, then returns.", "characters": [daniel()]}
    provider.visual_context = lambda narration: deepcopy(shared)
    original = provider.visuals

    def visuals(cues, previous_scenes=None, context=None):
        planned = original(cues, previous_scenes, context)
        for record in planned["cues"]:
            # The renderer must carry the identity even when the model omits appearance details.
            record["scene_description"] = 'A child explores beside a table. Draw the exact label "38%".'
            record["character_ids"] = [] if record["id"] == "cue_0025" else ["daniel"]
        return planned

    provider.visuals = visuals
    words = visual_words(store)
    pipeline = Pipeline(store, provider)
    pipeline.visuals(words)
    with pytest.raises(ImagesPending):
        pipeline.images()
    prompts = read_json(store.path("image_prompts.json"))
    assert len(prompts) > 48
    expected_identity = (
        "daniel: 8-year-old boy; head: simple round white head; "
        "hair: three spiky hair strands pointing right; hair color: jet-black; "
        "clothing: royal-blue short-sleeve boxy shirt, dark-navy knee-length shorts; "
        "signature anchor: silver flashlight, held in left mitten hand; "
        "proportions: child-sized torso and short stick limbs; "
        "relative height: half adult height, head below a standard tabletop."
    )
    for record in prompts:
        assert record["prompt"].endswith("\n" + STYLE)
        assert '"38%"' in record["prompt"]
        assert record["prompt"].count(expected_identity) == (0 if record["cue_id"] == "cue_0025" else 1)
    assert "image" not in provider.calls
    before = {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns)
              for name in ("visual_context.json", "visuals/batch_000.json", "image_prompts.json")}
    calls = list(provider.calls)
    resumed = Pipeline(Store(store.folder, store.config), provider)
    resumed.visuals(words)
    with pytest.raises(ImagesPending):
        resumed.images()
    assert before == {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns) for name in before}
    assert provider.calls == calls


@pytest.mark.parametrize("field", ["age_bracket", "head_shape", "hair_style", "hair_color", "clothing",
                                  "signature_anchor", "proportions", "relative_height"])
def test_identity_profile_requires_every_anchor(field):
    profile = daniel()
    del profile[field]
    with pytest.raises(TemporaryError, match="Invalid visual context"):
        validate_visual_context({"summary": "Daniel explores.", "characters": [profile]})


@pytest.mark.parametrize("field", ["age_bracket", "head_shape", "hair_style", "hair_color",
                                  "proportions", "relative_height"])
def test_identity_profile_rejects_blank_anchor_values(field):
    profile = daniel()
    profile[field] = " \t\n"
    with pytest.raises(TemporaryError, match="blank fields"):
        validate_visual_context({"summary": "Daniel explores.", "characters": [profile]})


@pytest.mark.parametrize("field", ["item", "color", "placement"])
@pytest.mark.parametrize("invalid", [None, " \t"])
def test_signature_accessory_requires_item_color_and_placement(field, invalid):
    profile = daniel()
    if invalid is None:
        del profile["signature_anchor"][field]
    else:
        profile["signature_anchor"][field] = invalid
    with pytest.raises(TemporaryError, match="Invalid visual context"):
        validate_visual_context({"summary": "Daniel explores.", "characters": [profile]})


@pytest.mark.parametrize("field", ["style", "color"])
def test_garments_need_both_exact_style_and_color(field):
    profile = daniel()
    profile["clothing"][0][field] = " "
    with pytest.raises(TemporaryError, match="blank fields"):
        validate_visual_context({"summary": "Daniel explores.", "characters": [profile]})


def test_signature_accessory_must_distinguish_characters():
    child, adult = daniel(), character_profile()
    adult["signature_anchor"] = {"item": "  FLASHLIGHT  ", "color": " SILVER ", "placement": "held in right hand"}
    with pytest.raises(TemporaryError, match="signature item/color combinations must be distinct"):
        validate_visual_context({"summary": "A child and an adult.", "characters": [child, adult]})


def test_bald_unclothed_subject_and_empty_cast_are_explicit():
    from video_automation.providers import scene_prompt
    profile = daniel()
    profile.update(hair_style="bald with no hair", hair_color="none", clothing=[])
    shared = {"summary": "An illustrative stick figure.", "characters": [profile]}
    validate_visual_context(shared)
    prompt = scene_prompt("daniel stands beside a table.", ["daniel"], shared)
    assert "hair: bald with no hair; hair color: none; clothing: none;" in prompt
    validate_visual_context({"summary": "A diagram without people.", "characters": []})
    assert scene_prompt("A diagram.", [], {"characters": []}) == "A diagram."


def test_old_description_only_context_refreshes_without_changing_existing_assets(store):
    import video_automation.pipeline as orchestration
    from video_automation.store import atomic_bytes
    store.manifest.image_mode = "external"
    words = visual_words(store, 8)
    narration = "A parent watches a child."
    old_context = {"summary": "A family.", "characters": [{"id": "parent", "description": "A parent in blue."}]}
    provider = FakeProviders()
    pipeline = Pipeline(store, provider)
    old_schema = {"type": "object", "properties": {"summary": {"type": "string"},
                  "characters": {"type": "array", "items": {"type": "object", "properties": {
                      "id": {"type": "string"}, "description": {"type": "string"}}}}}}
    inputs = {"narration": narration, "text_generation": store.config.text_cache_settings,
              "style": STYLE, "prompt": orchestration.VISUAL_CONTEXT_PROMPT, "schema": old_schema}
    pipeline.json_task("visual.context", inputs, "visual_context.json", lambda: old_context)
    atomic_bytes(store.path("audio/narration.wav"), b"completed speech")
    atomic_bytes(store.path("images/cue_0001.png"), image_bytes())
    atomic_bytes(store.path("responses/old_visual_context.json"), b"preserved raw response")
    preserved = {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns) for name in (
        "audio/narration.wav", "images/cue_0001.png", "responses/old_visual_context.json")}
    old_snapshot = store.path("visual_context.json").read_bytes()
    resumed = Pipeline(Store(store.folder, store.config), provider)
    assert store.path("visual_context.json").read_bytes() == old_snapshot
    resumed.visuals(words, narration)
    with pytest.raises(ImagesPending):
        resumed.images()
    assert read_json(store.path("visual_context.json"))["characters"] == [character_profile()]
    assert preserved == {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns) for name in preserved}
    assert provider.calls == ["visual_context", "visuals"]


def test_api_images_receive_exact_saved_identity_prompts(store):
    provider = FakeProviders()
    captured = []
    original = provider.image

    def image(prompt):
        captured.append(prompt)
        return original(prompt)

    provider.image = image
    pipeline = Pipeline(store, provider)
    pipeline.visuals(visual_words(store, 8))
    pipeline.images()
    prompts = read_json(store.path("image_prompts.json"))
    assert captured == [record["prompt"] for record in prompts]
    assert all("head: simple round white head" in prompt and "royal-blue short-sleeve boxy tunic" in prompt
               and "signature anchor: black round wire-rim glasses, worn on face" in prompt for prompt in captured)


def test_multiple_characters_keep_separate_identities_without_leaking_absent_cast():
    from video_automation.providers import scene_prompt
    shared = {"summary": "A child explores with a parent.", "characters": [daniel(), character_profile()]}
    before = deepcopy(shared)
    scene = "parent hands a map to daniel."
    together = scene_prompt(scene, ["parent", "daniel"], shared)
    assert together.startswith("Fixed character identities")
    assert together.index("parent:") < together.index("daniel:") < together.index(scene)
    assert "signature anchor: silver flashlight, held in left mitten hand" in together
    assert "signature anchor: black round wire-rim glasses, worn on face" in together
    alone = scene_prompt("daniel studies a map.", ["daniel"], shared)
    assert "parent:" not in alone and "wire-rim glasses" not in alone
    assert "do not draw them as text" in alone
    assert shared == before
