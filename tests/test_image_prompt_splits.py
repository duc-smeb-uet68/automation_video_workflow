import pytest

from test_pipeline import FakeProviders, config, store
from video_automation.models import Cue
from video_automation.pipeline import ImagesPending, Pipeline
from video_automation.store import Store, file_hash, read_json, write_json


PARTS = [f"image_prompts_{index}.json" for index in range(1, 4)]


def set_cues(store, count):
    store.manifest.cues = [
        Cue(id=f"cue_{index:04}", text="A scene.", start=index-1, end=index,
            word_start=index-1, word_end=index, scene_description=f"Scientist examines sample {index}.")
        for index in range(1, count + 1)
    ]


def assert_parts(store, sizes):
    parts = [read_json(store.path(name)) for name in PARTS]
    assert [len(part) for part in parts] == sizes
    assert [record for part in parts for record in part] == read_json(store.path("image_prompts.json"))


@pytest.mark.parametrize("mode", ["external", "api"])
@pytest.mark.parametrize("count,sizes", [(1, [1, 0, 0]), (2, [1, 1, 0]), (3, [1, 1, 1]),
                                        (4, [2, 1, 1]), (5, [2, 2, 1]), (6, [2, 2, 2]),
                                        (25, [9, 8, 8])])
def test_three_balanced_prompt_files_exist_before_image_handoff(store, mode, count, sizes):
    store.manifest.image_mode = mode
    set_cues(store, count)
    provider = FakeProviders()
    original_image = provider.image

    def image(prompt):
        assert_parts(store, sizes)
        return original_image(prompt)

    provider.image = image
    pipeline = Pipeline(store, provider)
    if mode == "external":
        with pytest.raises(ImagesPending):
            pipeline.images()
        assert provider.calls == []
    else:
        pipeline.images()
        assert provider.calls == ["image"] * count
    assert_parts(store, sizes)
    assert set(store.manifest.stages["image.prompts.split"].artifacts) == set(PARTS)


def test_resume_reuses_splits_repairs_files_and_backfills_older_jobs(store):
    set_cues(store, 4)
    provider = FakeProviders()
    Pipeline(store, provider).images()
    preserved = ["image_prompts.json", *[cue.image_path for cue in store.manifest.cues]]
    before = {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns)
              for name in preserved + PARTS}
    resumed = Pipeline(Store(store.folder, store.config), provider)
    resumed.images()
    assert before == {name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns)
                      for name in before}

    store.path(PARTS[0]).unlink()
    write_json(store.path(PARTS[1]), [{"cue_id": "wrong", "prompt": "Wrong prompt."}])
    resumed.images()
    assert_parts(resumed.store, [2, 1, 1])

    del resumed.store.manifest.stages["image.prompts.split"]
    resumed.store.save()
    for name in PARTS:
        store.path(name).unlink()
    older = Pipeline(Store(store.folder, store.config), provider)
    older.images()
    assert_parts(older.store, [2, 1, 1])
    assert provider.calls == ["image"] * 4
    assert {name: before[name] for name in preserved} == {
        name: (file_hash(store.path(name)), store.path(name).stat().st_mtime_ns) for name in preserved}


def test_prompt_changes_refresh_splits_and_only_affected_image(store):
    set_cues(store, 4)
    provider = FakeProviders()
    pipeline = Pipeline(store, provider)
    pipeline.images()
    store.manifest.cues[-1].scene_description = "Scientist holds a red sample."
    pipeline.images()
    assert_parts(store, [2, 1, 1])
    assert read_json(store.path(PARTS[-1]))[-1]["prompt"].startswith("Scientist holds a red sample.")
    assert provider.calls == ["image"] * 5
