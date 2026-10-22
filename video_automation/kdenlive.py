"""Optional editor export, adapted from the workspace's original build_project.py."""
from __future__ import annotations

import copy
import json
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from .media import prepare_image
from .store import PipelineError, atomic_bytes
from .timing import assign_frames


def properties(element):
    return {p.get("name"): p.text for p in element.findall("property")}


def put(element, name, value):
    item = next((p for p in element.findall("property") if p.get("name") == name), None)
    if item is None:
        item = ET.SubElement(element, "property", name=name)
    item.text = str(value)


def export(store, template: Path):
    config, manifest = store.config, store.manifest
    total = assign_frames(manifest.cues, config.fps, manifest.audio_duration)
    tree = ET.parse(template)
    root = tree.getroot()
    root.set("root", store.folder.as_posix())
    profile = root.find("profile")
    if profile is None:
        raise PipelineError("Kdenlive template has no profile")
    for key, value in {"frame_rate_num": config.fps, "frame_rate_den": 1, "width": config.width,
                       "height": config.height, "description": f"{config.width}x{config.height} {config.fps} fps"}.items():
        profile.set(key, str(value))
    try:
        original = next(e for e in root.findall("producer") if properties(e).get("mlt_service") == "qimage")
        audio = next(e for e in root.findall("chain") if properties(e).get("mlt_service", "").startswith("avformat"))
        main_bin = next(e for e in root.findall("playlist") if e.get("id") == "main_bin")
        sequence = next(e for e in root.findall("tractor") if "kdenlive:uuid" in properties(e))
        project = next(e for e in root.findall("tractor") if properties(e).get("kdenlive:projectTractor") == "1")
        video_playlist = next(e for e in root.findall("playlist") if e.get("id") == "playlist4")
        audio_playlist = next(e for e in root.findall("playlist") if e.get("id") == "playlist2")
    except StopIteration as exc:
        raise PipelineError("Template must use the supplied workspace's two-audio/two-video track layout") from exc
    root.remove(original)
    root.remove(audio)
    insert_at = list(root).index(root.find("playlist"))
    root.insert(insert_at, audio)
    insert_at += 1
    audio.set("in", "0")
    audio.set("out", str(total-1))
    put(audio, "resource", store.path(manifest.audio_path).as_posix())
    put(audio, "length", total)
    put(audio, "kdenlive:clipname", "Narration")
    for playlist in root.findall("playlist"):
        for entry in list(playlist.findall("entry")) + list(playlist.findall("blank")):
            playlist.remove(entry)
    put(main_bin, "kdenlive:folder.-1.6", f"Illustrations — {len(manifest.cues)} timed scenes")
    put(main_bin, "kdenlive:folder.-1.7", "Narration")
    ET.SubElement(main_bin, "entry", producer=sequence.get("id"), attrib={"in": "0", "out": str(total-1)})
    ET.SubElement(main_bin, "entry", producer=audio.get("id"), attrib={"in": "0", "out": str(total-1)})
    ET.SubElement(audio_playlist, "entry", producer=audio.get("id"), attrib={"in": "0", "out": str(total-1)})
    for index, cue in enumerate(manifest.cues, 1):
        count = cue.end_frame-cue.start_frame
        prepared = store.path(f"render/prepared/{cue.id}.png")
        prepare_image(config, store.path(cue.image_path), prepared)
        producer = copy.deepcopy(original)
        producer_id = f"image_{index:04}"
        producer.set("id", producer_id)
        producer.set("in", "0")
        producer.set("out", str(count-1))
        for old_filter in producer.findall("filter"):
            producer.remove(old_filter)
        for name, value in {"length": count, "ttl": config.fps, "resource": prepared.as_posix(),
                            "meta.media.width": config.width, "meta.media.height": config.height,
                            "kdenlive:id": 9+index, "kdenlive:control_uuid": "{"+str(uuid.uuid4())+"}",
                            "kdenlive:folderid": 6, "kdenlive:clipname": cue.id}.items():
            put(producer, name, value)
        if cue.motion == "slow_push":
            effect = ET.SubElement(producer, "filter", attrib={"in": "0", "out": str(count-1)})
            for name, value in {"mlt_service": "affine", "kdenlive:id": "affine", "kdenlive:effectname": "Transform",
                                "kdenlive:ix": 1, "transition.fill": 1, "transition.distort": 0,
                                "transition.rect": f"0=0 0 {config.width} {config.height} 1;{count-1}={-config.width*.0075} {-config.height*.0075} {config.width*1.015} {config.height*1.015} 1"}.items():
                put(effect, name, value)
        root.insert(insert_at, producer)
        insert_at += 1
        ET.SubElement(main_bin, "entry", producer=producer_id, attrib={"in": "0", "out": str(count-1)})
        ET.SubElement(video_playlist, "entry", producer=producer_id, attrib={"in": "0", "out": str(count-1)})
    for element in [sequence, project] + [e for e in root.findall("producer") if properties(e).get("resource") == "black"]:
        element.set("out", str(total-1))
    if project.find("track") is not None:
        project.find("track").set("out", str(total-1))
    put(sequence, "kdenlive:clipname", manifest.topic)
    seconds, frame = divmod(total, config.fps)
    minutes, second = divmod(seconds, 60)
    hour, minute = divmod(minutes, 60)
    put(sequence, "kdenlive:duration", f"{hour:02}:{minute:02}:{second:02}:{frame:02}")
    put(sequence, "kdenlive:maxduration", total)
    put(sequence, "kdenlive:sequenceproperties.zonein", 0)
    put(sequence, "kdenlive:sequenceproperties.zoneout", total)
    put(sequence, "kdenlive:sequenceproperties.guides", json.dumps([{"pos": c.start_frame, "comment": c.id, "type": 0} for c in manifest.cues]))
    ET.indent(tree, space=" ")
    destination = store.path("project.kdenlive")
    atomic_bytes(destination, ET.tostring(root, encoding="utf-8", xml_declaration=True))
    return destination
