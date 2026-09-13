import importlib.util
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "plugins" / "bilibili-understand" / "skills" / "bilibili-understand"
SCRIPTS = SKILL / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


probe = load_module("probe_bilibili", "probe_bilibili.py")
transcribe = load_module("transcribe_audio", "transcribe_audio.py")
pipeline = load_module("pipeline", "pipeline.py")


class ProbeTests(unittest.TestCase):
    def test_normalize_url_keeps_page_and_drops_tracking(self):
        actual = probe.normalize_url(
            "HTTPS://www.bilibili.com/video/BV1abc/?p=2&vd_source=private#reply"
        )
        self.assertEqual(actual, "https://www.bilibili.com/video/BV1abc/?p=2")

    def test_rejects_lookalike_host_and_embedded_credentials(self):
        for url in (
            "https://bilibili.com.example.org/video/BV1abc/",
            "https://user:secret@bilibili.com/video/BV1abc/",
        ):
            with self.subTest(url=url), self.assertRaises(probe.InputError):
                probe.normalize_url(url)

    def test_metadata_drops_temporary_urls_and_tracking(self):
        requested = "https://www.bilibili.com/video/BV1abc/"
        result = probe.sanitize_metadata(
            {
                "id": "BV1abc",
                "webpage_url": requested + "?vd_source=private",
                "url": "https://temporary.example/signed-media",
                "subtitles": {
                    "zh-CN": [
                        {
                            "ext": "json3",
                            "name": "Chinese",
                            "protocol": "https",
                            "url": "https://temporary.example/signed-subtitle",
                        }
                    ]
                },
            },
            requested,
        )
        self.assertEqual(result["webpage_url"], requested)
        self.assertNotIn("url", result)
        self.assertNotIn("url", result["native_subtitles"][0]["formats"][0])

    def test_tool_report_has_full_runtime_contract(self):
        report = probe.tool_report()
        self.assertEqual(report["required"], ["yt-dlp", "faster-whisper"])
        self.assertEqual(report["optional"], ["ffmpeg", "ffprobe"])
        self.assertEqual(report["status"], "ok" if not report["missing"] else "tool_missing")


class PackageTests(unittest.TestCase):
    def test_timestamp_format(self):
        self.assertEqual(transcribe.timestamp(3661.234), "01:01:01.234")

    def test_plugin_manifests_and_skill_metadata_exist(self):
        plugin = ROOT / "plugins" / "bilibili-understand"
        for manifest in (plugin / "plugin.json", plugin / ".codex-plugin" / "plugin.json"):
            data = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(data["name"], "bilibili-understand")
            self.assertEqual(data["version"], "0.2.0")

        marketplace = json.loads(
            (ROOT / ".agents" / "plugins" / "marketplace.json").read_text(encoding="utf-8")
        )
        entry = marketplace["plugins"][0]
        self.assertEqual(marketplace["name"], "bilibili-tools")
        self.assertTrue((ROOT / entry["source"]["path"]).is_dir())

        skill_text = (SKILL / "SKILL.md").read_text(encoding="utf-8")
        self.assertTrue(skill_text.startswith("---\nname: bilibili-understand\n"))
        openai_yaml = (SKILL / "agents" / "openai.yaml").read_text(encoding="utf-8")
        self.assertIn("$bilibili-understand", openai_yaml)


class PipelineTests(unittest.TestCase):
    def test_parse_time_and_interval(self):
        self.assertEqual(pipeline.parse_time("14:40"), 880)
        self.assertEqual(pipeline.parse_time("01:02:03.5"), 3723.5)
        self.assertEqual(pipeline.resolve_interval("14:40", None, "60"), (880, 940))
        with self.assertRaises(ValueError):
            pipeline.resolve_interval("14:40", "14:39", None)

    def test_selects_only_overlapping_segments(self):
        records = [
            {"start_s": 0, "end_s": 10, "text": "before"},
            {"start_s": 9, "end_s": 12, "text": "overlap"},
            {"start_s": 12, "end_s": 15, "text": "inside"},
            {"start_s": 20, "end_s": 21, "text": "after"},
        ]
        selected = pipeline.select_segments(records, 10, 20)
        self.assertEqual([record["text"] for record in selected], ["overlap", "inside"])

    def test_compact_transcript_drops_word_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "transcript.jsonl"
            destination = Path(temporary) / "segments.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "start_s": 1,
                        "end_s": 2,
                        "text": "hello",
                        "words": [{"text": "hello", "probability": 0.9}],
                        "source": "asr:test",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(pipeline.compact_transcript(source, destination), 1)
            compact = json.loads(destination.read_text(encoding="utf-8"))
            self.assertNotIn("words", compact)
            self.assertNotIn("source", compact)
            self.assertEqual(compact["text"], "hello")

    def test_asr_cache_matches_semantic_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transcript = root / "transcript.jsonl"
            metadata = root / "asr_metadata.json"
            transcript.write_text("{}\n", encoding="utf-8")
            metadata.write_text(
                json.dumps(
                    {
                        "model": "large-v3-turbo",
                        "language": "zh",
                        "device": "cuda",
                        "compute_type": "float16",
                        "vad_filter": False,
                        "hotwords": "实习 技术岗",
                    }
                ),
                encoding="utf-8",
            )
            requested = {
                "model": "large-v3-turbo",
                "language": "zh",
                "vad_filter": False,
                "hotwords": "实习 技术岗",
            }
            self.assertTrue(pipeline.asr_cache_matches(transcript, metadata, requested))
            requested["hotwords"] = "different"
            self.assertFalse(pipeline.asr_cache_matches(transcript, metadata, requested))

    def test_unspecified_asr_options_reuse_existing_configuration(self):
        args = Namespace(model=None, language=None, vad_filter=None, hotwords=None)
        existing = {
            "model": "large-v3-turbo",
            "language": "zh",
            "vad_filter": False,
            "hotwords": "实习 技术岗",
        }
        self.assertEqual(pipeline.resolve_asr_config(args, existing), existing)
        args.model = "small"
        self.assertEqual(pipeline.resolve_asr_config(args, existing)["model"], "small")

    def test_only_network_failures_are_retried(self):
        self.assertTrue(pipeline.should_retry("network_error"))
        self.assertFalse(pipeline.should_retry("anti_bot"))
        self.assertFalse(pipeline.should_retry("auth_required"))
        self.assertFalse(pipeline.should_retry("video_unavailable"))

    def test_status_reports_missing_cache_without_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with redirect_stdout(output):
                code = pipeline.status(
                    Namespace(target="BV1missing", output_root=Path(temporary))
                )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "missing")


if __name__ == "__main__":
    unittest.main()
