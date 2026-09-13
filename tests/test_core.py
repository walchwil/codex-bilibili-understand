import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "plugins" / "bilibili-understand" / "skills" / "bilibili-understand"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SKILL / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


probe = load_module("probe_bilibili", "probe_bilibili.py")
transcribe = load_module("transcribe_audio", "transcribe_audio.py")


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
            self.assertEqual(data["version"], "0.1.0")

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


if __name__ == "__main__":
    unittest.main()
