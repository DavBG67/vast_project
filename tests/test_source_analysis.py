from pathlib import Path

import pytest

from vast_upscaler.source_analysis import SourceAnalysisError, parse_source_probe


def test_parse_source_probe_includes_expected_metadata() -> None:
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "field_order": "progressive",
                "pix_fmt": "yuv420p",
                "sample_aspect_ratio": "1:1",
                "display_aspect_ratio": "16:9",
                "bit_rate": "4500000",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "channels": 2,
                "bit_rate": "128000",
                "tags": {"language": "eng"},
            },
            {
                "index": 2,
                "codec_type": "subtitle",
                "codec_name": "mov_text",
                "tags": {"language": "fre"},
            },
        ],
        "format": {
            "size": "1234567",
            "duration": "12.4",
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "bit_rate": "5000000",
        },
    }

    info = parse_source_probe(payload=payload, video_path=Path("input.mp4"))

    assert info.width == 1920
    assert info.height == 1080
    assert info.video_codec == "h264"
    assert info.frame_rate_fps == pytest.approx(29.970, abs=0.001)
    assert info.duration_seconds == 12.4
    assert info.overall_bitrate_bps == 5000000
    assert info.video_bitrate_bps == 4500000
    assert info.audio_stream_count == 1
    assert info.subtitle_stream_count == 1
    assert info.audio_tracks[0].codec == "aac"
    assert info.audio_tracks[0].language == "eng"
    assert info.subtitle_tracks[0].codec == "mov_text"
    assert info.subtitle_tracks[0].language == "fre"
    assert info.interlace.interlaced is False
    assert info.interlace.confidence == "certain"


def test_parse_source_probe_marks_interlace_unknown_as_uncertain() -> None:
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "mpeg2video",
                "width": 720,
                "height": 576,
                "avg_frame_rate": "25/1",
                "field_order": "unknown",
            }
        ],
        "format": {},
    }

    info = parse_source_probe(payload=payload, video_path=Path("input.mkv"))

    assert info.interlace.interlaced is None
    assert info.interlace.confidence == "uncertain"


def test_parse_source_probe_fails_without_video_stream() -> None:
    payload = {"streams": [{"codec_type": "audio"}], "format": {}}
    with pytest.raises(SourceAnalysisError):
        parse_source_probe(payload=payload, video_path=Path("input.mkv"))
