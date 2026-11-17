import pytest

from worker.video_probe import ProbeError, parse_probe


def test_parse_probe_ok() -> None:
    payload = {
        "streams": [
            {
                "codec_type": "video",
                "width": 720,
                "height": 576,
                "avg_frame_rate": "25/1",
                "sample_aspect_ratio": "64:45",
                "display_aspect_ratio": "16:9",
                "codec_name": "mpeg2video",
                "field_order": "tt",
            },
            {"codec_type": "audio"},
            {"codec_type": "subtitle"},
        ],
        "format": {"duration": "60.0"},
    }

    stream = parse_probe(payload)
    assert stream.width == 720
    assert stream.height == 576
    assert stream.fps == 25.0
    assert stream.audio_streams == 1
    assert stream.subtitle_streams == 1
    assert stream.interlaced is True


def test_parse_probe_missing_video() -> None:
    with pytest.raises(ProbeError):
        parse_probe({"streams": [{"codec_type": "audio"}]})
