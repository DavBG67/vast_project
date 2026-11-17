from pathlib import Path

from vast_upscaler.video import parse_ffprobe_data


def test_parse_ffprobe_data() -> None:
    sample = {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "mpeg2video",
                "width": 720,
                "height": 576,
                "avg_frame_rate": "25/1",
                "sample_aspect_ratio": "64:45",
                "display_aspect_ratio": "16:9",
                "field_order": "tt",
                "pix_fmt": "yuv420p",
            },
            {"codec_type": "audio", "codec_name": "ac3"},
            {"codec_type": "audio", "codec_name": "ac3"},
            {"codec_type": "subtitle", "codec_name": "dvd_subtitle"},
        ],
        "format": {
            "size": "123456",
            "duration": "12.5",
            "format_name": "matroska,webm",
        },
    }

    info = parse_ffprobe_data(sample, Path("movie.mkv"))

    assert info.video_codec == "mpeg2video"
    assert info.width == 720
    assert info.height == 576
    assert info.frame_rate_fps == 25.0
    assert info.sample_aspect_ratio == "64:45"
    assert info.display_aspect_ratio == "16:9"
    assert info.interlaced is True
    assert info.audio_stream_count == 2
    assert info.subtitle_stream_count == 1
    assert info.audio_codec == "ac3"
