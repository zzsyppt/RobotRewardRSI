"""Extract one unmodified native frame and prepare paired media transport probes."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    origin = ROOT / 'PLAN/2new_world/P6/go2_pose_01/native/video'
    manifest_path = origin / 'recordings.json'
    manifest = json.loads(manifest_path.read_text())
    video = origin / 'public_dev-01.mp4'
    media = OUT / 'media'
    media.mkdir(exist_ok=True)
    frame = media / 'go2_native_frame_150.png'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(video),
               '-vf', 'select=eq(n\\,150)', '-frames:v', '1', '-y', str(frame)]
    subprocess.run(command, check=True, capture_output=True)
    prompt = ('Describe only directly visible content of the supplied image. If no image is supplied, say unavailable. '
              'Return short JSON with robot_form, visible_leg_count, environment, posture, '
              'whether exact toe contact or contact force can be determined. Do not infer measurements.')
    for name, with_image in [('media_text_control', False), ('media_native', True)]:
        blocks = [{'type': 'text', 'text': prompt}]
        if with_image:
            blocks.append({'type': 'image_url', 'image_url': {
                'url': 'data:image/png;base64,' + base64.b64encode(frame.read_bytes()).decode(), 'detail': 'original'}})
        request = {'model': 'deepseek-v4-flash', 'messages': [{'role': 'user', 'content': blocks}],
                   'response_format': {'type': 'json_object'}, 'max_tokens': 192,
                   'thinking': {'type': 'disabled'}, 'temperature': 0, 'stream': False}
        (OUT / 'provider_requests' / (name + '.json')).write_text(json.dumps(request, ensure_ascii=False) + '\n', encoding='utf-8')
    receipt = {'source_video': video.relative_to(ROOT).as_posix(), 'source_video_sha256': sha(video),
               'source_recordings': manifest_path.relative_to(ROOT).as_posix(), 'source_recordings_sha256': sha(manifest_path),
               'frame_path': frame.relative_to(OUT).as_posix(), 'frame_sha256': sha(frame), 'frame_index_zero_based': 150,
               'transform': 'decode one original frame; no resize, crop, overlay or layout',
               'episode_identity_source': 'recordings.json recording public_dev-01; this is a diagnostic replay run',
               'recordings': manifest['recordings'], 'extraction_command': command}
    (OUT / 'media_baseline.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'video_bytes': video.stat().st_size, 'frame_bytes': frame.stat().st_size, 'frame_sha256': sha(frame)}))


if __name__ == '__main__':
    main()
