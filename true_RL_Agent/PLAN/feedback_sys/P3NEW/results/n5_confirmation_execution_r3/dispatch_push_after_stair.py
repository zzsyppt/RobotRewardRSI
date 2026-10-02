"""One-shot continuation of this frozen capture; no retry or model calls."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CARD_SHA = '5c97844ff53027b80991a7308f991960e013df52fbc268046721e9771d42abc4'
DRIVER_SHA = 'f812429b033b32c538818a668d55fec6d72963c7c254fcf9c47ade1e23591255'
STAIR_PID = 3749979


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def stair_running():
    try:
        return Path(f'/proc/{STAIR_PID}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    assert hashlib.sha256((HERE / 'capture_card.json').read_bytes()).hexdigest() == CARD_SHA
    assert hashlib.sha256((HERE / 'capture_driver.py').read_bytes()).hexdigest() == DRIVER_SHA
    card = read(HERE / 'capture_card.json')
    capture = Path(card['output_root'])
    assert read(HERE / 'stair.driver.dispatch.json')['driver_pid'] == STAIR_PID
    assert not (HERE / 'push.driver.dispatch.json').exists()
    save(HERE / 'push.queue.started.json', {
        'started_at_utc': now(), 'queue_pid': os.getpid(), 'wait_for_stair_driver_pid': STAIR_PID,
        'card_sha256': CARD_SHA, 'driver_sha256': DRIVER_SHA, 'automatic_retries': 0,
        'scope': 'Wait for current Stair success, then dispatch existing frozen Push driver once; no API.'})
    exit_path = HERE / 'stair.exit.json'
    while not exit_path.exists() or stair_running():
        assert exit_path.exists() or stair_running(), 'Stair driver missing without exit receipt; stop'
        time.sleep(10)
    assert read(exit_path)['returncode'] == 0, 'Stair failed; do not dispatch Push'
    assert read(capture / 'stair/result.json')['status'] == 'PASS'
    assert read(capture / 'stair/public_dev_result.json')['evaluation_binding_hash'] == (
        card['tasks']['stair']['current_evaluation_binding']['evaluation_binding_hash'])
    assert not (HERE / 'push.started.json').exists() and not (capture / 'push').exists()
    # The unchanged driver rechecks all source/checkpoint bytes, current disk/GPU
    # resources and fresh output; the production runner owns the GPU lease.
    with (HERE / 'push.driver.stdout.log').open('xb') as stdout, (HERE / 'push.driver.stderr.log').open('xb') as stderr:
        child = subprocess.Popen([sys.executable, '-B', str(HERE / 'capture_driver.py'), 'push'],
                                 cwd=HERE, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                 start_new_session=True)
    save(HERE / 'push.driver.dispatch.json', {
        'dispatched_at_utc': now(), 'task': 'push', 'driver_pid': child.pid,
        'card_sha256': CARD_SHA, 'driver_sha256': DRIVER_SHA, 'automatic_retry': False,
        'api_calls': 0, 'dispatch_source': 'one_shot_after_successful_stair',
        'stair_exit_sha256': hashlib.sha256(exit_path.read_bytes()).hexdigest()})
    return {'status': 'PUSH_DISPATCHED_ONCE', 'push_driver_pid': child.pid}


if __name__ == '__main__':
    result = {'status': 'FAILED_NO_RETRY'}
    try:
        result = main()
    finally:
        save(HERE / 'push.queue.finished.json', {**result, 'finished_at_utc': now()})
