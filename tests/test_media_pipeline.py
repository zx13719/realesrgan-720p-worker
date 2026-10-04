"""Exercise the real ffmpeg pipe/cleanup path with a CPU stand-in for inference."""
import ast
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest


@pytest.fixture
def pipeline():
    # Import only codec functions so CPU regression tests need no CUDA wheel/models.
    source = ast.parse((Path(__file__).parents[1] / 'handler.py').read_text())
    funcs = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in {'ffprobe', 'count_frames', 'upscale_video'}]
    ns = {'subprocess': subprocess, 'json': __import__('json'), 'os': __import__('os'),
          'threading': threading, 'queue': queue, 'np': np, 'time': time}
    def enhance(frames, *a, **kw):
        return [np.repeat(np.repeat(frame, 2, axis=0), 2, axis=1) for frame in frames]
    ns['enhance_batch'] = enhance
    exec(compile(ast.Module(body=funcs, type_ignores=[]), 'handler.py', 'exec'), ns)
    return ns


@pytest.fixture
def source(tmp_path):
    assert shutil.which('ffmpeg'), 'ffmpeg required for codec regression tests'
    src = tmp_path / 'in.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                    'testsrc=size=24x32:rate=10:duration=1', '-f', 'lavfi', '-i',
                    'sine=frequency=440:duration=1', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', str(src)], check=True)
    return src


def test_preserves_all_frames_and_audio(pipeline, source, tmp_path):
    dst = tmp_path / 'out.mp4'
    meta = pipeline['upscale_video'](str(source), str(dst), None, 2, 2, vcodec='libx264', batch=4)
    assert meta['frames'] == meta['verified_frames'] == 10
    assert pipeline['ffprobe'](str(dst))[:2] == (48, 64)
    assert pipeline['ffprobe'](str(dst))[-1] is True


def test_gpu_failure_cleans_up_child_processes(pipeline, source, tmp_path, monkeypatch):
    processes = []
    real_popen = subprocess.Popen
    def popen(*a, **kw):
        child = real_popen(*a, **kw); processes.append(child); return child
    monkeypatch.setattr(subprocess, 'Popen', popen)
    def fail(*a, **kw):
        raise RuntimeError('GPU out of memory')
    pipeline['enhance_batch'] = fail
    with pytest.raises(RuntimeError, match='GPU out of memory'):
        pipeline['upscale_video'](str(source), str(tmp_path / 'bad.mp4'), None, 2, 2, vcodec='libx264', batch=1)
    assert all(child.poll() is not None for child in processes)


def test_encoder_failure_never_passes_qc(pipeline, source, tmp_path, monkeypatch):
    processes = []
    real_popen = subprocess.Popen
    def popen(cmd, *a, **kw):
        if 'pipe:0' in cmd:
            cmd = [x if x != 'libx264' else 'nonexistent_encoder' for x in cmd]
        child = real_popen(cmd, *a, **kw); processes.append(child); return child
    monkeypatch.setattr(subprocess, 'Popen', popen)
    with pytest.raises((RuntimeError, BrokenPipeError)):
        pipeline['upscale_video'](str(source), str(tmp_path / 'bad.mp4'), None, 2, 2, vcodec='libx264', batch=1)
    assert all(child.poll() is not None for child in processes)
