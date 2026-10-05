import ast
from pathlib import Path
from types import SimpleNamespace


def probe_namespace(run):
    source = ast.parse((Path(__file__).parents[1] / 'handler.py').read_text())
    functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in {'nvenc_available','codec_info'}]
    ns = {'subprocess': SimpleNamespace(run=run), 'os': SimpleNamespace(environ={'NVIDIA_DRIVER_CAPABILITIES':'compute,utility,video'}), '_NVENC':None, '_NVENC_ERROR':None}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'handler.py','exec'),ns)
    return ns


def test_reports_driver_error_and_caches_real_probe():
    calls=[]
    def run(cmd,**kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=1,stderr=b'Cannot load libnvidia-encode.so.1')
    ns=probe_namespace(run)
    result=ns['codec_info']()
    assert result['nvenc_available'] is False
    assert 'libnvidia-encode' in result['nvenc_error']
    ns['codec_info']()
    assert len(calls)==1
    assert '-i' in calls[0] and 'h264_nvenc' in calls[0]


def test_success_and_probe_exception():
    ns=probe_namespace(lambda *a,**kw:SimpleNamespace(returncode=0,stderr=b''))
    assert ns['codec_info']()['nvenc_available'] is True
    assert ns['codec_info']()['nvenc_error'] is None
    def fail(*a,**kw):raise TimeoutError('probe timeout')
    ns=probe_namespace(fail)
    assert ns['codec_info']()['nvenc_available'] is False
    assert 'TimeoutError' in ns['codec_info']()['nvenc_error']
