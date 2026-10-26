import os
import stat
import pwd
import grp
import sys
import contextvars
from pathlib import Path

from loguru import logger

# 并行(asyncio.gather)运行多个 task 时, 给每条日志打上 task_id.
#
# 全项目已统一用裸包名(``tools_v2.general_tools``)导入, 正常情况下本模块只加载一份.
# 但为防止将来再次出现"同一模块被两个名字各导入一次 → 两份模块对象 → 模块级
# ContextVar 写/读落在不同副本上 → task_id 永远是 '-'"的坑, 这里把 ContextVar
# 挂到 ``loguru.logger`` 这个真正的单例对象上: 不论本模块被导入几次, ``logger``
# 始终是同一个实例, 各副本共享同一个 ContextVar.
# ContextVar 本身在每个 asyncio task 中天然隔离, 互不串扰.
if not hasattr(logger, "_zz_task_id_ctx"):
    logger._zz_task_id_ctx = contextvars.ContextVar("zz_log_task_id", default="-")
_current_task_id: contextvars.ContextVar[str] = logger._zz_task_id_ctx


def set_log_task_id(task_id: str) -> None:
    """设置当前 asyncio task 的日志 task_id (供 loguru patcher 注入到每条日志)."""
    _current_task_id.set(task_id)


from pydantic_ai.models import create_async_http_client
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from tools_v2.model_http import limit_http_client

# ── 本地默认模型配置 (docker 环境由外部环境变量注入, 不使用这些默认值) ──
DEFAULT_MODEL_API_URLS = os.getenv("DEFAULT_MODEL_API_URLS") or ""
DEFAULT_MODEL_API_URL = os.getenv("DEFAULT_MODEL_API_URL") or "http://10.166.163.101:8417/v1"
DEFAULT_MODEL_API_KEY = os.getenv("DEFAULT_MODEL_API_KEY") or "empty"
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME") or "qwen3.5-35b-a3b"

# ── HTTP 超时 (秒) ──
# pydantic-ai 默认 read timeout 是 600s 且无应用层重试: 一旦连接被静默掐断
# (如 Mac 睡眠/锁屏导致 docker VM 掉网), httpx 要等满 600s 才抛异常, 表现为
# "请求僵死很久". 这里把 read timeout 收紧到 300s, 断连后能更快落到
# zz_agent_v2 的 attempt 重试. 可用环境变量 MODEL_HTTP_TIMEOUT 覆盖.
MODEL_HTTP_TIMEOUT = int(os.environ.get("MODEL_HTTP_TIMEOUT", "300"))
MODEL_HTTP_CONNECT_TIMEOUT = int(os.environ.get("MODEL_HTTP_CONNECT_TIMEOUT", "5"))


def _build_http_client():
    """为单个 model 构造一个独立的 httpx.AsyncClient (短超时).

    每次调用返回新实例, 不共享连接池: 避免某个 model 的一条坏连接 (睡眠掐断/
    服务端不返回) 通过共享池拖垮其他 model 的请求. read timeout 默认 300s,
    connect 5s, 均可经环境变量覆盖.
    """
    return limit_http_client(create_async_http_client(
        timeout=MODEL_HTTP_TIMEOUT,
        connect=MODEL_HTTP_CONNECT_TIMEOUT,
    ))


def is_docker() -> bool:
    if os.environ.get("RUNTIME_ENV") == "docker":
        return True
    return os.path.exists('/.dockerenv')


def _macos_clonefile(src, dst) -> bool:
    """macOS APFS: 用系统调用 clonefile 做写时复制(copy-on-write).

    数据块共享, 任何一方写入时自动分裂复制 -> 既不占额外磁盘, 又物理隔离零污染.
    成功返回 True; 不支持/失败返回 False (由调用方回退到普通 copy).
    """
    import ctypes
    import ctypes.util

    try:
        lib = ctypes.CDLL(
            ctypes.util.find_library('System') or 'libSystem.dylib',
            use_errno=True,
        )
        fn = lib.clonefile
        fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        fn.restype = ctypes.c_int
        rc = fn(os.fspath(src).encode(), os.fspath(dst).encode(), 0)
        return rc == 0
    except Exception:
        return False


def clone_or_copy(src, dst):
    """copytree 的 copy_function: 本地 macOS 优先 clonefile, 否则普通 copy2.

    - 非 docker 的 macOS(APFS): clonefile 写时复制, 省存储且安全;
    - docker / clonefile 失败: 回退 shutil.copy2, 行为与原来一致.
    """
    import shutil

    if not is_docker() and sys.platform == 'darwin':
        if _macos_clonefile(src, dst):
            return
    shutil.copy2(src, dst)


def _setup_logger():
    """根据运行环境配置 loguru：本地输出 DEBUG+，docker 静默。"""
    if getattr(logger, "_zz_loguru_configured", False):
        return

    logger.remove()

    # 全局 patcher: 把当前 ContextVar 里的 task_id 注入到每条日志的 extra,
    # 这样所有子模块的日志(无需改动)都能在格式里展示 task_id.
    def _inject_task_id(record):
        record["extra"]["task_id"] = _current_task_id.get()

    logger.configure(patcher=_inject_task_id)

    if not is_docker():
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <magenta>{extra[task_id]}</magenta> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )
    # docker 环境不添加任何 sink → 所有日志静默
    logger._zz_loguru_configured = True


_setup_logger()


def add_loguru_file_sink(log_path: os.PathLike | str) -> None:
    """给当前进程追加一个 loguru 文件 sink；重复调用会替换旧文件 sink."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    old_sink_id = getattr(logger, "_zz_loguru_file_sink_id", None)
    old_sink_path = getattr(logger, "_zz_loguru_file_sink_path", None)
    if old_sink_id is not None:
        if old_sink_path == str(path):
            return
        try:
            logger.remove(old_sink_id)
        except ValueError:
            pass

    sink_id = logger.add(
        path,
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {extra[task_id]} | {name}:{line} - {message}",
        encoding="utf-8",
        backtrace=True,
        diagnose=False,
    )
    logger._zz_loguru_file_sink_id = sink_id
    logger._zz_loguru_file_sink_path = str(path)


def enable_faulthandler(log_path: os.PathLike | str | None = None):
    """启用 faulthandler: 段错误(SIGSEGV)等致命信号时打印所有线程的 C/Python 栈.

    native 扩展(ctranslate2/faster-whisper、duckdb、PyAV/ffmpeg)崩溃发出 SIGSEGV
    时, Python 的 try/except 接不住, 进程直接死且无任何堆栈. faulthandler 在收到
    SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL 时由 C 信号处理器直接 dump 出所有线程的
    调用栈, 据此可定位到底是哪个 native 库在并发下崩的.

    Args:
        log_path: 额外把崩溃栈写到该文件(批量跑时 stderr 可能被吞/错位, 落文件更稳).
            为 None 时只输出到 stderr. 该文件句柄需常驻(faulthandler 在信号处理时写),
            故挂到 logger 上防止被 GC.
    """
    import faulthandler

    # all_threads=True: 并发场景必须 dump 全部线程, 否则只看到主线程栈定位不到 worker.
    faulthandler.enable(all_threads=True)

    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 以行缓冲打开并常驻持有, 避免被 GC 关闭导致信号处理时写入失效.
        fh = open(path, "a", buffering=1)
        logger._zz_faulthandler_file = fh  # 防 GC
        faulthandler.enable(file=fh, all_threads=True)
        logger.info("faulthandler enabled, crash traceback -> stderr + {}", path)
    else:
        logger.info("faulthandler enabled, crash traceback -> stderr")


def print_dir_permissions(*dirs: Path):
    """打印当前进程用户及各目录的权限信息，便于排查挂载权限问题。"""
    cur_uid = os.getuid()
    cur_gid = os.getgid()
    try:
        cur_user = pwd.getpwuid(cur_uid).pw_name
    except KeyError:
        cur_user = str(cur_uid)
    logger.info("[PROC] 当前进程: user={}  uid={}  gid={}", cur_user, cur_uid, cur_gid)

    for p in dirs:
        if p.exists():
            s = p.stat()
            mode = stat.filemode(s.st_mode)
            try:
                owner = pwd.getpwuid(s.st_uid).pw_name
            except KeyError:
                owner = str(s.st_uid)
            try:
                group = grp.getgrgid(s.st_gid).gr_name
            except KeyError:
                group = str(s.st_gid)
            writable = os.access(p, os.W_OK)
            logger.info("[DIR] {}  mode={}  owner={}({})  group={}({})  writable={}",
                         p, mode, owner, s.st_uid, group, s.st_gid, writable)
        else:
            logger.info("[DIR] {}  (不存在)", p)


def _normalize_base_url(url: str) -> str:
    """Strip trailing /chat/completion(s) so OpenAIProvider gets a clean base URL."""
    import re
    return re.sub(r'/chat/completions?/?$', '', url.rstrip('/'))


def _setup_default_model_env():
    """Inject local default model env while keeping docker fully env-driven."""
    if is_docker():
        return
    os.environ.setdefault("MODEL_API_URL", DEFAULT_MODEL_API_URL)
    os.environ.setdefault("MODEL_API_KEY", DEFAULT_MODEL_API_KEY)
    os.environ.setdefault("MODEL_NAME", DEFAULT_MODEL_NAME)
    # 多 endpoint 默认 (非空时); 显式 export MODEL_API_URLS 可覆盖, 设空串关闭.
    if DEFAULT_MODEL_API_URLS:
        os.environ.setdefault("MODEL_API_URLS", DEFAULT_MODEL_API_URLS)


_registered_prices: set = set()


def _register_self_hosted_model_price(model_name: str):
    """为参考自建模型注册占位估价, 让 CostTracking 不再报
    'failed to resolve price for model openai:<name>' 的 warning.

    pydantic-ai 会把无 provider 前缀的模型解析成 openai:<model_name>,
    因此这里在 genai_prices 的 openai provider 下补一条匹配记录.
    """
    if os.environ.get('MODEL_API_PROFILE') == 'deepseek':
        # 官方付费 API 不套用参考项目的自建模型占位单价。
        return
    if not model_name or model_name in _registered_prices:
        return
    try:
        from decimal import Decimal
        from genai_prices.data_snapshot import get_snapshot, set_custom_snapshot, DataSnapshot
        from genai_prices.types import ModelInfo, ModelPrice, ClauseContains

        snap = get_snapshot()
        for p in snap.providers:
            if p.id == 'openai':
                if not any(m.id == model_name for m in p.models):
                    p.models.append(ModelInfo(
                        id=model_name,
                        name=model_name,
                        match=ClauseContains(contains=model_name),
                        prices=ModelPrice(
                            input_mtok=Decimal('0.056'),
                            output_mtok=Decimal('0.444'),
                        ),
                    ))
                break
        set_custom_snapshot(DataSnapshot(providers=snap.providers, from_auto_update=False))
        _registered_prices.add(model_name)
    except Exception as e:  # 价格注册失败不应影响主流程
        logger.debug(f"register self-hosted model price failed: {e}")


import itertools
import threading

# 多 endpoint 负载均衡: 设置 MODEL_API_URLS (逗号分隔多个 base_url) 即开启 task 级
# round-robin; 不设则退回单一 MODEL_API_URL, 行为与原来字节级一致 (docker 提交端
# 只注入单数 MODEL_API_URL, 走此 fallback, 不受影响).
_endpoint_cycle = None
_endpoint_cycle_lock = threading.Lock()


def _next_base_url() -> str:
    """返回下一个 base_url. 多 endpoint 时 round-robin 轮询, 否则退回单 endpoint.

    示例 (本地多机摊负载, 不同 task 轮流打不同推理机):
        export MODEL_API_URLS="http://10.164.41.71:8417/v1,http://10.164.169.63:8417/v1,http://10.166.163.101:8417/v1"
        # 第 1/4/7... 个 task -> .71, 第 2/5/8... -> .63, 第 3/6/9... -> .101

    不设 MODEL_API_URLS 时退回单一 MODEL_API_URL, 行为与原来字节级一致
    (docker 提交端只注入单数 MODEL_API_URL, 走此 fallback, 不受影响).
    """
    urls = os.environ.get("MODEL_API_URLS")
    if not urls:
        return _normalize_base_url(os.environ["MODEL_API_URL"])
    parsed = [_normalize_base_url(u.strip()) for u in urls.split(",") if u.strip()]
    if not parsed:
        return _normalize_base_url(os.environ["MODEL_API_URL"])
    if len(parsed) == 1:
        return parsed[0]
    # cycle + next 本身在单线程 asyncio 下无竞态; 加锁兼容多线程调用场景.
    global _endpoint_cycle
    with _endpoint_cycle_lock:
        if _endpoint_cycle is None:
            _endpoint_cycle = itertools.cycle(parsed)
        return next(_endpoint_cycle)


def get_model_config():
    _setup_default_model_env()
    model_name = os.environ["MODEL_NAME"]
    video_model = os.environ.get('MODEL_VIDEO_NAME', '').strip()
    task_id = _current_task_id.get()
    input_dir = os.environ.get('INPUT_DIR')
    if video_model and input_dir and task_id.startswith('task_'):
        if (Path(input_dir) / task_id / 'context/video/briefing.mp4').is_file():
            model_name = video_model
    _register_self_hosted_model_price(model_name)
    return {
        "base_url": _next_base_url(),
        "api_key": os.environ.get("MODEL_API_KEY", "empty"),
        "model_name": model_name,
    }


def _model_profile(model_name):
    profile_name = os.environ.get('MODEL_API_PROFILE', 'openai')
    if profile_name == 'openai':
        return None
    if profile_name == 'deepseek':
        from pydantic_ai.providers.deepseek import DeepSeekProvider
        # 锁定依赖尚不识别新官方别名 deepseek-flash，沿用 V4 的 reasoning_content 协议。
        name = 'deepseek-v4-flash' if model_name == 'deepseek-flash' else model_name
        return DeepSeekProvider.model_profile(name)
    raise ValueError('MODEL_API_PROFILE must be openai/deepseek')


def _solver_settings():
    mode = os.environ.get('MODEL_SOLVER_THINKING', 'default')
    if mode == 'default':
        return None
    if mode not in {'enabled', 'disabled'}:
        raise ValueError('MODEL_SOLVER_THINKING must be default/enabled/disabled')
    return OpenAIChatModelSettings(extra_body={'thinking': {'type': mode}})


def build_model():
    config = get_model_config()
    _env_provider = OpenAIProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        http_client=_build_http_client(),
    )
    model_auto = OpenAIChatModel(
        model_name=config["model_name"],
        provider=_env_provider,
        profile=_model_profile(config['model_name']),
        settings=_solver_settings(),
    )
    return model_auto


def build_openai_model():
    config = get_model_config()
    provider = OpenAIProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        http_client=_build_http_client(),
    )
    model = OpenAIChatModel(
        model_name=config["model_name"],
        provider=provider,
        profile=_model_profile(config['model_name']),
        settings=_solver_settings(),
    )
    return provider, model


def build_model_nothink():
    config = get_model_config()
    _env_provider = OpenAIProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        http_client=_build_http_client(),
    )
    thinking_format = os.environ.get('MODEL_THINKING_FORMAT', 'legacy')
    if thinking_format == 'legacy':
        settings = OpenAIChatModelSettings(extra_body={
            'enable_thinking': False,
            'chat_template_kwargs': {'enable_thinking': False},
        })
    elif thinking_format in {'ecnu', 'deepseek'}:
        settings = OpenAIChatModelSettings(extra_body={'thinking': {'type': 'disabled'}})
    elif thinking_format == 'none':
        settings = OpenAIChatModelSettings()
    else:
        raise ValueError('MODEL_THINKING_FORMAT must be legacy/ecnu/deepseek/none')
    model_nothink = OpenAIChatModel(
        model_name=config["model_name"],
        provider=_env_provider,
        settings=settings,
        profile=_model_profile(config['model_name']),
    )
    return model_nothink


if __name__ == '__main__':
    def _test_normalize_base_url():
        cases = [
            ("http://host/v1/chat/completions", "http://host/v1"),
            ("http://host/v1/chat/completion", "http://host/v1"),
            ("http://host/v1/chat/completions/", "http://host/v1"),
            ("http://host/v1/chat/completion/", "http://host/v1"),
            ("http://host/v1", "http://host/v1"),
            ("http://host/v1/", "http://host/v1"),
        ]
        all_pass = True
        for url, expected in cases:
            result = _normalize_base_url(url)
            status = "✓" if result == expected else "✗"
            if result != expected:
                all_pass = False
            print(f"  {status}  {url!r:50s} => {result!r}  (expected {expected!r})")
        print("All passed!" if all_pass else "Some tests FAILED.")

    _test_normalize_base_url()
