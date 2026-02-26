import re
import pathlib

# Fix specific no-untyped-def occurrences identified by mypy
fixes = {
    'ecodiaos/systems/axon/executors/data.py': [
        # line 294: def __init__(self, redis_client=None)
        ('def __init__(self, redis_client=None) -> None:\n        self._redis = redis_client\n        self._logger = logger.bind(system="axon.executor.schedule_event")',
         'def __init__(self, redis_client: Any = None) -> None:\n        self._redis = redis_client\n        self._logger = logger.bind(system="axon.executor.schedule_event")'),
        # line 442: def __init__(self, redis_client=None)
        ('def __init__(self, redis_client=None) -> None:\n        self._redis = redis_client\n        self._logger = logger.bind(system="axon.executor.set_reminder")',
         'def __init__(self, redis_client: Any = None) -> None:\n        self._redis = redis_client\n        self._logger = logger.bind(system="axon.executor.set_reminder")'),
    ],
    'ecodiaos/systems/axon/executors/communication.py': [
        ('def __init__(self, redis_client=None) -> None:\n        self._redis = redis_client\n        self._logger = logger.bind(system="axon.executor.notification")',
         'def __init__(self, redis_client: Any = None) -> None:\n        self._redis = redis_client\n        self._logger = logger.bind(system="axon.executor.notification")'),
    ],
    'ecodiaos/systems/axon/executors/__init__.py': [
        ('def build_default_registry(\n    memory=None,\n    voxis=None,\n    redis_client=None,\n)',
         'def build_default_registry(\n    memory: Any = None,\n    voxis: Any = None,\n    redis_client: Any = None,\n)'),
    ],
    'ecodiaos/systems/axon/pipeline.py': [
        ('    def set_atune(self, atune) -> None:',
         '    def set_atune(self, atune: Any) -> None:'),
        ('async def _run_step_with_timeout(\n    executor,\n',
         'async def _run_step_with_timeout(\n    executor: Any,\n'),
    ],
    'ecodiaos/systems/axon/service.py': [
        ('    redis_client=None,\n    instance_id: str = "eos-default",\n)',
         '    redis_client: Any = None,\n    instance_id: str = "eos-default",\n)'),
        ('    def set_atune(self, atune) -> None:',
         '    def set_atune(self, atune: Any) -> None:'),
        ('    def register_executor(self, executor) -> None:',
         '    def register_executor(self, executor: Any) -> None:'),
    ],
    'ecodiaos/systems/evo/service.py': [
        ('    async def _generate_goal_from_hypothesis(self, hypothesis) -> None:',
         '    async def _generate_goal_from_hypothesis(self, hypothesis: Any) -> None:'),
    ],
    'ecodiaos/systems/nova/service.py': [],  # handled by dict script already
}

for fpath, replacements in fixes.items():
    p = pathlib.Path(fpath)
    if not p.exists():
        print(f'SKIP (not found): {fpath}')
        continue
    text = p.read_text(encoding='utf-8')
    original = text
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new, 1)
        else:
            print(f'  WARNING: pattern not found in {fpath}: {repr(old[:60])}')
    if text != original:
        # Ensure Any is imported
        if 'Any' in text and 'from typing import' in text:
            if not re.search(r'from typing import[^\n]*\bAny\b', text):
                text = re.sub(
                    r'(from typing import )([^\n]+)',
                    lambda m: m.group(1) + ('Any, ' + m.group(2) if 'Any' not in m.group(2) else m.group(2)),
                    text, count=1
                )
        p.write_text(text, encoding='utf-8')
        print(f'FIXED: {fpath}')
    else:
        print(f'no change: {fpath}')
