import re
import pathlib

files_to_fix = [
    'ecodiaos/systems/axon/audit.py',
    'ecodiaos/systems/axon/executors/internal.py',
    'ecodiaos/systems/axon/executors/data.py',
    'ecodiaos/systems/axon/executors/communication.py',
    'ecodiaos/systems/axon/executors/__init__.py',
    'ecodiaos/systems/axon/pipeline.py',
    'ecodiaos/systems/axon/service.py',
    'ecodiaos/systems/nova/intent_router.py',
    'ecodiaos/systems/nova/service.py',
    'ecodiaos/systems/simula/applicator.py',
]

for fpath in files_to_fix:
    p = pathlib.Path(fpath)
    if not p.exists():
        print(f'SKIP (not found): {fpath}')
        continue
    text = p.read_text(encoding='utf-8')
    has_any = bool(re.search(r'from typing import[^\n]*\bAny\b', text))
    modified = text
    modified = re.sub(r'\bparams: dict\b(?!\[)', 'params: dict[str, Any]', modified)
    modified = re.sub(r'-> dict\b(?!\[)(\s*:)', r'-> dict[str, Any]\1', modified)
    modified = re.sub(r'\bdict\b(?!\[) \| None\b', 'dict[str, Any] | None', modified)
    changed = (modified != text)
    if changed:
        if not has_any and 'dict[str, Any]' in modified:
            modified = re.sub(
                r'(from typing import )([^\n]+)',
                lambda m: m.group(1) + ('Any, ' + m.group(2) if 'Any' not in m.group(2) else m.group(2)),
                modified, count=1
            )
        p.write_text(modified, encoding='utf-8')
        print(f'FIXED: {fpath}')
    else:
        print(f'no change: {fpath}')
