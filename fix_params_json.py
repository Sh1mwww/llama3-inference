#!/usr/bin/env python3
"""
修复 params.json，将 vocab_size 从 TP 分片值改为完整值
"""
import json
from pathlib import Path
import shutil

params_path = Path('/home/roger/.llama/checkpoints/Llama3.1-70B/params.json')

# 备份
backup_path = params_path.with_suffix('.json.backup')
if not backup_path.exists():
    shutil.copy(params_path, backup_path)
    print(f"✅ 已备份到: {backup_path}")

# 读取
params = json.loads(params_path.read_text())
print(f"原始 vocab_size: {params['vocab_size']}")

# 修改
params['vocab_size'] = 128256
print(f"新的 vocab_size: {params['vocab_size']}")

# 写回
params_path.write_text(json.dumps(params, indent=2) + '\n')
print(f"✅ 已更新: {params_path}")
