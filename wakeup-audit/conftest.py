# 使 pytest 在 `python -I` 隔离模式下也能导入 app 包
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
