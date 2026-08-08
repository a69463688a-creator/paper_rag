import os
import sys

module_dir=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.dirname(module_dir)
#   插入系统路径 优先加载
if module_dir not in sys.path:
    sys.path.insert(0,module_dir)

if project_root not in sys.path:
    sys.path.insert(0,project_root)

# print(sys.path)
from config import Config
from logger import logger