import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入配置
from base.config import Config

# 添加配置中的路径到 sys.path
sys.path.append(Config().DOCUMENT_LOADERS_DIR)

# 条件导入各个文档加载器，允许在缺少依赖时继续运行
try:
    from .doc_loader import *
except ImportError:
    pass

try:
    from .ppt_loader import *
except ImportError:
    pass

try:
    from .img_loader import *
except ImportError:
    pass

try:
    from .pdf_loader import *
except ImportError:
    pass
