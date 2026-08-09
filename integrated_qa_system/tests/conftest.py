"""pytest 共享 fixtures"""
import sys
import os
import pytest

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def project_root():
    """项目根目录"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def config():
    """加载全局配置"""
    from base.config import Config
    return Config()
