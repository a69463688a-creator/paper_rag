"""测试配置模块"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConfig:
    """全局配置"""

    @pytest.fixture(autouse=True)
    def setup(self):
        from base.config import Config
        self.config = Config()

    def test_milvus_config(self):
        """Milvus 配置项非空"""
        assert len(self.config.MILVUS_HOST) > 0
        assert int(self.config.MILVUS_PORT) > 0
        assert len(self.config.MILVUS_DATABASE_NAME) > 0

    def test_retrieval_params(self):
        """检索参数在合理范围内"""
        assert 1 <= self.config.RETRIEVAL_K <= 20
        assert 1 <= self.config.CANDIDATE_M <= 10
        assert self.config.CHILD_CHUNK_SIZE > 0
        assert self.config.CHUNK_OVERLAP >= 0

    def test_llm_config(self):
        """LLM 配置非空"""
        assert len(self.config.LLM_MODEL) > 0
        assert len(self.config.DASHSCOPE_BASE_URL) > 0

    def test_valid_sources_is_list(self):
        """领域列表是 list 类型"""
        assert isinstance(self.config.VALID_SOURCES, list)
        assert len(self.config.VALID_SOURCES) > 0

    def test_paths_exist(self):
        """路径配置指向存在的目录"""
        assert os.path.exists(self.config.PROJECT_ROOT)
        assert os.path.exists(self.config.DATA_DIR)

    def test_mysql_config(self):
        """MySQL 配置非空"""
        assert len(self.config.MYSQL_HOST) > 0
        assert len(self.config.MYSQL_DATABASE) > 0
