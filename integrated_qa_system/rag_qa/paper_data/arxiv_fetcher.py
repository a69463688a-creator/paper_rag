"""
arXiv API 论文检索与下载模块

通过 arXiv 官方 API (https://info.arxiv.org/help/api/) 检索和下载学术论文 PDF
使用标准库 urllib + xml.etree，不新增外部依赖
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import ssl
import time
from typing import List, Dict, Optional
from base.logger import logger

# 修复 Windows 下 SSL 证书验证问题：优先使用 certifi 的 CA bundle
try:
    import certifi
    _SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CONTEXT = ssl.create_default_context()

# arXiv API 基础地址
ARXIV_API_BASE = "http://export.arxiv.org/api/query"

# 请求间隔（秒），arxiv 建议每次请求间隔至少 3 秒
REQUEST_DELAY = 3.0


class ArxivFetcher:
    """
    arXiv 论文检索与下载器

    用法:
        fetcher = ArxivFetcher(data_dir="../data/paper_data")
        papers = fetcher.search("attention is all you need", max_results=5)
        fetcher.download_pdf("1706.03762")
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化 arXiv 获取器

        :param data_dir: 论文存储目录，默认使用 paper_data 目录
        """
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_dir = os.path.join(os.path.dirname(current_dir), "data", "paper_data")
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self._last_request_time = 0.0

    def _rate_limit(self):
        """控制请求频率，确保两次请求间隔 >= REQUEST_DELAY 秒"""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _build_query_url(self, query: str, max_results: int = 20, start: int = 0) -> str:
        """
        构建 arXiv API 查询 URL

        :param query: 搜索关键词
        :param max_results: 最大返回数（每页）
        :param start: 分页偏移
        :return: 完整 API URL
        """
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        query_string = urllib.parse.urlencode(params)
        return f"{ARXIV_API_BASE}?{query_string}"

    def _parse_atom_response(self, xml_content: str) -> List[Dict]:
        """
        解析 arXiv API 返回的 Atom XML

        :param xml_content: Atom XML 字符串
        :return: 论文元数据列表
        """
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(xml_content)
        papers = []

        for entry in root.findall("atom:entry", ns):
            paper = {}

            # 论文 ID（去掉 arXiv URL 前缀，保留纯 ID）
            id_elem = entry.find("atom:id", ns)
            if id_elem is not None and id_elem.text:
                paper["arxiv_id"] = id_elem.text.split("/abs/")[-1]

            # 标题（去除多余空白）
            title_elem = entry.find("atom:title", ns)
            paper["title"] = " ".join(title_elem.text.split()) if title_elem is not None and title_elem.text else ""

            # 摘要
            summary_elem = entry.find("atom:summary", ns)
            paper["summary"] = " ".join(summary_elem.text.split()) if summary_elem is not None and summary_elem.text else ""

            # 作者
            authors = []
            for author in entry.findall("atom:author", ns):
                name_elem = author.find("atom:name", ns)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)
            paper["authors"] = authors

            # 发布时间
            published_elem = entry.find("atom:published", ns)
            paper["published"] = published_elem.text if published_elem is not None else ""

            # 分类（主分类 + 次级分类）
            categories = []
            primary_category = ""
            for cat in entry.findall("atom:category", ns):
                term = cat.attrib.get("term", "")
                categories.append(term)
                if cat.attrib.get("{http://arxiv.org/schemas/atom}primary") == "1":
                    primary_category = term
            paper["categories"] = categories
            paper["primary_category"] = primary_category

            # PDF 链接
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            if not pdf_url and paper.get("arxiv_id"):
                pdf_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}"
            paper["pdf_url"] = pdf_url

            papers.append(paper)

        return papers

    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        """
        搜索 arXiv 论文

        :param query: 搜索关键词，支持 arXiv 查询语法
                      例如: "ti:transformer AND ti:attention" 按标题搜索
        :param max_results: 最大返回数量
        :return: 论文元数据列表，每项包含 arxiv_id, title, summary, authors, pdf_url 等字段
        """
        self._rate_limit()
        url = self._build_query_url(query, max_results=max_results)
        logger.info(f"arXiv 搜索: '{query}' (最大 {max_results} 条)")

        try:
            with urllib.request.urlopen(url, timeout=30, context=_SSL_CONTEXT) as response:
                xml_content = response.read().decode("utf-8")
            papers = self._parse_atom_response(xml_content)
            logger.info(f"arXiv 搜索完成: 找到 {len(papers)} 篇论文")
            return papers
        except Exception as e:
            logger.error(f"arXiv API 请求失败: {e}")
            return []

    def download_pdf(self, arxiv_id: str, output_dir: Optional[str] = None) -> Optional[str]:
        """
        下载论文 PDF

        :param arxiv_id: arXiv 论文 ID（如 "1706.03762"）
        :param output_dir: 输出目录，默认使用 data_dir
        :return: 下载后的文件路径，失败返回 None
        """
        if output_dir is None:
            output_dir = self.data_dir

        os.makedirs(output_dir, exist_ok=True)

        filename = f"{arxiv_id}.pdf"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            logger.info(f"论文已存在，跳过下载: {filepath}")
            return filepath

        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        self._rate_limit()
        logger.info(f"下载论文: {arxiv_id} -> {filepath}")

        try:
            req = urllib.request.Request(
                pdf_url,
                headers={"User-Agent": "PaperRAG/1.0 (Academic Research Assistant)"},
            )
            with urllib.request.urlopen(req, timeout=60, context=_SSL_CONTEXT) as response:
                pdf_data = response.read()

            with open(filepath, "wb") as f:
                f.write(pdf_data)

            logger.info(f"论文下载完成: {filepath} ({len(pdf_data)} bytes)")
            return filepath
        except Exception as e:
            logger.error(f"论文下载失败 {arxiv_id}: {e}")
            return None

    def search_and_download(self, query: str, max_results: int = 10, download: bool = True) -> List[Dict]:
        """
        搜索并下载论文（便捷方法）

        :param query: 搜索关键词
        :param max_results: 最大数量
        :param download: 是否下载 PDF
        :return: 论文元数据列表（含本地文件路径）
        """
        papers = self.search(query, max_results=max_results)

        if download:
            for paper in papers:
                if paper.get("arxiv_id"):
                    filepath = self.download_pdf(paper["arxiv_id"])
                    paper["local_path"] = filepath

        return papers


if __name__ == "__main__":
    fetcher = ArxivFetcher()
    results = fetcher.search("ti:transformer AND ti:attention", max_results=3)

    for i, paper in enumerate(results):
        print(f"\n--- 论文 {i+1} ---")
        print(f"arXiv ID: {paper['arxiv_id']}")
        print(f"标题: {paper['title']}")
        print(f"作者: {', '.join(paper['authors'][:3])}")
        print(f"主分类: {paper['primary_category']}")
        print(f"PDF: {paper['pdf_url']}")
