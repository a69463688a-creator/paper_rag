"""
论文 PDF 图表提取与视觉描述模块

从论文 PDF 中提取图表图片，使用 Ollama 本地视觉模型（qwen3-vl:8b）
生成图表内容描述，供多模态检索使用。

依赖:
    - PyMuPDF (fitz): PDF 解析与图片提取
    - ollama: 本地视觉模型调用
    - Pillow: 图片处理
"""

import fitz  # PyMuPDF
import os
import re
import hashlib
from typing import List, Dict, Optional, Iterator
from base.logger import logger

# 图片最小尺寸阈值（像素），过滤掉水印、icon 等小图
MIN_FIGURE_WIDTH = 100
MIN_FIGURE_HEIGHT = 100

# 视觉模型名称（Ollama）
VISION_MODEL = "qwen3-vl:8b"


class FigureExtractor:
    """
    论文图表提取器

    用法:
        extractor = FigureExtractor()
        figures = extractor.extract_from_pdf("paper.pdf")
        for fig in figures:
            print(fig["caption"], fig["description"])
    """

    def __init__(self, output_dir: Optional[str] = None, vision_model: Optional[str] = None):
        """
        :param output_dir: 提取的图片保存目录
        :param vision_model: Ollama 视觉模型名称
        """
        if output_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, "figures")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.vision_model = vision_model or VISION_MODEL

    def extract_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        从 PDF 中提取所有图表

        :param pdf_path: PDF 文件路径
        :return: 图表信息列表，每项包含 image_path, caption, description, page_num, bbox
        """
        doc = fitz.open(pdf_path)
        paper_id = self._paper_id_from_path(pdf_path)
        all_figures = []

        for page_num, page in enumerate(doc):
            text_blocks = page.get_text("blocks")
            img_list = page.get_image_info(xrefs=True)

            for img_idx, img in enumerate(img_list):
                xref = img.get("xref")
                if not xref:
                    continue

                bbox = img["bbox"]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                if width < MIN_FIGURE_WIDTH or height < MIN_FIGURE_HEIGHT:
                    continue

                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK → RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_bytes = pix.tobytes("png")
                except Exception as e:
                    logger.warning(f"提取图片失败 (page={page_num+1}, xref={xref}): {e}")
                    continue

                img_filename = f"{paper_id}_p{page_num+1}_fig{img_idx}.png"
                img_path = os.path.join(self.output_dir, img_filename)
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

                caption = self._find_caption(text_blocks, bbox)

                all_figures.append({
                    "figure_id": hashlib.md5(img_bytes).hexdigest()[:16],
                    "paper_id": paper_id,
                    "image_path": img_path,
                    "caption": caption,
                    "page_num": page_num + 1,
                    "bbox": bbox,
                })

        doc.close()
        logger.info(f"图表提取完成: {pdf_path} → {len(all_figures)} 张图")
        return all_figures

    def describe_figures(self, figures: List[Dict]) -> List[Dict]:
        """
        使用 Ollama 视觉模型为图表生成文字描述

        :param figures: extract_from_pdf 的返回值
        :return: 追加了 description 字段的图表列表
        """
        import ollama

        for fig in figures:
            caption = fig.get("caption", "")
            prompt = (
                "You are an academic paper analyst. Describe this figure in detail. "
                "Include: (1) figure type (architecture diagram, chart, table image, etc.), "
                "(2) key data or components shown, (3) the main conclusion or takeaway. "
                f"The figure caption is: '{caption}'."
            )
            try:
                # 截断过长的 caption，避免超出 context 限制
                max_caption_len = 1500
                if len(caption) > max_caption_len:
                    caption = caption[:max_caption_len] + "..."
                    prompt = (
                        "You are an academic paper analyst. Describe this figure in detail. "
                        "Include: (1) figure type (architecture diagram, chart, table image, etc.), "
                        "(2) key data or components shown, (3) the main conclusion or takeaway. "
                        f"The figure caption is: '{caption}'."
                    )
                response = ollama.chat(
                    model=self.vision_model,
                    messages=[{
                        "role": "user",
                        "content": prompt,
                        "images": [fig["image_path"]],
                    }],
                    options={"num_ctx": 8192}
                )
                fig["description"] = response["message"]["content"]
                logger.info(f"视觉描述完成: {fig['figure_id']} ({fig['caption'][:50]}...)")
            except Exception as e:
                logger.error(f"视觉模型调用失败 {fig['figure_id']}: {e}")
                fig["description"] = fig.get("caption", "")

        return figures

    def _find_caption(self, text_blocks: List, img_bbox: tuple) -> str:
        """
        在图片附近查找 Figure/Table 标题

        :param text_blocks: 页面文本块列表
        :param img_bbox: 图片边界框 (x0, y0, x1, y1)
        :return: 标题文本，未找到则返回空字符串
        """
        candidates = []

        for block in text_blocks:
            block_bbox = block[:4]
            block_text = block[4]
            block_center_y = (block_bbox[1] + block_bbox[3]) / 2

            # 标题通常在图片正上方或正下方（80pt 范围内）
            if block_center_y < img_bbox[1] and abs(block_center_y - img_bbox[1]) < 80:
                candidates.append(("above", block_text.strip(), img_bbox[1] - block_center_y))
            elif block_center_y > img_bbox[3] and abs(block_center_y - img_bbox[3]) < 80:
                candidates.append(("below", block_text.strip(), block_center_y - img_bbox[3]))

        for _, text, _ in sorted(candidates, key=lambda x: x[2]):
            if re.search(r"(Figure|Fig\.?|图|Table)\s*\d+", text, re.IGNORECASE):
                return text.strip()

        return candidates[0][1] if candidates else ""

    @staticmethod
    def _paper_id_from_path(pdf_path: str) -> str:
        """从 PDF 文件路径提取论文 ID（文件名去扩展名）"""
        basename = os.path.basename(pdf_path)
        return os.path.splitext(basename)[0]


if __name__ == "__main__":
    import sys
    from base.config import Config

    conf = Config()
    data_dir = os.path.join(conf.DATA_DIR, "paper_data")
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"请先将论文 PDF 放入 {data_dir}")
        sys.exit(1)

    pdf_path = os.path.join(data_dir, pdf_files[0])
    print(f"测试文件: {pdf_path}")

    extractor = FigureExtractor()
    figures = extractor.extract_from_pdf(pdf_path)
    print(f"\n提取到 {len(figures)} 张图:")
    for fig in figures:
        print(f"  [{fig['figure_id']}] p{fig['page_num']}: {fig['caption'][:80]}")
