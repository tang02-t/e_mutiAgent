from __future__ import annotations

import os
import pickle
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

"""
Milvus 向量知识库初始化与入库脚本。

功能概览：
- 连接 Milvus
- （如不存在）创建 collection：主键 id + 文本字段 + 向量字段 + 元数据字段
- 创建向量索引并 load
- 递归读取本地目录下的文本（.txt 或 MinerU full.md）
- 调用 embedding 模型把文本编码为向量
- 批量插入 Milvus 并 flush

支持两种模式：
- 默认模式（.txt）：读取 data/ 目录下的所有 .txt 文件
- Markdown 模式（--md-mode）：读取 MinerU 导出的 full.md，按 ## / ### 标题做父子分块入库

运行：
`python -m src.tools.milvus_setup --data-dir data/fast_md --md-mode`
"""

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)

from src.utils.config import load_config
from src.utils.embedding import LocalEmbeddingModel
from src.utils.logging import get_logger


logger = get_logger(__name__)


# ----------------------------------------------------------------------
# 父子分块：Markdown 解析
# ----------------------------------------------------------------------

HEADING_RE = re.compile(r'^(#{1,3})\s+(.+?)\s*$', re.MULTILINE)
IMAGE_RE = re.compile(r'!\[.*?\]\((.+?)\)', re.MULTILINE)
FORMULA_RE = re.compile(r'^\s*\$\$.*?\$\$\s*$', re.MULTILINE | re.DOTALL)
TABLE_RE = re.compile(r'<table>.*?</table>', re.DOTALL | re.IGNORECASE)
REFERENCE_SECTION_RE = re.compile(
    r'^#+\s*【?\s*参考文献\s*】?\s*$',
    re.MULTILINE | re.IGNORECASE
)


@dataclass
class ChildChunk:
    """子块：最小检索单元（段落/公式/表格 + 关联图片）。"""
    chunk_id: str
    parent_id: str
    parent_heading: str        # 所属父块的标题，如 "1.2.1 外部检查"
    section_heading: str       # 本子块所在的最细粒度标题（## 或 ###）
    text: str                 # 纯文本内容（不含 markdown 图片语法）
    image_paths: List[str]     # 属于本段落的图片路径列表
    chunk_category: str = "paragraph"  # 块类型: "paragraph" | "formula" | "table"


@dataclass
class ParentChunk:
    """父块：完整章节内容（一个 ## / ### 标题下的所有内容）。"""
    chunk_id: str
    heading: str
    heading_level: int         # 2 = ##, 3 = ###
    doc_name: str
    source: str
    title: str
    published_date: str
    full_text: str            # 章节完整文本
    child_ids: List[str]      # 直接下属子块的 ID 列表


@dataclass
class ChunkResult:
    """一篇 md 文件的分块结果。"""
    doc_name: str
    source: str
    title: str
    published_date: str
    parents: List[ParentChunk]
    children: List[ChildChunk]


def _chunk_markdown(path: Path) -> ChunkResult:
    """
    将一篇 full.md 解析为父子分块结构。

    分块策略：
    - 预处理：删除参考文献章节（# 参考文献 / # 【参考文献】）
    - 父块：以 ## / ### 标题为界，每个标题（含标题行）下所有内容为一个父块
    - 子块类型：
      - 段落（paragraph）：普通文本段落（以空行分隔）
      - 公式（formula）：独立的 $$...$$ 块，包含编号和变量说明
      - 表格（table）：独立的 <table>...</table> 块
    - 图片归属：图片跟在哪个子段落后面，就归哪个子段落所有
    - 摘要（# 标题至第一个 ## 之间的内容）作为 level=1 顶级父块
    """
    content = path.read_text(encoding="utf-8", errors="ignore")

    # --- 删除参考文献章节 ---
    # 匹配: # 参考文献 / # 【参考文献】 / ## 参考文献 等
    ref_match = REFERENCE_SECTION_RE.search(content)
    if ref_match:
        # 删除参考文献章节到文档末尾
        content = content[:ref_match.start()]
        logger.debug("Removed reference section from %s", path.name)

    lines = content.splitlines(keepends=False)

    # --- 提取元数据 ---
    title_m = re.search(r'^#\s+(.+?)\s*$', content, re.MULTILINE)
    title = title_m.group(1).strip() if title_m else ""
    date_m = re.search(r'【收稿日期】\s*(\d{4}-\d{2}-\d{2})', content)
    published_date = date_m.group(1).strip() if date_m else ""
    doc_name = path.parent.name
    source = str(path.resolve())

    # --- 收集图片路径（绝对路径） ---
    images_dir = path.parent / "images"
    all_images: List[str] = []
    if images_dir.is_dir():
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
                all_images.append(str(img_file.resolve()))

    # --- 解析标题位置 ---
    heading_rows: List[Tuple[int, int, str]] = []  # (line_index, level, text)
    for i, line in enumerate(lines):
        stripped = line.strip()
        m = HEADING_RE.match(stripped)
        if m:
            heading_rows.append((i, len(m.group(1)), m.group(2).strip()))

    # --- 构建父块 + 子块 ---
    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []

    def make_id(*parts: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, "|".join(parts)))

    def _parse_section(
        section_lines: List[str],
        section_heading: str,
        parent_id: str,
        doc_name: str,
        path: Path,
        max_chunk_chars: int = 800,
    ) -> Tuple[List[ChildChunk], List[str]]:
        """
        解析一个章节内的所有子块（段落/公式/表格）。

        参数：
        - max_chunk_chars: 子块最大字符数，超过则拆分为多个块
        """
        section_children: List[ChildChunk] = []

        # 先提取章节内所有图片的绝对路径
        section_img_abs = _extract_images_from_section(section_lines, path)
        img_iter = iter(section_img_abs)

        # 收集当前累积的段落行和图片
        cur_lines: List[str] = []
        cur_imgs: List[str] = []
        seq = 0

        def _flush_current_paragraph(is_final: bool = False) -> None:
            """将当前累积的段落作为一个子块保存。"""
            nonlocal seq, cur_lines, cur_imgs
            if not cur_lines:
                return

            text = "\n".join(cur_lines).strip()
            if not text:
                cur_lines, cur_imgs = [], []
                return

            # 如果文本过长，拆分成多个块
            while len(text) > max_chunk_chars:
                # 找到一个合适的断点（句号或换行附近）
                split_point = text.rfind('。', 0, max_chunk_chars)
                if split_point == -1:
                    split_point = text.rfind('\n', 0, max_chunk_chars)
                if split_point == -1:
                    split_point = max_chunk_chars

                part_text = text[:split_point + 1]
                part_imgs = cur_imgs[:1] if cur_imgs else []  # 只在第一块保留图片

                section_children.append(ChildChunk(
                    chunk_id=make_id(doc_name, section_heading, "p", str(seq)),
                    parent_id=parent_id,
                    parent_heading=section_heading,
                    section_heading=section_heading,
                    text=part_text,
                    image_paths=part_imgs,
                    chunk_category="paragraph",
                ))
                seq += 1
                text = text[split_point + 1:]

                # 清理已使用的图片
                if cur_imgs:
                    cur_imgs = cur_imgs[1:]

            # 保存剩余文本
            if text.strip():
                section_children.append(ChildChunk(
                    chunk_id=make_id(doc_name, section_heading, "p", str(seq)),
                    parent_id=parent_id,
                    parent_heading=section_heading,
                    section_heading=section_heading,
                    text=text,
                    image_paths=cur_imgs[:],
                    chunk_category="paragraph",
                ))
                seq += 1

            cur_lines, cur_imgs = [], []

        i = 0
        while i < len(section_lines):
            line = section_lines[i]
            stripped = line.strip()

            # 检测公式块（多行 $$...$$）
            if stripped.startswith('$$'):
                _flush_current_paragraph()

                # 收集整个公式块
                formula_lines = [line]
                j = i + 1
                while j < len(section_lines) and '$$' not in section_lines[j - 1]:
                    formula_lines.append(section_lines[j])
                    j += 1
                if j < len(section_lines):
                    formula_lines.append(section_lines[j])
                    j += 1

                formula_text = "\n".join(formula_lines).strip()
                section_children.append(ChildChunk(
                    chunk_id=make_id(doc_name, section_heading, "f", str(seq)),
                    parent_id=parent_id,
                    parent_heading=section_heading,
                    section_heading=section_heading,
                    text=formula_text,
                    image_paths=[],
                    chunk_category="formula",
                ))
                seq += 1
                i = j
                continue

            # 检测表格块
            if stripped.startswith('<table'):
                _flush_current_paragraph()

                # 收集整个表格块
                table_lines = [line]
                j = i + 1
                while j < len(section_lines):
                    t_line = section_lines[j]
                    table_lines.append(t_line)
                    if t_line.strip().lower().endswith('</table>'):
                        break
                    j += 1

                table_text = "\n".join(table_lines).strip()
                section_children.append(ChildChunk(
                    chunk_id=make_id(doc_name, section_heading, "t", str(seq)),
                    parent_id=parent_id,
                    parent_heading=section_heading,
                    section_heading=section_heading,
                    text=table_text,
                    image_paths=[],
                    chunk_category="table",
                ))
                seq += 1
                i = j + 1
                continue

            # 处理图片行：图片跟在段落后面，归入当前段落
            img_m = IMAGE_RE.findall(line)
            if img_m:
                for _r in img_m:
                    try:
                        cur_imgs.append(next(img_iter))
                    except StopIteration:
                        pass
                cur_lines.append(line)
            elif stripped == "":
                # 空行：结束当前段落
                _flush_current_paragraph()
            else:
                cur_lines.append(line)

            i += 1

        # 处理最后累积的段落
        _flush_current_paragraph(is_final=True)

        return section_children, []

    def _extract_images_from_section(section_lines: List[str], path: Path) -> List[str]:
        """提取章节内所有图片的绝对路径。"""
        section_text = "\n".join(section_lines)
        img_abs: List[str] = []
        for img_rel in IMAGE_RE.findall(section_text):
            img_rel = img_rel.strip()
            if img_rel:
                img_abs.append(str((path.parent / img_rel).resolve()))
        return img_abs

    # 1) 摘要块：# 标题行 到 第一个 ## 之间的内容
    abstract_end = heading_rows[0][0] if heading_rows else len(lines)
    abstract_lines = lines[:abstract_end]
    abs_text = "\n".join(abstract_lines).strip()
    # 父块文本最大字符数（模型限制 8192 token ≈ 12000 字符）
    MAX_PARENT_CHARS = 12000

    def _truncate_text(text: str, max_chars: int = MAX_PARENT_CHARS) -> str:
        """截断文本，超长部分用占位符替代。"""
        if len(text) > max_chars:
            return text[:max_chars] + "\n...（内容已截断）"
        return text

    if abs_text:
        abs_id = make_id(doc_name, "摘要")
        parents.append(ParentChunk(
            chunk_id=abs_id,
            heading="摘要",
            heading_level=1,
            doc_name=doc_name,
            source=source,
            title=title,
            published_date=published_date,
            full_text=_truncate_text(abs_text),
            child_ids=[],
        ))
        # 摘要的子块
        abs_children, _ = _parse_section(abstract_lines, "摘要", abs_id, doc_name, path)
        children.extend(abs_children)

    # 2) 章节块：每个 ## / ### 标题
    for idx, (h_line, h_level, h_text) in enumerate(heading_rows):
        start = h_line
        end = heading_rows[idx + 1][0] if idx + 1 < len(heading_rows) else len(lines)
        section_lines = lines[start:end]
        section_text = "\n".join(section_lines).strip()

        p_id = make_id(doc_name, h_text)

        # 解析子块
        section_children, _ = _parse_section(section_lines, h_text, p_id, doc_name, path)
        children.extend(section_children)

        # 父块 child_ids
        child_ids = [c.chunk_id for c in section_children]

        # 父块文本截断（防止超过模型 token 限制）
        parent_text = _truncate_text(section_text)

        parents.append(ParentChunk(
            chunk_id=p_id,
            heading=h_text,
            heading_level=h_level,
            doc_name=doc_name,
            source=source,
            title=title,
            published_date=published_date,
            full_text=parent_text,
            child_ids=child_ids,
        ))

    return ChunkResult(
        doc_name=doc_name,
        source=source,
        title=title,
        published_date=published_date,
        parents=parents,
        children=children,
    )


def _connect_milvus(
    *,
    uri: str | None = None,
    token: str | None = None,
    host: str | None = None,
    port: int | None = None,
) -> None:
    """
    连接 Milvus / Zilliz Cloud 服务。

    支持两种连接方式（二选一）：
    - **Zilliz Cloud**：使用 `uri + token`
    - **自建 Milvus**：使用 `host + port`

    参数：
    - **uri**: Zilliz Cloud 的连接 URI（通常以 https:// 开头）
    - **token**: Zilliz Cloud 的 token（常见为 "<username>:<password>"）
    - **host**: 自建 Milvus 地址
    - **port**: 自建 Milvus 端口（通常 19530）
    """
    if uri and token:
        if "://" not in uri:
            raise ValueError(
                "Zilliz Cloud 的 uri 必须是完整连接地址，例如以 https:// 或 tcp:// 开头；"
                f"你当前填的是：{uri!r}（看起来像 project/cluster id，不是 uri）。"
            )
        connections.connect(alias="default", uri=uri, token=token)
        return
    if host and port is not None:
        connections.connect(alias="default", host=host, port=str(port))
        return
    raise ValueError("Milvus connection info missing: provide (uri, token) or (host, port).")


def _create_collection_if_not_exists(
    name: str,
    text_field: str,
    vector_field: str,
    dim: int,
    enable_metadata: bool = False,
    parent_child: bool = False,
) -> Collection:
    """
    若 collection 不存在则创建；存在则直接打开并返回。

    创建内容：
    - `id`：INT64 主键，auto_id=True
    - `text_field`：VARCHAR，用于存原文
    - `vector_field`：FLOAT_VECTOR，用于存 embedding
    - `chunk_id`：VARCHAR，块唯一标识符（UUID）
    - `chunk_type`：VARCHAR，"parent" | "child"
    - `parent_id`：VARCHAR，父块 chunk_id；父块自身则与 chunk_id 相同
    - `chunk_level`：INT，层级（1=摘要，2=##，3=###）
    - `source`、`doc_name`、`title`、`published_date`、`image_paths`：元数据

    参数：
    - **name**: collection 名称
    - **text_field**: 文本字段名
    - **vector_field**: 向量字段名
    - **dim**: 向量维度
    - **enable_metadata**: 是否启用元数据字段
    - **parent_child**: 是否启用父子分块模式
    """
    if utility.has_collection(name):
        logger.info("Milvus collection %s already exists.", name)
        return Collection(name)

    logger.info("Creating Milvus collection %s ...", name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]

    if parent_child:
        fields.extend([
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chunk_level", dtype=DataType.INT64),
            FieldSchema(name="chunk_category", dtype=DataType.VARCHAR, max_length=32),
        ])

    if enable_metadata:
        fields.extend([
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="doc_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="published_date", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="image_paths", dtype=DataType.VARCHAR, max_length=4096),
        ])

    schema = CollectionSchema(fields=fields, description="Metallurgy KB")
    collection = Collection(name=name, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024},
    }
    collection.create_index(field_name=vector_field, index_params=index_params)
    collection.load()

    logger.info("Collection %s created and loaded.", name)
    return collection


def _load_markdown_files(data_dir: str) -> List[dict]:
    """
    递归读取目录下所有 `full.md` 文件，返回结构化数据列表。

    每个元素包含：
    - text: 文件的完整文本内容（已删除参考文献章节）
    - source: 原始文件的完整路径
    - doc_name: 文档名称（从 md 标题或文件夹名提取）
    - title: 文章标题（从 md 第一行 # 标题提取）
    - published_date: 发表时间（从 【收稿日期】 提取）
    - image_paths: images/ 子文件夹下所有图片的绝对路径列表
    """
    root = Path(data_dir)
    if not root.exists():
        logger.warning("Data dir %s does not exist, nothing to insert.", root)
        return []

    records: List[dict] = []
    for path in root.rglob("full.md"):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        content = content.strip()
        if not content:
            continue

        # 删除参考文献章节
        ref_match = REFERENCE_SECTION_RE.search(content)
        if ref_match:
            content = content[:ref_match.start()]

        # 提取文章标题：从第一行 # 标题
        title_match = re.search(r'^#\s+(.+?)\s*$', content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else ""

        # 提取发表时间：从 【收稿日期】YYYY-MM-DD
        date_match = re.search(r'【收稿日期】\s*(\d{4}-\d{2}-\d{2})', content)
        published_date = date_match.group(1).strip() if date_match else ""

        # 收集 images/ 子文件夹下所有图片
        images_dir = path.parent / "images"
        image_paths: List[str] = []
        if images_dir.is_dir():
            for img in images_dir.iterdir():
                if img.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"):
                    image_paths.append(str(img.resolve()))

        doc_name = path.parent.name
        records.append({
            "text": content,
            "source": str(path.resolve()),
            "doc_name": doc_name,
            "title": title,
            "published_date": published_date,
            "image_paths": image_paths,
        })

    return records


def _load_texts_from_dir(data_dir: str) -> List[str]:
    """
    递归读取目录下所有 `.txt` 文件，返回其内容列表。

    行为：
    - 遍历 `data_dir` 下的 `*.txt`
    - UTF-8 读取，遇到非法字符忽略（errors="ignore"）
    - 去除首尾空白；空文本跳过

    返回：
    - `List[str]`：每个元素为一个文件的完整文本（当前不做 chunk 切分）
    """
    root = Path(data_dir)
    if not root.exists():
        logger.warning("Data dir %s does not exist, nothing to insert.", root)
        return []

    texts: List[str] = []
    for path in root.rglob("*.txt"):
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        txt = txt.strip()
        if not txt:
            continue
        texts.append(txt)
    return texts


def build_milvus_kb_from_local_files(
    data_dir: str | None = None,
    dim: int = 1024,  # text-embedding-v4 向量维度为 1024
    md_mode: bool = False,
    parent_child: bool = True,
    batch_size: int = 8,  # 减小 batch_size，避免单次请求超过 8192 token
    cleanup: bool = False,  # 插入前删除旧 collection
    md_file_batch: int = 10,  # 每 N 个 md 文件插入一次向量数据库
) -> None:
    """
    从本地目录读取文本，使用 embedding 编码后写入 Milvus（建库 + 入库一体）。

    支持两种模式：
    - **md_mode=False（默认）**：读取 `.txt` 文件，整篇入库
    - **md_mode=True**：读取 MinerU 导出的 `full.md` 文件

    父子分块策略（parent_child=True 时启用）：
    - 预处理：自动删除参考文献章节（# 参考文献 / # 【参考文献】）
    - 父块：每个 ## / ### 标题下的完整内容，嵌入完整章节文本
    - 子块类型：
      - paragraph（段落）：普通文本段落（以空行分隔）
      - formula（公式）：独立的 $$...$$ 块
      - table（表格）：独立的 <table>...</table> 块
    - 图片归��：图片跟在哪个子段落后面，就归哪个子段落所有
    - 检索链路：先命中子块 → 通过 parent_id 展开父块上下文

    参数：
    - **data_dir**: 文本目录
    - **dim**: 预设向量维度（自动覆盖）
    - **md_mode**: 是否启用 Markdown 模式
    - **parent_child**: 是否启用父子分块（仅 md_mode=True 时有效）
    - **batch_size**: 每次 embedding 请求的批大小
    - **md_file_batch**: 每 N 个 md 文件插入一次向量数据库（仅 md_mode=True 时有效）
    """
    import json

    config = load_config()

    kb_cfg = config["knowledge_base"]
    milvus_cfg = kb_cfg["milvus"]
    embedding_cfg = config.get("embedding", {})
    embedding_api_cfg = embedding_cfg.get("api", {})

    uri = milvus_cfg.get("uri") or None
    token = milvus_cfg.get("token") or None
    host = milvus_cfg.get("host") or None
    port = milvus_cfg.get("port")
    collection_name = milvus_cfg["collection"]
    text_field = milvus_cfg["text_field"]
    vector_field = milvus_cfg["vector_field"]

    _connect_milvus(uri=uri, token=token, host=host, port=port)

    # 插入前清理旧 collection（避免重复插入）
    if cleanup:
        from pymilvus import utility
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            logger.info("Dropped existing collection '%s'.", collection_name)

    model = LocalEmbeddingModel(
        model_path=embedding_cfg.get("model_path", ""),
        base_url=embedding_api_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        model=embedding_api_cfg.get("model", "text-embedding-v4"),
        api_key=embedding_api_cfg.get("api_key", ""),
        api_key_env=embedding_api_cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
    )

    dim_env = os.getenv("EMBEDDING_DIM")
    if dim_env is not None:
        try:
            dim = int(dim_env)
        except ValueError:
            logger.warning("Invalid EMBEDDING_DIM=%s, using dim=%d.", dim_env, dim)

    data_dir = data_dir or kb_cfg.get("data_dir", "data")
    root = Path(data_dir)
    if not root.exists():
        logger.error("Data dir %s does not exist.", data_dir)
        return

    if md_mode and parent_child:
        # ============================================================
        # 父子分块模式
        # ============================================================
        md_files = list(root.rglob("full.md"))
        if not md_files:
            logger.warning("No full.md files found in %s.", data_dir)
            return
        logger.info("Found %d full.md files, chunking in batches of %d ...", len(md_files), md_file_batch)

        # 先创建 collection（需要先知道 embedding dim，先用 sample 计算）
        collection_created = False
        collection = None
        actual_dim = dim

        for batch_start in range(0, len(md_files), md_file_batch):
            batch_end = min(batch_start + md_file_batch, len(md_files))
            batch_files = md_files[batch_start:batch_end]
            batch_num = batch_start // md_file_batch + 1
            total_batches = (len(md_files) + md_file_batch - 1) // md_file_batch
            logger.info("Processing md file batch [%d/%d] (%d files) ...", batch_num, total_batches, len(batch_files))

            all_chunks: List[dict] = []
            for md_path in batch_files:
                try:
                    result = _chunk_markdown(md_path)
                except Exception as e:
                    logger.warning("Failed to chunk %s: %s", md_path, e)
                    continue

                for parent in result.parents:
                    all_chunks.append({
                        "chunk_id": parent.chunk_id,
                        "chunk_type": "parent",
                        "parent_id": parent.chunk_id,
                        "chunk_level": parent.heading_level,
                        "chunk_category": "section",
                        "text": parent.full_text,
                        "source": parent.source,
                        "doc_name": parent.doc_name,
                        "title": parent.title,
                        "published_date": parent.published_date,
                        "image_paths": json.dumps([]),
                    })

                for child in result.children:
                    all_chunks.append({
                        "chunk_id": child.chunk_id,
                        "chunk_type": "child",
                        "parent_id": child.parent_id,
                        "chunk_level": 0,
                        "chunk_category": child.chunk_category,
                        "text": child.text,
                        "source": result.source,
                        "doc_name": result.doc_name,
                        "title": result.title,
                        "published_date": result.published_date,
                        "image_paths": json.dumps(child.image_paths),
                    })

            if not all_chunks:
                logger.warning("No chunks generated in batch [%d/%d].", batch_num, total_batches)
                continue

            logger.info("Batch [%d/%d]: %d chunks generated, computing embeddings ...", batch_num, total_batches, len(all_chunks))

            texts = [c["text"] for c in all_chunks]
            all_embeddings: List[List[float]] = []
            total_emb_batches = (len(texts) + batch_size - 1) // batch_size
            skip_indices: set[int] = set()

            for i in range(0, len(texts), batch_size):
                batch_idx = i // batch_size + 1
                batch = texts[i:i + batch_size]
                logger.info("Batch [%d/%d] embedding [%d/%d] (%d texts) ...", batch_num, total_batches, batch_idx, total_emb_batches, len(batch))

                try:
                    batch_embeddings = model.embed(batch)
                    all_embeddings.extend(batch_embeddings)
                except Exception as batch_err:
                    logger.warning("Batch [%d/%d] embedding [%d/%d] failed: %s, trying one by one ...", batch_num, total_batches, batch_idx, total_emb_batches, batch_err)
                    for j, text in enumerate(batch):
                        global_idx = i + j
                        if global_idx in skip_indices:
                            continue
                        try:
                            emb = model.embed([text])
                            all_embeddings.append(emb[0])
                        except Exception as single_err:
                            logger.warning("Skipping text [%d]: %s - %s", global_idx, str(single_err)[:80], text[:50])
                            skip_indices.add(global_idx)
                            all_embeddings.append([0.0] * 768)

                if i + batch_size < len(texts):
                    import time
                    time.sleep(1.5)

            if skip_indices:
                logger.warning("Batch [%d/%d]: Skipped %d texts due to API errors.", batch_num, total_batches, len(skip_indices))

            filtered_chunks = [c for idx, c in enumerate(all_chunks) if idx not in skip_indices]
            filtered_texts = [c["text"] for c in filtered_chunks]
            filtered_embeddings = [emb for idx, emb in enumerate(all_embeddings) if idx not in skip_indices]

            if not filtered_embeddings:
                logger.warning("Batch [%d/%d]: No valid embeddings after filtering.", batch_num, total_batches)
                continue

            actual_dim = len(filtered_embeddings[0])

            # 首次批次：创建 collection
            if not collection_created:
                collection = _create_collection_if_not_exists(
                    name=collection_name,
                    text_field=text_field,
                    vector_field=vector_field,
                    dim=actual_dim,
                    enable_metadata=True,
                    parent_child=True,
                )
                collection_created = True
                logger.info("Milvus collection created with dim=%d.", actual_dim)

            # 批量插入（分批，避免超过 Milvus 64MB 限制）
            insert_batch_size = 50
            chunk_ids = [c["chunk_id"] for c in filtered_chunks]
            chunk_types = [c["chunk_type"] for c in filtered_chunks]
            parent_ids = [c["parent_id"] for c in filtered_chunks]
            chunk_levels = [c["chunk_level"] for c in filtered_chunks]
            chunk_categories = [c["chunk_category"] for c in filtered_chunks]
            sources = [c["source"] for c in filtered_chunks]
            doc_names = [c["doc_name"] for c in filtered_chunks]
            titles = [c["title"] for c in filtered_chunks]
            published_dates = [c["published_date"] for c in filtered_chunks]
            image_paths_list = [c["image_paths"] for c in filtered_chunks]

            total_chunks = len(filtered_chunks)
            total_insert_batches = (total_chunks + insert_batch_size - 1) // insert_batch_size

            for i in range(0, total_chunks, insert_batch_size):
                insert_idx = i // insert_batch_size + 1
                batch_chunk_ids = chunk_ids[i:i + insert_batch_size]
                batch_texts = filtered_texts[i:i + insert_batch_size]
                batch_embeddings = filtered_embeddings[i:i + insert_batch_size]
                batch_types = chunk_types[i:i + insert_batch_size]
                batch_parents = parent_ids[i:i + insert_batch_size]
                batch_levels = chunk_levels[i:i + insert_batch_size]
                batch_categories = chunk_categories[i:i + insert_batch_size]
                batch_sources = sources[i:i + insert_batch_size]
                batch_doc_names = doc_names[i:i + insert_batch_size]
                batch_titles = titles[i:i + insert_batch_size]
                batch_dates = published_dates[i:i + insert_batch_size]
                batch_images = image_paths_list[i:i + insert_batch_size]

                batch_entities = [
                    batch_texts,
                    batch_embeddings,
                    batch_chunk_ids,
                    batch_types,
                    batch_parents,
                    batch_levels,
                    batch_categories,
                    batch_sources,
                    batch_doc_names,
                    batch_titles,
                    batch_dates,
                    batch_images,
                ]
                collection.insert(batch_entities)
                if insert_idx % 10 == 0 or insert_idx == total_insert_batches:
                    logger.info("Batch [%d/%d] insert [%d/%d]", batch_num, total_batches, insert_idx, total_insert_batches)

            collection.flush()
            logger.info("Batch [%d/%d] done: inserted %d chunks into %s.", batch_num, total_batches, total_chunks, collection_name)

        if collection:
            logger.info("All batches inserted: %d md files processed in %d batches.", len(md_files), total_batches)

    elif md_mode:
        # ============================================================
        # Markdown 整篇入库（无分块）
        # ============================================================
        records = _load_markdown_files(data_dir)
        if not records:
            logger.warning("No full.md files found in %s.", data_dir)
            return
        texts = [r["text"] for r in records]
        sources = [r["source"] for r in records]
        doc_names = [r["doc_name"] for r in records]
        titles = [r["title"] for r in records]
        published_dates = [r["published_date"] for r in records]
        image_paths_list = [json.dumps(r["image_paths"]) for r in records]
        logger.info("Loaded %d markdown files, generating embeddings ...", len(texts))

        embeddings = model.embed(texts)
        dim = len(embeddings[0])

        collection = _create_collection_if_not_exists(
            name=collection_name,
            text_field=text_field,
            vector_field=vector_field,
            dim=dim,
            enable_metadata=True,
            parent_child=False,
        )

        # 分批插入（避免超过 Milvus 64MB 限制）
        insert_batch_size = 50
        total_records = len(texts)
        total_batches = (total_records + insert_batch_size - 1) // insert_batch_size
        logger.info("Inserting %d records in batches of %d ...", total_records, insert_batch_size)

        for i in range(0, total_records, insert_batch_size):
            batch_texts = texts[i:i + insert_batch_size]
            batch_emb = embeddings[i:i + insert_batch_size]
            batch_entities = [batch_texts, batch_emb]
            if enable_metadata:
                batch_entities.extend([
                    sources[i:i + insert_batch_size],
                    doc_names[i:i + insert_batch_size],
                    titles[i:i + insert_batch_size],
                    published_dates[i:i + insert_batch_size],
                    image_paths_list[i:i + insert_batch_size],
                ])
            collection.insert(batch_entities)
            if (i // insert_batch_size + 1) % 10 == 0 or i + insert_batch_size >= total_records:
                logger.info("Inserted batch [%d/%d]", i // insert_batch_size + 1, total_batches)

        collection.flush()
        logger.info("Inserted %d records into %s.", total_records, collection_name)

    else:
        # ============================================================
        # .txt 模式（原有逻辑）
        # ============================================================
        texts = _load_texts_from_dir(data_dir)
        if not texts:
            logger.warning("No .txt files found in %s.", data_dir)
            return
        logger.info("Loaded %d texts, generating embeddings ...", len(texts))

        embeddings = model.embed(texts)
        dim = len(embeddings[0])

        collection = _create_collection_if_not_exists(
            name=collection_name,
            text_field=text_field,
            vector_field=vector_field,
            dim=dim,
            enable_metadata=False,
            parent_child=False,
        )

        insert_batch_size = 50
        total = len(texts)
        for i in range(0, total, insert_batch_size):
            collection.insert([texts[i:i + insert_batch_size], embeddings[i:i + insert_batch_size]])
        collection.flush()
        logger.info("Inserted %d records into %s.", total, collection_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Milvus knowledge base from local files.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory containing .txt or full.md files.",
    )
    parser.add_argument(
        "--md-mode",
        action="store_true",
        help="Enable Markdown mode: read MinerU full.md files.",
    )
    parser.add_argument(
        "--no-parent-child",
        action="store_true",
        help="Disable parent-child chunking (only effective with --md-mode). "
             "Without this flag, parent-child chunking is enabled by default.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Drop existing collection before inserting (avoids duplicate data).",
    )
    parser.add_argument(
        "--md-file-batch",
        type=int,
        default=10,
        help="Number of md files to process before inserting into Milvus (default: 10).",
    )
    args = parser.parse_args()

    build_milvus_kb_from_local_files(
        data_dir=args.data_dir,
        md_mode=args.md_mode,
        parent_child=not args.no_parent_child,
        cleanup=args.cleanup,
        md_file_batch=args.md_file_batch,
    )

