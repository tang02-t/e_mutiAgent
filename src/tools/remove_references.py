"""
去除指定目录下所有 .md 文件中 "参考文献" 及其之后的内容。
"""

import os
from pathlib import Path


def remove_references_section(file_path: Path) -> tuple[bool, int]:
    """
    去除单个 md 文件中的参考文献部分。
    返回 (是否修改, 去除的字符数)。
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            content = file_path.read_text(encoding="gbk")
        except Exception:
            print(f"  [跳过] 无法读取文件: {file_path}")
            return False, 0

    # 匹配 "## 参考文献" 或 "# 参考文献" 等变体（忽略大小写）
    marker_lower = "参考文献"
    content_lower = content.lower()

    # 查找 ## 参考文献 或 # 参考文献（## 或 ### 等多级标题）
    import re
    # 匹配任意数量 # 后跟空格和 "参考文献"
    match = re.search(r"(#{1,6}\s*参考文献)", content)

    if not match:
        # 也尝试简单字符串查找（忽略大小写）
        idx = content_lower.rfind(marker_lower)
        if idx == -1:
            return False, 0
        # 确认上下文是标题格式（前面是 # 或换行）
        before = content[:idx].rstrip()
        if before and not before.endswith("#") and not before.endswith("\n"):
            # 不是标题格式，可能是正文中的词，跳过
            return False, 0
        cut_idx = idx
    else:
        cut_idx = match.start()

    before_text = content[:cut_idx].rstrip()
    if not before_text:
        print(f"  [跳过] 文件内容已为空或无实质内容: {file_path.name}")
        return False, 0

    removed_chars = len(content) - len(before_text)
    file_path.write_text(before_text, encoding="utf-8")
    return True, removed_chars


def process_directory(dir_path: str):
    dir_path = Path(dir_path)

    if not dir_path.exists():
        print(f"目录不存在: {dir_path}")
        return

    md_files = list(dir_path.glob("*.md"))
    if not md_files:
        print(f"目录中没有找到 .md 文件: {dir_path}")
        return

    print(f"找到 {len(md_files)} 个 .md 文件\n")

    total_modified = 0
    total_removed = 0

    for f in md_files:
        modified, removed = remove_references_section(f)
        if modified:
            total_modified += 1
            total_removed += removed
            print(f"  [已修改] {f.name} (去除 {removed} 字符)")
        else:
            print(f"  [无需修改] {f.name}")

    print(f"\n处理完成: 共 {total_modified}/{len(md_files)} 个文件被修改，共去除 {total_removed} 字符")


if __name__ == "__main__":
    process_directory(r"E:\workspace\code\multi_Agent\data\fast_md")
