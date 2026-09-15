# P1-1 内容单元清洗报告

- 输入文献目录：182（失败 0）
- 文献类型：期刊/短文 164，学位论文/长文 18
- 输出单元总数：15849

## 单元类型分布

| 类型 | 数量 |
|---|---:|
| code | 3 |
| equation | 848 |
| footnote | 59 |
| image | 956 |
| list | 1031 |
| paragraph | 8395 |
| table | 627 |
| title | 3930 |

## 处理统计

| 项 | 数量 |
|---|---:|
| dropped_noise | 5303 |
| dropped_empty | 46 |
| merged_cross_page | 418 |
| table_caption_inferred | 2 |
| table_caption_missing | 38 |
| image_caption_inferred | 2 |
| image_caption_missing | 73 |
| equation_units | 848 |

## 说明

- 标题层级由编号模式重建（level 0 = 文献主标题）。
- 表格 caption 由相邻「表N」段落推断，`caption_inferred=true` 标记。
- 公式单元合并了引出段落与「式中」解释段落，原 LaTeX 保存在 `latex` 列表中。
- 图片当前仅有 caption 与路径，图像描述生成留待 P1-2（需多模态模型）。