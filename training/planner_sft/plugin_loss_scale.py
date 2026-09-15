# -*- coding: utf-8 -*-
"""
ms-swift 外部插件：结构 Token / 领域 Token 损失加权 + 归一化加权交叉熵。

加载：--external_plugins training/planner_sft/plugin_loss_scale.py
  M1 普通 LoRA-SFT   ：--loss_scale default                                            （不需本插件）
  M2 结构 Token 加权  ：--loss_scale planner_struct --loss_type normalized_weighted_ce
  M3 结构 + 领域加权  ：--loss_scale planner_domain --loss_type normalized_weighted_ce

设计：
  * 权重 0（system / user / tool 返回）由 ms-swift 的 ContextType 机制处理：非 RESPONSE 上下文 loss_scale=0，
    Agent 训练中的 tool_response 也不计损失；本插件只切分 assistant 响应。
  * 响应文本由 weighting.split_weighted 切成片段：普通文本 1 / 结构 2 / 领域 3。
  * normalized_weighted_ce：L = Σ w_i·CE_i / Σ w_i，替代默认按 token 数归一的实现，
    使不同加权方案下 loss 量级一致（PLAN P5-2「归一化加权交叉熵」）。

兼容性：同时尝试 ms-swift 4.x（swift.loss_scale）与 3.x（swift.plugin.loss_scale）导入路径；
        get_loss_scale 用 *args/**kwargs 兼容两代签名（3.x: context, context_type, is_last_round；4.x: context）。
本文件在无 torch / swift 的本地环境导入不报错，仅在 swift 存在时完成注册（便于本地 dry-run）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from weighting import W_DOMAIN, W_STRUCT, W_TEXT, load_domain_terms, split_weighted  # noqa: E402

_DOMAIN_TERMS = load_domain_terms(Path(os.environ.get("PLANNER_DOMAIN_TERMS", HERE / "domain_terms.json")))
_STRUCT_W = float(os.environ.get("PLANNER_STRUCT_WEIGHT", W_STRUCT))
_DOMAIN_W = float(os.environ.get("PLANNER_DOMAIN_WEIGHT", W_DOMAIN))
print(f"[planner_sft] plugin loaded: struct={_STRUCT_W} domain={_DOMAIN_W} terms={len(_DOMAIN_TERMS)}", file=sys.stderr)

# ── 1. loss_scale 注册 ─────────────────────────────────────────────────────────
LossScale = None
loss_scale_map = None
_API = None
try:  # ms-swift >= 4.x
    from swift.loss_scale.base import LossScale  # type: ignore
    from swift.loss_scale.mapping import loss_scale_map  # type: ignore
    _API = "v4"
except Exception:  # noqa: BLE001
    try:  # ms-swift 3.x
        from swift.plugin.loss_scale.loss_scale import LossScale, loss_scale_map  # type: ignore
        _API = "v3"
    except Exception:  # noqa: BLE001
        pass

if LossScale is not None:
    ContextType = None
    for _mod in ("swift.llm.template.utils", "swift.template.utils"):
        try:
            ContextType = __import__(_mod, fromlist=["ContextType"]).ContextType  # type: ignore
            break
        except Exception:  # noqa: BLE001
            continue

    def _is_response(context_type) -> bool:
        if context_type is None:
            return True  # 4.x 仅对响应片段调用 get_loss_scale
        if ContextType is not None:
            return context_type == ContextType.RESPONSE
        return str(context_type).lower().endswith("response")

    class PlannerStructLossScale(LossScale):  # type: ignore[misc]
        """M2：结构 token 2，其余响应 token 1。"""
        struct_weight = _STRUCT_W
        domain_weight = W_TEXT
        domain_terms: list = []

        def get_loss_scale(self, context, *args, **kwargs):  # type: ignore[override]
            context_type = args[0] if args else kwargs.get("context_type")
            if isinstance(context, str) and _is_response(context_type):
                return split_weighted(context, struct_weight=self.struct_weight,
                                      domain_weight=self.domain_weight, domain_terms=self.domain_terms)
            return super().get_loss_scale(context, *args, **kwargs)

    class PlannerDomainLossScale(PlannerStructLossScale):
        """M3：结构 token 2 + 领域关键标识 3。"""
        domain_weight = _DOMAIN_W
        domain_terms = _DOMAIN_TERMS

    if _API == "v4":
        loss_scale_map["planner_struct"] = PlannerStructLossScale
        loss_scale_map["planner_domain"] = PlannerDomainLossScale
    else:
        loss_scale_map["planner_struct"] = PlannerStructLossScale()
        loss_scale_map["planner_domain"] = PlannerDomainLossScale()
    print(f"[planner_sft] registered loss_scale: planner_struct, planner_domain (api={_API})", file=sys.stderr)

# ── 2. 归一化加权交叉熵 ────────────────────────────────────────────────────────
try:
    import torch  # type: ignore  # noqa: F401
    from torch.nn import functional as F  # type: ignore
    try:
        from swift.plugin.loss import register_loss_func  # type: ignore
    except Exception:  # noqa: BLE001
        from swift.loss import register_loss_func  # type: ignore

    @register_loss_func("normalized_weighted_ce")
    def normalized_weighted_ce(outputs, labels, loss_scale=None, num_items_in_batch=None):  # type: ignore
        """L = Σ w_i·CE_i / Σ w_i；loss_scale 为空时退化为普通平均 CE。"""
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                             ignore_index=-100, reduction="none")
        mask = (shift_labels.view(-1) != -100).to(ce.dtype)
        w = mask if loss_scale is None else loss_scale[..., 1:].contiguous().view(-1).to(ce.dtype) * mask
        denom = w.sum()
        if float(denom) == 0.0:
            return ce.sum() * 0.0
        return (ce * w).sum() / denom

    print("[planner_sft] registered loss_type: normalized_weighted_ce", file=sys.stderr)
except Exception as exc:  # noqa: BLE001
    if LossScale is not None:
        print(f"[planner_sft] loss_func 未注册（{exc}），退回 ms-swift 默认加权 CE", file=sys.stderr)
