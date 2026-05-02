## SSH执行规则

通过 SSH 在远程节点执行命令时，在./tmp中创建想要运行的命令的sh脚本，再用ssh bash运行，可以在sh脚本内cd到项目根目录；
因为当前“SSH 远程执行命令” 有bug，会反复尝试pwd但pwd得不到继承

## 文档目录约定：`logs/` vs `my-docs/`

两个目录服务的读者不同，落盘前先想清楚谁会读它，再决定放哪。

### `logs/` — agent 运行时记录（读者：agent 自己 / 后续 session）
- 内容：调查过程中的 finding、profiling 数据、failure root cause、bug fix 记录、临时 progress 笔记、日志文件等。
- 命名：`findings_<topic>.md`、`progress_<topic>.md`、`<task>.log`，无日期前缀。
- 文风：粗糙可，重事实+引用，可能很长且不一定收尾，是 agent 自我笔记的延伸。
- 用途：下次 agent session 接手时快速 catch-up，避免重复探索。
- 不要：写成给人看的 polished 总结；不要放阶段性交付物。

### `my-docs/` — 阶段性人读文档（读者：用户本人 / 合作者 review）
- 内容：阶段总结、设计 review、roadmap、方法论文档、行动指南、向合作者交付的分析报告等。
- 命名：`YYYYMMDD-<topic>.md`，必须有日期前缀（与既有约定一致，见目录里其他文件）。
- 文风：组织清晰、有 TL;DR、结构化分节、有动机/方法/结论/行动项；写作像在交付一份"成果文档"。
- 用途：用户回顾一个阶段的工作；交给协作者做 code review、方向 review、批准下一阶段计划。
- 不要：把它当成 scratch 笔记；不要把粗糙未整理的材料直接落到这里。

### 简单判断
- "这份内容三个月后我（用户）需要拿出来给合作者看" → `my-docs/`
- "这份内容下次 agent 接手时需要读，但用户不会主动看" → `logs/`
- 两者都需要：先在 `logs/` 写过程笔记，最后在 `my-docs/` 写一份精炼总结，互相引用。