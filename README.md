# CrackTree Diffusion Workspace

这个仓库用于 **CrackTree 路面裂缝生成实验**，同时包含：
- 训练与评测脚本（ControlNet / LoRA / SD）
- 一个可运行的 Web Demo（FastAPI + Next.js）
- FreeControl 适配层与上游代码副本（用于后续集成）

本 README 目标是让后续接手的 Agent 在 3-5 分钟内建立全局理解并开始改动。

## 1. 仓库总览（先看这里）

顶层关键路径：
- `api/`：FastAPI 后端与推理核心（Web Demo 实际后端）
- `web_demo/`：Next.js 前端（Web Demo 实际前端）
- `eval_all.py`：统一评测脚本（SD / SD+LoRA / SD+ControlNet / SD+LoRA+ControlNet）
- `run_binary.sh`：训练 Binary 条件 ControlNet
- `run_dt.sh`：训练 DT 条件 ControlNet
- `run_lora.sh`：训练 LoRA（真实感增强）
- `write_experiment_cards.py`：为训练输出目录写 `experiment_card.md`
- `cracktree_dataset/`：自定义 HF Dataset 脚本
- `docs/freecontrol_integration.md`：FreeControl 集成现状说明
- `free_control/freecontrol/`：FreeControl 上游风格代码（含单独 README 与环境定义）
- `diffusers/`：diffusers 源码与 examples（本项目训练脚本依赖其 examples）
- `outputs/`：训练、评测、Web 产物默认输出目录

建议先忽略：
- `.venvs/`：本地虚拟环境内容（体积大、搜索噪声高）
- `outputs/`：实验产物目录（非源码）

## 2. 当前架构关系

- **Web 路径**：`web_demo` -> `POST /generate` -> `api/server.py` -> `api/infer.py`
- **实验路径**：`run_*.sh` 训练 -> 产物到 `outputs/exp_*` -> `eval_all.py` 统一评测 -> `write_experiment_cards.py` 写卡片
- **FreeControl 路径（当前阶段）**：`api/freecontrol/*` 可被 Python 调用，但默认未接入 `/generate` HTTP 流程

## 3. 快速启动（Web Demo）

### 后端
在仓库根目录执行：

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

默认访问：
- OpenAPI: `http://localhost:8000/docs`
- API: `POST /generate`

### 前端

```bash
cd web_demo
npm run dev
```

默认访问：`http://localhost:3000`

### 常用环境变量
- `NEXT_PUBLIC_API_URL`：前端指向后端地址（默认 `http://localhost:8000`）
- `WEB_DEMO_CORS_ALLOW_ORIGINS`：后端 CORS 白名单
- `WEB_DEMO_OUTPUT_DIR`：Web 生成图片输出目录

## 4. 推理与评测入口

### Web 推理核心
- `api/infer.py`
  - 模型：`SD_BASE_MODEL`（默认 `runwayml/stable-diffusion-v1-5`）
  - ControlNet 目录：
    - `CONTROLNET_BINARY_BASE_DIR`
    - `CONTROLNET_DT_BASE_DIR`
  - checkpoint 选择：`CONTROLNET_STEP`（默认 2000）
  - 缓存键：
    - `sd::...`（纯 SD / SD+LoRA）
    - `cn::...`（ControlNet 路径）

### 批量评测
- `eval_all.py`（配置集中在文件顶部）
  - 默认使用外部数据路径（如 `/CrackTree260/...`）
  - 默认输出：`/work/outputs/exp_eval_all`
  - 四种模式一次跑齐，输出按 `id` 和模式分层保存
- `evaluation/run_eval_metrics.sh`（统一指标入口）
  - `--profile fairness`：扩散公平性评测（含 topology）
  - `--profile dt_family`：DT family 评测指标
  - `--profile baseline --baseline <pix2pix|cyclegan|vqgan>`：GAN baseline 指标

## 5. 训练入口

- `run_binary.sh`：Binary 条件图 ControlNet 训练
- `run_dt.sh`：DT 条件图 ControlNet 训练
- `run_lora.sh`：LoRA 训练（UNet，偏真实感增强）

三个脚本共同特点：
- 均依赖 `diffusers/examples/*` 下的训练脚本
- 均使用 `accelerate launch ...`
- 均假设外部数据目录存在（`/CrackTree260/...`）

## 6. FreeControl 现状

参考 `docs/freecontrol_integration.md`：
- 已新增 `api/freecontrol` 适配层（请求类型、配置、缓存、调用封装）
- 当前是“加法集成”：不影响 `api/infer.py` 原有 SD/ControlNet/LoRA 路径
- 目前没有新增 FreeControl HTTP endpoint
- 烟雾测试脚本：
  - `scripts/smoke_freecontrol.py`（I2I）
  - `scripts/smoke_freecontrol_t2i.py`（T2I）

## 7. 对新 Agent 的改动建议

改动前优先定位：
1. Web 接口行为：`api/server.py`, `api/infer.py`, `web_demo/src/app/page.tsx`
2. 训练协议：`run_*.sh`, `cracktree_dataset/*.py`
3. 评测协议：`eval_all.py`, `write_experiment_cards.py`
4. FreeControl：`api/freecontrol/*` 与 `docs/freecontrol_integration.md`

搜索建议（减少噪声）：
- 优先在 `api/`, `web_demo/`, 根目录脚本中搜索
- 尽量避免全仓库直接搜索 `diffusers/` 和 `.venvs/`

## 8. 已知坑点 / 风险

- 许多默认路径是**绝对路径**（如 `/CrackTree260/...`），新环境通常需要改路径或挂载数据
- `free_control/freecontrol/` 与 `diffusers/` 体积大，且可能包含上游代码副本，改动前确认影响面
- 根目录没有统一测试入口；请按改动范围做最小可复现实验（API 调用、脚本试跑、前端手动验证）

## 9. 常用命令速查

```bash
# 后端
uvicorn api.server:app --host 0.0.0.0 --port 8000

# 前端
cd web_demo && npm run dev

# 训练（示例）
bash run_binary.sh
bash run_dt.sh
bash run_lora.sh

# 批量评测
python eval_all.py

# 写实验卡片
python write_experiment_cards.py
```

---

如果你是后续接手 Agent，建议第一步先确认：
1) 当前机器是否可用 GPU；2) `/CrackTree260` 数据是否存在；3) `outputs/exp_*` 下是否已有可复用 checkpoint。
