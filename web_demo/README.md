# web_demo (CrackTree Web Demo)

本仓库的**实际前端/后端**在：
- **前端**：`web_demo/`（Next.js 13, App Router）
- **后端**：`api/`（FastAPI，提供 `POST /generate`）

`webui/` 是参考 UI（用于对齐设计语言），不作为实际运行入口。

---

## 启动方式

### 1) 启动后端（推荐从仓库根目录启动）

```bash
cd /work
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

验证：打开 `http://localhost:8000/docs`。

> 说明：如果在 `api/` 目录下用 `uvicorn server:app` 启动，可能出现 `ModuleNotFoundError: No module named 'api'`。\n+

### 2) 启动前端

```bash
cd /work/web_demo
npm run dev
```

访问：`http://localhost:3000`

### 3) 配置后端地址（可选）

前端通过环境变量读取 API 地址：
- `NEXT_PUBLIC_API_URL`（默认：`http://localhost:8000`）

示例：

```bash
NEXT_PUBLIC_API_URL="http://localhost:8000" npm run dev
```

---

## 页面结构（按本次对话改造结果）

前端主逻辑在 `web_demo/src/app/page.tsx`，样式在 `web_demo/src/app/page.module.css`，全局变量在 `web_demo/src/app/globals.css`。

### 三栏布局
- **左侧 Modules**：\n+  - `Stable Diffusion`（常亮）\n+  - `ControlNet` / `LoRA`（点击联动右侧 Advanced Settings 的 enable 开关）
- **中间 Result Preview**：结果预览 + 下载按钮 + 配置摘要 + 历史缩略图
- **右侧 Parameter**：基础参数表单 + `Advanced Settings`（可折叠）

### Advanced Settings
- 结构重构为两块：`ControlNet` 与 `LoRA`\n+  - `Enable` 使用 iOS 风格滑动开关\n+  - 已移除额外的 `ON/OFF` 状态标签\n+  - ControlNet 启用时要求上传 `condition_image`

---

## 交互与逻辑优化（本次对话）

- **接口字段保持不变**：仍使用 `FormData` 发送到 `POST /generate`。\n+  - `prompt`, `negative_prompt`, `seed`, `num_inference_steps`, `guidance_scale`, `width`, `height`\n+  - `enable_controlnet`, `controlnet_type`, `controlnet_conditioning_scale`, `condition_image`\n+  - `enable_lora`, `lora_path`, `lora_scale`
- **前端输入保护**：生成前对 steps / guidance / width / height / scale 做轻量归一化与边界保护（不改变后端字段）。\n+  - width/height 会按 8 对齐\n+  - scale 做合理 clamp
- **状态记忆**：使用 `localStorage` 记住主要参数、模块开关和 Advanced 折叠状态，刷新后自动恢复（不包含上传文件本身）。\n+  - key：`cracktree-webdemo-ui-state-v1`

---

## 资源与头部

- 头部采用参考 UI 的风格，仅保留 logo + 标题，不添加 tools/login 等功能区。\n+  - `web_demo/public/aistudio.svg`

---

## 已知问题 / 排错

### 1) 刷新后样式“像没生效”
我们遇到过 dev 环境缓存/多进程导致样式没更新的问题。\n+
处理方式：确保只跑一个前端进程，并清理 Next 缓存后重启：

```bash
cd /work/web_demo
rm -rf .next
npm run dev
```

### 2) `npm run lint` 报错 `Failed to load config "next/typescript"`
这是项目当前 `.eslintrc.json` 配置导致的（不影响 `npm run build` 产物）。\n+

---

## 主要文件
- `web_demo/src/app/page.tsx`\n+- `web_demo/src/app/page.module.css`\n+- `web_demo/src/app/layout.tsx`\n+- `web_demo/src/app/globals.css`\n+- `api/server.py`
