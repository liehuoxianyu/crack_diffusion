# web_demo (CrackTree Web Demo)

本仓库的实际前后端为：
- 前端：`web_demo/`（Next.js 13, App Router）
- 后端：`api/`（FastAPI，提供 `POST /generate`）

`webui/` 仅是参考 UI，不作为运行入口。

---

## 启动方式

### 1) 启动后端（推荐从仓库根目录）

```bash
cd /work
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

验证：打开 `http://localhost:8000/docs`。

说明：如果在 `api/` 目录直接执行 `uvicorn server:app`，可能出现 `ModuleNotFoundError: No module named 'api'`。

### 2) 启动前端

```bash
cd /work/web_demo
npm run dev
```

访问：`http://localhost:3000`

### 3) 配置 API 地址（可选）

前端通过 `NEXT_PUBLIC_API_URL` 读取后端地址，默认 `http://localhost:8000`。

示例：

```bash
NEXT_PUBLIC_API_URL="http://localhost:8000" npm run dev
```

### 4) 配置 CORS（可选）

后端通过 `WEB_DEMO_CORS_ALLOW_ORIGINS` 控制允许来源，默认：

- `http://localhost:3000`
- `http://127.0.0.1:3000`

示例：

```bash
WEB_DEMO_CORS_ALLOW_ORIGINS="http://localhost:3000,http://127.0.0.1:3000" uvicorn api.server:app --host 0.0.0.0 --port 8000
```

---

## 页面交互说明

- 左侧 `ControlNet` / `LoRA` 点击后会切换开关，并自动展开右侧 `Advanced Settings`。
- `Advanced Settings` 支持展开/折叠，并带有 `Expanded` / `Collapsed` 状态提示。
- `Generate` 会显示请求阶段状态，失败时给出更可读错误信息（连接失败、超时、后端错误等）。

---

## 故障排查

### 1) 点击 Generate 后“没反应”

先打开浏览器开发者工具 `Network`，确认是否发出了 `POST ${NEXT_PUBLIC_API_URL}/generate` 请求。

- **没有请求发出**：通常是前端校验未通过。  
  例如：开启 `ControlNet` 但没有上传 `condition_image`。
- **请求发出但失败**：查看状态码和响应内容。  
  常见原因：API 地址错误、后端未启动、模型或 LoRA 路径不存在。
- **请求长时间不返回**：可能是后端推理耗时过长或服务不可达。前端会显示超时/连接失败提示。

### 2) 点击 ControlNet / LoRA 看起来没生效

- 观察左侧卡片的 ON/OFF 状态是否变化。
- 右侧 `Advanced Settings` 会自动展开，可在对应 section 内看到 Enable 开关同步变化。

### 3) Advanced Settings 无法展开

- 检查按钮右侧是否从 `Collapsed` 变为 `Expanded`。
- 若样式异常，先清理 Next 缓存并重启前端：

```bash
cd /work/web_demo
rm -rf .next
npm run dev
```

---

## 主要文件

- `web_demo/src/app/page.tsx`
- `web_demo/src/app/page.module.css`
- `web_demo/src/app/layout.tsx`
- `web_demo/src/app/globals.css`
- `api/server.py`
