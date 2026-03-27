"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ControlNetType = "DT" | "Binary";

type HistoryItem = {
  id: string;
  createdAt: number;
  imageBase64: string | null;
  imagePath: string | null;
  elapsedMs: number | null;
  configSummary: string | null;
  mode: string | null;
};

function snapTo8(v: number) {
  const n = Math.max(8, Math.floor(v));
  return n - (n % 8);
}

function buildModeTag(enableControlnet: boolean, enableLora: boolean, cnType: string) {
  const cn = (cnType || "").trim().toLowerCase();
  if (enableControlnet && enableLora) return `controlnet+lora`;
  if (enableControlnet) return `controlnet+${cn || "dt"}`;
  if (enableLora) return `lora`;
  return `sd`;
}

export default function Page() {
  const API_URL = useMemo(
    () => process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000",
    []
  );

  // ---- Left panel: inputs ----
  const [prompt, setPrompt] = useState(
    "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting"
  );
  const [negativePrompt, setNegativePrompt] = useState("");
  const [seed, setSeed] = useState<number>(42);

  // Advanced controls (kept but moved into collapsible group)
  const [numInferenceSteps, setNumInferenceSteps] = useState<number>(25);
  const [guidanceScale, setGuidanceScale] = useState<number>(7.5);
  const [width, setWidth] = useState<number>(512);
  const [height, setHeight] = useState<number>(512);

  // ControlNet
  const [enableControlnet, setEnableControlnet] = useState<boolean>(false);
  const [controlnetType, setControlnetType] = useState<ControlNetType>("DT");
  const [controlnetConditioningScale, setControlnetConditioningScale] =
    useState<number>(1.0);
  const [conditionImageFile, setConditionImageFile] = useState<File | null>(
    null
  );
  const [conditionPreviewUrl, setConditionPreviewUrl] = useState<string | null>(null);

  // LoRA
  const [enableLora, setEnableLora] = useState<boolean>(false);
  const [loraPath, setLoraPath] = useState<string>(
    "/work/outputs/exp_lora_realism/pytorch_lora_weights.safetensors"
  );
  const [loraScale, setLoraScale] = useState<number>(0.7);

  // ---- UI states ----
  const [showAdvanced, setShowAdvanced] = useState<boolean>(true);

  // ---- Right panel: outputs ----
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [configSummary, setConfigSummary] = useState<string | null>(null);
  const [modeTag, setModeTag] = useState<string>(() =>
    buildModeTag(false, false, "DT")
  );

  const [history, setHistory] = useState<HistoryItem[]>([]);

  const toastTimerRef = useRef<number | null>(null);
  const imageDownloadUrl = useMemo(() => {
    if (!imageBase64) return null;
    return `data:image/png;base64,${imageBase64}`;
  }, [imageBase64]);

  useEffect(() => {
    const tag = buildModeTag(enableControlnet, enableLora, controlnetType);
    setModeTag(tag);
  }, [enableControlnet, enableLora, controlnetType]);

  useEffect(() => {
    if (!conditionImageFile) {
      setConditionPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(conditionImageFile);
    setConditionPreviewUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [conditionImageFile]);

  const reset = () => {
    setPrompt(
      "a photo of pavement crack, realistic texture, high detail, natural shadowing, realistic lighting"
    );
    setNegativePrompt("");
    setSeed(42);
    setNumInferenceSteps(25);
    setGuidanceScale(7.5);
    setWidth(512);
    setHeight(512);
    setEnableControlnet(false);
    setControlnetType("DT");
    setControlnetConditioningScale(1.0);
    setConditionImageFile(null);
    setConditionPreviewUrl(null);
    setEnableLora(false);
    setLoraPath("/work/outputs/exp_lora_realism/pytorch_lora_weights.safetensors");
    setLoraScale(0.7);

    setLoading(false);
    setError(null);
    setImageBase64(null);
    setImagePath(null);
    setElapsedMs(null);
    setConfigSummary(null);
  };

  const showError = (msg: string) => {
    setError(msg);
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => setError(null), 4200);
  };

  const handleConditionFile = (f: File | null) => {
    setConditionImageFile(f);
  };

  const onGenerate = async () => {
    setLoading(true);
    setError(null);
    try {
      if (enableControlnet && !conditionImageFile) {
        throw new Error("enable_controlnet=true 时需要上传 condition_image。");
      }

      const form = new FormData();
      form.append("prompt", prompt);
      form.append("negative_prompt", negativePrompt);
      form.append("seed", String(seed));
      form.append("num_inference_steps", String(numInferenceSteps));
      form.append("guidance_scale", String(guidanceScale));
      form.append("width", String(width));
      form.append("height", String(height));

      form.append("enable_controlnet", enableControlnet ? "true" : "false");
      form.append("controlnet_type", controlnetType === "DT" ? "DT" : "Binary");
      form.append("controlnet_conditioning_scale", String(controlnetConditioningScale));

      form.append("enable_lora", enableLora ? "true" : "false");
      form.append("lora_path", enableLora ? loraPath : "");
      form.append("lora_scale", String(loraScale));

      if (enableControlnet) {
        form.append("condition_image", conditionImageFile as File);
      }

      const resp = await fetch(`${API_URL}/generate`, {
        method: "POST",
        body: form,
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data?.detail ?? "Request failed");
      }

      setImageBase64(data.image_base64);
      setImagePath(data.image_path);
      setElapsedMs(data.elapsed_ms);
      setConfigSummary(data.config_summary);

      const item: HistoryItem = {
        id: `${Date.now()}_${Math.random().toString(16).slice(2)}`,
        createdAt: Date.now(),
        imageBase64: data.image_base64 ?? null,
        imagePath: data.image_path ?? null,
        elapsedMs: data.elapsed_ms ?? null,
        configSummary: data.config_summary ?? null,
        mode: data.mode ?? null,
      };
      setHistory((prev) => [item, ...prev].slice(0, 12));
    } catch (e: any) {
      showError(e?.message ? String(e.message) : String(e));
    } finally {
      setLoading(false);
    }
  };

  const onPickHistory = (item: HistoryItem) => {
    setImageBase64(item.imageBase64);
    setImagePath(item.imagePath);
    setElapsedMs(item.elapsedMs);
    setConfigSummary(item.configSummary);
    setError(null);
  };

  const onDownload = () => {
    if (!imageDownloadUrl) return;
    const a = document.createElement("a");
    a.href = imageDownloadUrl;
    a.download = `generated_${Date.now()}.png`;
    a.click();
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="p-4 md:p-6 max-w-[1400px] mx-auto">
        <div className="flex items-center justify-between gap-4 mb-4">
          <h1 className="text-lg md:text-xl font-semibold">CrackTree Web Demo</h1>
          <div className="text-sm text-zinc-400">
            API: <span className="text-zinc-200">{API_URL}</span>
          </div>
        </div>

        <div className="flex gap-4 md:gap-6">
          {/* Left panel */}
          <div className="w-full md:w-[420px] shrink-0">
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
              <div className="space-y-4">
                <div>
                  <div className="text-sm text-zinc-300 mb-1">prompt</div>
                  <textarea
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                    rows={4}
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                  />
                </div>

                <div>
                  <div className="text-sm text-zinc-300 mb-1">negative prompt</div>
                  <textarea
                    className="w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                    rows={3}
                    value={negativePrompt}
                    onChange={(e) => setNegativePrompt(e.target.value)}
                  />
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <label className="text-sm text-zinc-300">
                    seed
                    <input
                      className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                    />
                  </label>
                  <label className="text-sm text-zinc-300">
                    steps
                    <input
                      className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                      type="number"
                      value={numInferenceSteps}
                      onChange={(e) => setNumInferenceSteps(Number(e.target.value))}
                    />
                  </label>

                  <label className="text-sm text-zinc-300">
                    guidance_scale
                    <input
                      className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                      type="number"
                      step={0.1}
                      value={guidanceScale}
                      onChange={(e) => setGuidanceScale(Number(e.target.value))}
                    />
                  </label>

                  <div className="flex items-end gap-2">
                    <div className="w-full">
                      <label className="text-sm text-zinc-300 block">
                        width / height
                      </label>
                      <div className="grid grid-cols-2 gap-2 mt-1">
                        <input
                          className="rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                          type="number"
                          step={8}
                          value={width}
                          onChange={(e) => {
                            const v = Math.max(8, Number(e.target.value));
                            setWidth(v - (v % 8));
                          }}
                        />
                        <input
                          className="rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                          type="number"
                          step={8}
                          value={height}
                          onChange={(e) => {
                            const v = Math.max(8, Number(e.target.value));
                            setHeight(v - (v % 8));
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="pt-2 border-t border-zinc-800">
                  <div className="text-sm font-semibold text-zinc-200 mb-2">
                    ControlNet Settings
                  </div>
                  <label className="flex items-center gap-2 text-sm text-zinc-200">
                    <input
                      type="checkbox"
                      checked={enableControlnet}
                      onChange={(e) => setEnableControlnet(e.target.checked)}
                    />
                    Enable ControlNet
                  </label>

                  {enableControlnet ? (
                    <div className="mt-3 space-y-3">
                      <div>
                        <div className="text-sm text-zinc-300 mb-1">
                          条件图上传（condition_image）
                        </div>
                        <input
                          className="w-full text-sm"
                          type="file"
                          accept="image/*"
                          onChange={(e) => {
                            const f = e.target.files?.[0] ?? null;
                            handleConditionFile(f);
                          }}
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <label className="text-sm text-zinc-300">
                          条件类型
                          <select
                            className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                            value={controlnetType}
                            onChange={(e) => setControlnetType(e.target.value as ControlNetType)}
                          >
                            <option value="DT">DT</option>
                            <option value="Binary">Binary</option>
                          </select>
                        </label>

                        <label className="text-sm text-zinc-300">
                          ControlNet conditioning scale
                          <input
                            className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                            type="number"
                            step={0.05}
                            value={controlnetConditioningScale}
                            onChange={(e) =>
                              setControlnetConditioningScale(Number(e.target.value))
                            }
                          />
                        </label>
                      </div>
                    </div>
                  ) : null}
                </div>

                <div className="pt-2 border-t border-zinc-800">
                  <div className="text-sm font-semibold text-zinc-200 mb-2">
                    LoRA Settings
                  </div>
                  <label className="flex items-center gap-2 text-sm text-zinc-200">
                    <input
                      type="checkbox"
                      checked={enableLora}
                      onChange={(e) => setEnableLora(e.target.checked)}
                    />
                    Enable LoRA
                  </label>

                  {enableLora ? (
                    <div className="mt-3 space-y-3">
                      <label className="text-sm text-zinc-300 block">
                        LoRA path
                        <input
                          className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                          value={loraPath}
                          onChange={(e) => setLoraPath(e.target.value)}
                        />
                      </label>
                      <label className="text-sm text-zinc-300 block">
                        LoRA scale
                        <input
                          className="mt-1 w-full rounded-lg border border-zinc-700 bg-zinc-950 p-2 text-sm outline-none focus:border-zinc-500"
                          type="number"
                          step={0.05}
                          value={loraScale}
                          onChange={(e) => setLoraScale(Number(e.target.value))}
                        />
                      </label>
                    </div>
                  ) : null}
                </div>

                <div className="flex gap-3 pt-2">
                  <button
                    className="flex-1 rounded-lg bg-zinc-100 text-zinc-950 px-3 py-2 text-sm font-semibold hover:bg-zinc-200 disabled:opacity-60 disabled:cursor-not-allowed"
                    onClick={onGenerate}
                    disabled={loading}
                  >
                    {loading ? "Generating..." : "Generate"}
                  </button>
                  <button
                    className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-sm hover:bg-zinc-800 disabled:opacity-60 disabled:cursor-not-allowed"
                    onClick={reset}
                    disabled={loading}
                  >
                    Reset
                  </button>
                </div>

                {error ? (
                  <div className="text-sm text-red-300 bg-red-950/40 border border-red-900 rounded-lg p-3">
                    {error}
                  </div>
                ) : null}
              </div>
            </div>
          </div>

          {/* Right panel */}
          <div className="flex-1 min-w-0">
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-sm text-zinc-300">Result Preview</div>
                  <div className="text-xs text-zinc-500 mt-1">
                    {loading
                      ? "Generating..."
                      : elapsedMs !== null
                        ? `elapsed: ${elapsedMs} ms`
                        : " "}
                  </div>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <div className="flex items-center gap-2">
                    <span className="inline-flex items-center rounded-full border border-zinc-800 bg-zinc-950/40 px-2 py-1 text-[11px] text-zinc-200">
                      {modeTag}
                    </span>

                  <button
                    className="rounded-lg border border-zinc-800 bg-zinc-950/40 px-3 py-2 text-xs text-zinc-100 hover:bg-zinc-900 disabled:opacity-60 disabled:cursor-not-allowed"
                    onClick={onDownload}
                    disabled={!imageDownloadUrl || loading}
                    type="button"
                    title={imageDownloadUrl ? "Download generated PNG" : "No image yet"}
                  >
                    Download PNG
                  </button>
                  </div>
                </div>
              </div>

              <div className="mt-4 rounded-lg border border-zinc-800 bg-zinc-950 p-3">
                {imageBase64 ? (
                  <div
                    className="w-full h-[360px] flex items-center justify-center"
                    style={{ position: "relative" }}
                  >
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={`data:image/png;base64,${imageBase64}`}
                      alt="generated"
                      className="w-full h-full object-contain rounded-md"
                    />

                    {loading ? (
                      <div
                        style={{
                          position: "absolute",
                          inset: 0,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          flexDirection: "column",
                          gap: 12,
                          background: "rgba(9, 9, 11, 0.55)",
                          borderRadius: "0.375rem",
                        }}
                      >
                        <div className="w-10 h-10 rounded-full border-2 border-zinc-700 border-t-zinc-100 animate-spin" />
                        <div className="text-zinc-200">Generating...</div>
                        <div className="text-xs text-zinc-500">
                          Please keep this tab open.
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : (
                  <div className="h-[360px] flex flex-col items-center justify-center gap-3 text-sm">
                    {loading ? (
                      <>
                        <div className="w-10 h-10 rounded-full border-2 border-zinc-700 border-t-zinc-100 animate-spin" />
                        <div className="text-zinc-200">Generating...</div>
                        <div className="text-xs text-zinc-500">
                          Please keep this tab open.
                        </div>
                      </>
                    ) : (
                      <div className="text-zinc-600">
                        点击 <span className="text-zinc-200">Generate</span> 开始推理
                      </div>
                    )}
                  </div>
                )}
              </div>

              <div className="mt-4">
                <div className="text-sm text-zinc-300 mb-2">推理配置摘要</div>
                <pre className="text-xs text-zinc-200 bg-zinc-950 border border-zinc-800 rounded-lg p-3 whitespace-pre-wrap">
                  {configSummary ?? " "}
                </pre>
              </div>

              <div className="mt-4">
                <div className="text-sm text-zinc-300 mb-2">History</div>
                <div className="text-xs text-zinc-500 mb-3">recent generated results</div>

                {history.length === 0 ? (
                  <div className="text-sm text-zinc-600 py-4">暂无历史记录</div>
                ) : (
                  <div className="grid grid-cols-3 sm:grid-cols-4 gap-3">
                    {history.map((h) => (
                      <button
                        key={h.id}
                        type="button"
                        onClick={() => onPickHistory(h)}
                        className="group rounded-lg border border-zinc-800 bg-zinc-950/40 p-1 hover:border-zinc-500 transition disabled:opacity-60"
                        disabled={!h.imageBase64}
                      >
                        {h.imageBase64 ? (
                          // eslint-disable-next-line @next/next/no-img-element
                          <img
                            src={`data:image/png;base64,${h.imageBase64}`}
                            alt="history thumbnail"
                            className="w-full h-auto rounded-md object-contain"
                          />
                        ) : (
                          <div className="aspect-square flex items-center justify-center text-zinc-600 text-[11px]">
                            no image
                          </div>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

