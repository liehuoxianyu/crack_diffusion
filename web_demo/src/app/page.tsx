"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import styles from "./page.module.css";

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

type PersistedUiState = {
  prompt: string;
  negativePrompt: string;
  seed: number;
  numInferenceSteps: number;
  guidanceScale: number;
  width: number;
  height: number;
  enableControlnet: boolean;
  controlnetType: ControlNetType;
  controlnetConditioningScale: number;
  enableLora: boolean;
  loraPath: string;
  loraScale: number;
  advancedOpen: boolean;
};

const UI_STATE_STORAGE_KEY = "cracktree-webdemo-ui-state-v1";

function clamp(v: number, min: number, max: number) {
  return Math.min(max, Math.max(min, v));
}

function snapTo8(v: number) {
  const n = Math.max(64, Math.floor(v));
  return n - (n % 8);
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
  const [advancedOpen, setAdvancedOpen] = useState<boolean>(true);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // ---- Right panel: outputs ----
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [elapsedMs, setElapsedMs] = useState<number | null>(null);
  const [configSummary, setConfigSummary] = useState<string | null>(null);

  const [history, setHistory] = useState<HistoryItem[]>([]);

  const toastTimerRef = useRef<number | null>(null);
  const imageDownloadUrl = useMemo(() => {
    if (!imageBase64) return null;
    return `data:image/png;base64,${imageBase64}`;
  }, [imageBase64]);

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

  useEffect(() => {
    return () => {
      if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    };
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(UI_STATE_STORAGE_KEY);
      if (!raw) return;
      const saved = JSON.parse(raw) as Partial<PersistedUiState>;
      if (typeof saved.prompt === "string") setPrompt(saved.prompt);
      if (typeof saved.negativePrompt === "string") setNegativePrompt(saved.negativePrompt);
      if (typeof saved.seed === "number") setSeed(saved.seed);
      if (typeof saved.numInferenceSteps === "number") setNumInferenceSteps(saved.numInferenceSteps);
      if (typeof saved.guidanceScale === "number") setGuidanceScale(saved.guidanceScale);
      if (typeof saved.width === "number") setWidth(saved.width);
      if (typeof saved.height === "number") setHeight(saved.height);
      if (typeof saved.enableControlnet === "boolean") setEnableControlnet(saved.enableControlnet);
      if (saved.controlnetType === "DT" || saved.controlnetType === "Binary") {
        setControlnetType(saved.controlnetType);
      }
      if (typeof saved.controlnetConditioningScale === "number") {
        setControlnetConditioningScale(saved.controlnetConditioningScale);
      }
      if (typeof saved.enableLora === "boolean") setEnableLora(saved.enableLora);
      if (typeof saved.loraPath === "string") setLoraPath(saved.loraPath);
      if (typeof saved.loraScale === "number") setLoraScale(saved.loraScale);
      if (typeof saved.advancedOpen === "boolean") setAdvancedOpen(saved.advancedOpen);
    } catch {
      // Ignore malformed persisted state.
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const payload: PersistedUiState = {
      prompt,
      negativePrompt,
      seed,
      numInferenceSteps,
      guidanceScale,
      width,
      height,
      enableControlnet,
      controlnetType,
      controlnetConditioningScale,
      enableLora,
      loraPath,
      loraScale,
      advancedOpen,
    };
    window.localStorage.setItem(UI_STATE_STORAGE_KEY, JSON.stringify(payload));
  }, [
    prompt,
    negativePrompt,
    seed,
    numInferenceSteps,
    guidanceScale,
    width,
    height,
    enableControlnet,
    controlnetType,
    controlnetConditioningScale,
    enableLora,
    loraPath,
    loraScale,
    advancedOpen,
  ]);

  useEffect(() => {
    if (!statusMessage) return;
    const timer = window.setTimeout(() => setStatusMessage(null), 2600);
    return () => window.clearTimeout(timer);
  }, [statusMessage]);

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
    setStatusMessage("Parameters reset.");
  };

  const showError = (msg: string) => {
    setError(msg);
    setStatusMessage(null);
    if (toastTimerRef.current) window.clearTimeout(toastTimerRef.current);
    toastTimerRef.current = window.setTimeout(() => setError(null), 4200);
  };

  const handleConditionFile = (f: File | null) => {
    setConditionImageFile(f);
  };

  const onGenerate = async () => {
    setLoading(true);
    setError(null);
    setStatusMessage(null);
    try {
      if (enableControlnet && !conditionImageFile) {
        throw new Error("enable_controlnet=true 时需要上传 condition_image。");
      }

      const normalizedSeed = Number.isFinite(seed) ? Math.floor(seed) : 42;
      const normalizedSteps = clamp(Math.floor(numInferenceSteps || 25), 1, 100);
      const normalizedGuidance = clamp(Number(guidanceScale || 7.5), 1, 20);
      const normalizedWidth = clamp(snapTo8(width || 512), 64, 2048);
      const normalizedHeight = clamp(snapTo8(height || 512), 64, 2048);
      const normalizedCnScale = clamp(Number(controlnetConditioningScale || 1.0), 0, 2);
      const normalizedLoraScale = clamp(Number(loraScale || 0.7), 0, 2);

      setSeed(normalizedSeed);
      setNumInferenceSteps(normalizedSteps);
      setGuidanceScale(normalizedGuidance);
      setWidth(normalizedWidth);
      setHeight(normalizedHeight);
      setControlnetConditioningScale(normalizedCnScale);
      setLoraScale(normalizedLoraScale);

      const form = new FormData();
      form.append("prompt", prompt);
      form.append("negative_prompt", "");
      form.append("seed", String(normalizedSeed));
      form.append("num_inference_steps", String(normalizedSteps));
      form.append("guidance_scale", String(normalizedGuidance));
      form.append("width", String(normalizedWidth));
      form.append("height", String(normalizedHeight));

      form.append("enable_controlnet", enableControlnet ? "true" : "false");
      form.append("controlnet_type", controlnetType === "DT" ? "DT" : "Binary");
      form.append("controlnet_conditioning_scale", String(normalizedCnScale));

      form.append("enable_lora", enableLora ? "true" : "false");
      form.append("lora_path", enableLora ? loraPath : "");
      form.append("lora_scale", String(normalizedLoraScale));

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
      setStatusMessage(`Generation complete${data.elapsed_ms ? ` in ${data.elapsed_ms} ms` : "."}`);

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
    <div className={styles.page}>
      <div className={styles.shell}>
        <header className={styles.header}>
          <div className={styles.brand}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/aistudio.svg" alt="AIStudio logo" className={styles.brandLogo} />
            <h1 className={styles.title}>CrackTree Web Demo</h1>
          </div>
        </header>

        <div className={styles.content}>
          <aside className={styles.leftRail}>
            <div className={styles.leftRailInner}>
              <div className={styles.panelTitle}>Modules</div>
              <div className={styles.moduleStack}>
                <div className={`${styles.moduleCard} ${styles.moduleCardActive}`}>
                  <div className={styles.moduleName}>Stable Diffusion</div>
                  <div className={styles.moduleDesc}>Base model, always active</div>
                  <span className={styles.moduleBadge}>ON</span>
                </div>

                <button
                  type="button"
                  className={`${styles.moduleCard} ${enableControlnet ? styles.moduleCardActive : ""}`}
                  onClick={() => setEnableControlnet((v) => !v)}
                >
                  <div className={styles.moduleName}>ControlNet</div>
                  <div className={styles.moduleDesc}>Optional condition branch</div>
                  <span className={styles.moduleBadge}>{enableControlnet ? "ON" : "OFF"}</span>
                </button>

                <button
                  type="button"
                  className={`${styles.moduleCard} ${enableLora ? styles.moduleCardActive : ""}`}
                  onClick={() => setEnableLora((v) => !v)}
                >
                  <div className={styles.moduleName}>LoRA</div>
                  <div className={styles.moduleDesc}>Optional finetune adapter</div>
                  <span className={styles.moduleBadge}>{enableLora ? "ON" : "OFF"}</span>
                </button>
              </div>
            </div>
          </aside>

          <main className={styles.centerMain}>
            <div className={styles.card}>
              <div className={styles.cardBody}>
                <div className={styles.resultHead}>
                  <div>
                    <div className={styles.panelTitle}>Result Preview</div>
                    <div className={styles.muted}>
                      {loading
                        ? "Generating..."
                        : elapsedMs !== null
                          ? `elapsed: ${elapsedMs} ms`
                          : " "}
                    </div>
                  </div>
                  <div className={styles.resultActions}>
                    <button
                      className={styles.secondaryButton}
                      onClick={onDownload}
                      disabled={!imageDownloadUrl || loading}
                      type="button"
                      title={imageDownloadUrl ? "Download generated PNG" : "No image yet"}
                    >
                      Download PNG
                    </button>
                  </div>
                </div>

                <div className={styles.previewBox}>
                  {imageBase64 ? (
                    <div className={styles.previewFrame}>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={`data:image/png;base64,${imageBase64}`}
                        alt="generated"
                        className={styles.previewImage}
                      />

                      {loading ? (
                        <div className={styles.overlay}>
                          <div className={styles.spinner} />
                          <div>Generating...</div>
                          <div className={styles.muted}>Please keep this tab open.</div>
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    <div className={styles.empty}>
                      {loading ? (
                        <>
                          <div className={styles.spinner} />
                          <div>Generating...</div>
                          <div className={styles.muted}>Please keep this tab open.</div>
                        </>
                      ) : (
                        <div className={styles.muted}>
                          点击 <strong>Generate</strong> 开始推理
                        </div>
                      )}
                    </div>
                  )}
                </div>

                <div className={styles.summarySection}>
                  <div className={styles.panelTitle}>推理配置摘要</div>
                  <pre className={styles.summaryPre}>{configSummary ?? " "}</pre>
                  {imagePath ? <div className={styles.muted}>saved: {imagePath}</div> : null}
                </div>

                <div className={styles.summarySection}>
                  <div className={styles.panelTitle}>History</div>
                  <div className={styles.muted}>recent generated results</div>

                  {history.length === 0 ? (
                    <div className={styles.muted}>暂无历史记录</div>
                  ) : (
                    <div className={styles.historyGrid}>
                      {history.map((h) => (
                        <button
                          key={h.id}
                          type="button"
                          onClick={() => onPickHistory(h)}
                          className={styles.historyButton}
                          disabled={!h.imageBase64}
                        >
                          {h.imageBase64 ? (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img
                              src={`data:image/png;base64,${h.imageBase64}`}
                              alt="history thumbnail"
                              className={styles.thumb}
                            />
                          ) : (
                            <div className={styles.thumbPlaceholder}>no image</div>
                          )}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </main>

          <aside className={styles.rightPanel}>
            <div className={styles.card}>
              <div className={`${styles.cardBody} ${styles.stack}`}>
                <div className={styles.panelTitle}>Parameter</div>

                <div>
                  <label className={styles.label}>prompt</label>
                  <textarea
                    className={styles.textArea}
                    rows={4}
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                  />
                </div>

                {/* negative prompt removed from UI by request */}

                <div className={styles.grid2}>
                  <label>
                    <span className={styles.label}>seed</span>
                    <input
                      className={styles.input}
                      type="number"
                      value={seed}
                      onChange={(e) => setSeed(Number(e.target.value))}
                    />
                  </label>

                  <label>
                    <span className={styles.label}>steps</span>
                    <input
                      className={styles.input}
                      type="number"
                      value={numInferenceSteps}
                      onChange={(e) => setNumInferenceSteps(Number(e.target.value))}
                    />
                  </label>

                  <label>
                    <span className={styles.label}>guidance_scale</span>
                    <input
                      className={styles.input}
                      type="number"
                      step={0.1}
                      value={guidanceScale}
                      onChange={(e) => setGuidanceScale(Number(e.target.value))}
                    />
                  </label>

                  <div>
                    <span className={styles.label}>width / height</span>
                    <div className={styles.inline2}>
                      <input
                        className={styles.input}
                        type="number"
                        step={8}
                        value={width}
                        onChange={(e) => {
                          const v = Math.max(8, Number(e.target.value));
                          setWidth(v - (v % 8));
                        }}
                      />
                      <input
                        className={styles.input}
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

                {statusMessage ? <div className={styles.statusBox}>{statusMessage}</div> : null}
                <div className={styles.actions}>
                  <button
                    className={styles.primaryButton}
                    onClick={onGenerate}
                    disabled={loading}
                  >
                    {loading ? "Generating..." : "Generate"}
                  </button>
                  <button
                    className={styles.secondaryButton}
                    onClick={reset}
                    disabled={loading}
                  >
                    Reset
                  </button>
                </div>

                <div className={styles.sectionDivider}>
                  <button
                    type="button"
                    className={styles.advancedToggle}
                    onClick={() => setAdvancedOpen((v) => !v)}
                  >
                    <span className={styles.panelTitle}>Advanced Settings</span>
                    <span className={styles.advancedChevron}>{advancedOpen ? "▾" : "▸"}</span>
                  </button>
                  {advancedOpen ? (
                    <div className={styles.advancedPanel}>
                      <section className={styles.advancedSection}>
                        <div className={styles.advancedSectionHead}>
                          <div className={styles.sectionTitle}>ControlNet</div>
                          <label className={styles.switchControl}>
                            <input
                              className={styles.switchInput}
                              type="checkbox"
                              checked={enableControlnet}
                              onChange={(e) => setEnableControlnet(e.target.checked)}
                            />
                            <span className={styles.switchTrack} />
                            <span className={styles.switchText}>Enable</span>
                          </label>
                        </div>

                        {enableControlnet ? (
                          <div className={styles.advancedSectionBody}>
                            <div>
                              <label className={styles.label}>condition_image upload</label>
                              <input
                                className={styles.fileInput}
                                type="file"
                                accept="image/*"
                                onChange={(e) => {
                                  const f = e.target.files?.[0] ?? null;
                                  handleConditionFile(f);
                                }}
                              />
                            </div>

                            {conditionPreviewUrl ? (
                              // eslint-disable-next-line @next/next/no-img-element
                              <img
                                className={styles.thumb}
                                src={conditionPreviewUrl}
                                alt="condition preview"
                              />
                            ) : null}

                            <div className={styles.grid2}>
                              <label>
                                <span className={styles.label}>condition type</span>
                                <select
                                  className={styles.select}
                                  value={controlnetType}
                                  onChange={(e) =>
                                    setControlnetType(e.target.value as ControlNetType)
                                  }
                                >
                                  <option value="DT">DT</option>
                                  <option value="Binary">Binary</option>
                                </select>
                              </label>

                              <label>
                                <span className={styles.label}>conditioning scale</span>
                                <input
                                  className={styles.input}
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
                        ) : (
                          <div className={styles.muted}>
                            Enable to configure condition image and strength.
                          </div>
                        )}
                      </section>

                      <section className={styles.advancedSection}>
                        <div className={styles.advancedSectionHead}>
                          <div className={styles.sectionTitle}>LoRA</div>
                          <label className={styles.switchControl}>
                            <input
                              className={styles.switchInput}
                              type="checkbox"
                              checked={enableLora}
                              onChange={(e) => setEnableLora(e.target.checked)}
                            />
                            <span className={styles.switchTrack} />
                            <span className={styles.switchText}>Enable</span>
                          </label>
                        </div>

                        {enableLora ? (
                          <div className={styles.advancedSectionBody}>
                            <label>
                              <span className={styles.label}>LoRA path</span>
                              <input
                                className={styles.input}
                                value={loraPath}
                                onChange={(e) => setLoraPath(e.target.value)}
                              />
                            </label>
                            <label>
                              <span className={styles.label}>LoRA scale</span>
                              <input
                                className={styles.input}
                                type="number"
                                step={0.05}
                                value={loraScale}
                                onChange={(e) => setLoraScale(Number(e.target.value))}
                              />
                            </label>
                          </div>
                        ) : (
                          <div className={styles.muted}>Enable to set LoRA path and scale.</div>
                        )}
                      </section>
                    </div>
                  ) : null}
                </div>

                {error ? <div className={styles.errorBox}>{error}</div> : null}
              </div>
            </div>
          </aside>
        </div>
      </div>
    </div>
  );
}

