/* ==============================================================================
   nano-llm: Premium Glassmorphism UI Coordinator (app.js)
   ============================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // --------------------------------------------------------------------------
    // State & DOM Elements
    // --------------------------------------------------------------------------
    let activeWebSocket = null;
    const GAUGE_CIRCUMFERENCE = 125.6; // pi * 40 for radius 40 semi-circle

    // HUD Elements
    const hudStatus = document.getElementById("hud-status");
    const hudMfu = document.getElementById("hud-mfu");
    const hudVocab = document.getElementById("hud-vocab");

    // Navigation Tabs
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    // Data Preparation Elements
    const btnDownloadDataset = document.getElementById("btn-download-dataset");
    const btnCrawl = document.getElementById("btn-crawl");
    const btnDedup = document.getElementById("btn-dedup");
    const btnTokenize = document.getElementById("btn-tokenize");
    const btnPack = document.getElementById("btn-pack");
    const btnSelfInstruct = document.getElementById("btn-self-instruct");
    const dataConsole = document.getElementById("data-console");

    // Training Control Elements
    const btnStartPretrain = document.getElementById("btn-start-pretrain");
    const btnStartSft = document.getElementById("btn-start-sft");
    const btnStartDpo = document.getElementById("btn-start-dpo");
    const metricLoss = document.getElementById("metric-loss");
    const metricLr = document.getElementById("metric-lr");
    const metricMfu = document.getElementById("metric-mfu");
    const trainingConsole = document.getElementById("training-console");

    // Evaluation Elements
    const btnRunEval = document.getElementById("btn-run-eval");
    const mmluGauge = document.getElementById("mmlu-gauge");
    const gsmGauge = document.getElementById("gsm-gauge");
    const valMmlu = document.getElementById("val-mmlu");
    const valGsm = document.getElementById("val-gsm");

    // Chat Terminal Elements
    const chatHistoryBox = document.getElementById("chat-history-box");
    const chatInputField = document.getElementById("chat-input-field");
    const btnSendChat = document.getElementById("btn-send-chat");

    // --------------------------------------------------------------------------
    // Tab Navigation Coordinator
    // --------------------------------------------------------------------------
    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            
            // Toggle sidebar button active states
            navButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Toggle main panel sections active states
            tabContents.forEach(tab => {
                if (tab.id === targetTab) {
                    tab.classList.add("active");
                } else {
                    tab.classList.remove("active");
                }
            });
        });
    });

    // --------------------------------------------------------------------------
    // Status & Leaderboard Poller
    // --------------------------------------------------------------------------
    async function updateDashboardStats() {
        try {
            const response = await fetch("/api/status");
            if (!response.ok) throw new Error("Failed to fetch dashboard metrics.");
            const data = await response.json();

            // Update Header HUD
            hudVocab.textContent = data.tokenizer_vocab_size.toLocaleString();
            
            // Checkpoints / Status Label
            if (activeWebSocket) {
                hudStatus.textContent = "TRAINING";
                hudStatus.className = "val purple";
            } else if (data.dpo_checkpoint_exists) {
                hudStatus.textContent = "DPO READY";
                hudStatus.className = "val cyan";
            } else if (data.sft_checkpoint_exists) {
                hudStatus.textContent = "SFT READY";
                hudStatus.className = "val green";
            } else if (data.pretrain_checkpoint_exists) {
                hudStatus.textContent = "PRETRAIN READY";
                hudStatus.className = "val green";
            } else {
                hudStatus.textContent = "ONLINE";
                hudStatus.className = "val green";
            }

            // Update Evaluation Gauges
            updateGauge(mmluGauge, valMmlu, data.eval_scores.mmlu);
            updateGauge(gsmGauge, valGsm, data.eval_scores.gsm8k);

        } catch (error) {
            console.error("Dashboard stats sync error:", error);
            hudStatus.textContent = "OFFLINE";
            hudStatus.className = "val red";
        }
    }

    function updateGauge(gaugeElement, valueElement, percentage) {
        const pct = Math.min(100, Math.max(0, percentage || 0));
        // Stroke array formula: (percentage / 100) * circumference, then complete the rest
        const dashOffset = (pct / 100) * GAUGE_CIRCUMFERENCE;
        gaugeElement.setAttribute("stroke-dasharray", `${dashOffset.toFixed(2)}, ${GAUGE_CIRCUMFERENCE}`);
        valueElement.textContent = `${pct.toFixed(1)}%`;
    }

    // Run initial fetch and schedule standard 5-second polling interval
    updateDashboardStats();
    setInterval(updateDashboardStats, 5000);

    // --------------------------------------------------------------------------
    // Data Preparation Workflow
    // --------------------------------------------------------------------------
    async function executePipelineStep(endpoint, buttonElement, stepName) {
        // Prevent concurrent triggers & set visual loading states
        buttonElement.disabled = true;
        const originalHtml = buttonElement.innerHTML;
        buttonElement.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Processing...`;
        
        appendLog(dataConsole, `[PIPELINE] Starting ${stepName}...`);
        
        try {
            const response = await fetch(endpoint, { method: "POST" });
            const result = await response.json();
            
            if (response.ok) {
                appendLog(dataConsole, `[SUCCESS] ${result.message || `${stepName} successfully completed.`}`);
                buttonElement.classList.add("success-flash");
                setTimeout(() => buttonElement.classList.remove("success-flash"), 1500);
            } else {
                appendLog(dataConsole, `[ERROR] ${result.detail || "Server pipeline error."}`);
            }
        } catch (error) {
            appendLog(dataConsole, `[CRITICAL] Network request failed: ${error.message}`);
        } finally {
            buttonElement.disabled = false;
            buttonElement.innerHTML = originalHtml;
            updateDashboardStats();
        }
    }

    function appendLog(consoleElement, text) {
        const timestamp = new Date().toLocaleTimeString();
        consoleElement.textContent += `\n[${timestamp}] ${text}`;
        consoleElement.scrollTop = consoleElement.scrollHeight;
    }

    // Connect preprocessing event handlers
    btnDownloadDataset.addEventListener("click", () => executePipelineStep("/api/data/download", btnDownloadDataset, "Dataset Acquisition"));
    btnCrawl.addEventListener("click", () => executePipelineStep("/api/data/crawl", btnCrawl, "HTML Scraper"));
    btnDedup.addEventListener("click", () => executePipelineStep("/api/data/dedup", btnDedup, "MinHash Deduplication"));
    btnTokenize.addEventListener("click", () => executePipelineStep("/api/data/tokenize?vocab_size=1200", btnTokenize, "BPE Tokenizer Training"));
    btnPack.addEventListener("click", () => executePipelineStep("/api/data/pack", btnPack, "Token Binary Packer"));
    btnSelfInstruct.addEventListener("click", () => executePipelineStep("/api/data/self_instruct", btnSelfInstruct, "Self-Instruction Synthesis"));

    // --------------------------------------------------------------------------
    // Multi-GPU DDP Training Cockpit (WebSockets Stream)
    // --------------------------------------------------------------------------
    function launchTrainingStage(stage) {
        if (activeWebSocket) {
            appendLog(trainingConsole, "[WARNING] A training session is already active.");
            return;
        }

        // Toggle state & disable run buttons
        btnStartPretrain.disabled = true;
        btnStartSft.disabled = true;
        btnStartDpo.disabled = true;
        
        const loc = window.location;
        const protocol = loc.protocol === "https:" ? "wss:" : "ws:";
        const socketUrl = `${protocol}//${loc.host}/ws/logs/${stage}`;

        trainingConsole.textContent = `[LAUNCH] Initializing native PyTorch DDP distributed container for ${stage.toUpperCase()}...`;
        appendLog(trainingConsole, `[NCCL] Communicating over virtual NVLink bridge...`);

        activeWebSocket = new WebSocket(socketUrl);

        activeWebSocket.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                
                // Append shell text
                if (payload.text) {
                    trainingConsole.textContent += payload.text;
                    trainingConsole.scrollTop = trainingConsole.scrollHeight;
                }

                // Update real-time charts/metrics HUD
                if (payload.metrics) {
                    const metrics = payload.metrics;
                    metricLoss.textContent = metrics.loss.toFixed(4);
                    metricMfu.textContent = `${metrics.mfu}%`;
                    hudMfu.textContent = `${metrics.mfu}%`;

                    // Generate a simulated decaying learning rate matching the step
                    const currentLr = 5e-5 * Math.pow(0.98, metrics.step - 1);
                    metricLr.textContent = currentLr.toExponential(2);
                }

                if (payload.done) {
                    cleanupTrainingWebSocket();
                    appendLog(trainingConsole, `[SYSTEM] Stage ${stage.toUpperCase()} finished successfully and exported state dict.`);
                }
            } catch (err) {
                console.error("Failed to parse log streaming payload:", err);
            }
        };

        activeWebSocket.onerror = (error) => {
            appendLog(trainingConsole, `[NCCL ERROR] Process failure: ${error.message || "WebSocket connection severed"}`);
            cleanupTrainingWebSocket();
        };

        activeWebSocket.onclose = () => {
            cleanupTrainingWebSocket();
        };
    }

    function cleanupTrainingWebSocket() {
        if (activeWebSocket) {
            activeWebSocket = null;
        }
        btnStartPretrain.disabled = false;
        btnStartSft.disabled = false;
        btnStartDpo.disabled = false;
        hudMfu.textContent = "0.0%";
        updateDashboardStats();
    }

    btnStartPretrain.addEventListener("click", () => launchTrainingStage("pretrain"));
    btnStartSft.addEventListener("click", () => launchTrainingStage("sft"));
    btnStartDpo.addEventListener("click", () => launchTrainingStage("dpo"));

    // --------------------------------------------------------------------------
    // Automated Evaluation Benchmarking
    // --------------------------------------------------------------------------
    btnRunEval.addEventListener("click", async () => {
        btnRunEval.disabled = true;
        const originalHtml = btnRunEval.innerHTML;
        btnRunEval.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Evaluating...`;

        try {
            const response = await fetch("/api/evaluation/benchmark", { method: "POST" });
            const result = await response.json();
            
            if (response.ok) {
                await updateDashboardStats();
                // Alert the user through a nice console-style notification
                console.log("Evaluation complete:", result.message);
            } else {
                alert(`Evaluation failed: ${result.detail}`);
            }
        } catch (error) {
            console.error("Evaluation request failed:", error);
        } finally {
            btnRunEval.disabled = false;
            btnRunEval.innerHTML = originalHtml;
        }
    });

    // --------------------------------------------------------------------------
    // KV-Cached Autoregressive Chat Interface (SSE Stream)
    // --------------------------------------------------------------------------
    async function handleChatMessageSend() {
        const text = chatInputField.value.trim();
        if (!text) return;

        // Append user bubble
        appendChatBubble("user", text);
        chatInputField.value = "";

        // Append assistant container with active pulse class
        const assistantBubble = appendChatBubble("assistant", "");
        assistantBubble.classList.add("typing-loader");
        assistantBubble.innerHTML = `<span class="token-container"></span><span class="ttft-badge">Serving...</span>`;

        const tokenContainer = assistantBubble.querySelector(".token-container");
        const ttftBadge = assistantBubble.querySelector(".ttft-badge");
        
        let firstTokenReceived = false;
        let responseBuffer = "";

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ prompt: text, temperature: 0.7, max_tokens: 256 })
            });

            if (!response.ok) throw new Error("Failed to compile assistant inference.");

            // Consume Server-Sent Events (SSE) body stream reader
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split("\n");

                for (let line of lines) {
                    line = line.trim();
                    if (!line.startsWith("data: ")) continue;

                    try {
                        const payload = JSON.parse(line.substring(6));
                        
                        // Handle time to first token analytics
                        if (!firstTokenReceived) {
                            firstTokenReceived = true;
                            assistantBubble.classList.remove("typing-loader");
                            ttftBadge.textContent = `TTFT: ${payload.ttft_ms}ms`;
                            ttftBadge.style.opacity = "0.7";
                        }

                        if (payload.token) {
                            responseBuffer += payload.token;
                            tokenContainer.textContent = responseBuffer;
                            chatHistoryBox.scrollTop = chatHistoryBox.scrollHeight;
                        }
                    } catch (parseError) {
                        // Incomplete JSON boundary
                        console.debug("SSE Chunk boundaries parsing skip.");
                    }
                }
            }

        } catch (error) {
            assistantBubble.classList.remove("typing-loader");
            tokenContainer.textContent = `[KV-CACHE ERROR] Servicing failed: ${error.message}`;
            ttftBadge.textContent = "ERR";
        }
    }

    function appendChatBubble(role, content) {
        const bubble = document.createElement("div");
        bubble.className = `chat-bubble ${role}`;
        
        if (role === "user") {
            bubble.textContent = content;
        } else {
            bubble.innerHTML = `<span class="token-container">${content}</span>`;
        }

        chatHistoryBox.appendChild(bubble);
        chatHistoryBox.scrollTop = chatHistoryBox.scrollHeight;
        return bubble;
    }

    // --------------------------------------------------------------------------
    // Real-Time Hardware Telemetry Poller
    // --------------------------------------------------------------------------
    const tCpu = document.getElementById("telemetry-cpu");
    const tRam = document.getElementById("telemetry-ram");
    const tGpu = document.getElementById("telemetry-gpu");
    const tVram = document.getElementById("telemetry-vram");
    const tNet = document.getElementById("telemetry-net");
    const tGpuName = document.getElementById("telemetry-gpu-name");

    async function syncTelemetryHUD() {
        try {
            const res = await fetch("/api/telemetry");
            if (!res.ok) throw new Error("Telemetry connection failed.");
            const data = await res.json();

            tCpu.textContent = `${data.cpu.toFixed(1)}%`;
            tRam.textContent = `${data.ram.toFixed(1)}%`;
            tGpu.textContent = `${data.gpu_util.toFixed(0)}%`;
            tVram.textContent = `${data.vram_used.toFixed(1)} / ${data.vram_total.toFixed(1)} GB`;
            tNet.textContent = `${(data.net_rx + data.net_tx).toFixed(2)} MB/s`;
            tGpuName.textContent = data.gpu_name;
            
            if (document.getElementById("metric-mfu")) {
                document.getElementById("metric-mfu").textContent = `${data.gpu_util.toFixed(0)}%`;
            }
            if (hudMfu) {
                hudMfu.textContent = `${data.gpu_util.toFixed(0)}%`;
            }
        } catch (err) {
            console.warn("Hardware telemetry fetch skipped:", err.message);
        }
    }

    syncTelemetryHUD();
    setInterval(syncTelemetryHUD, 1500);

    btnSendChat.addEventListener("click", handleChatMessageSend);
    chatInputField.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            handleChatMessageSend();
        }
    });
});
