(() => {
    "use strict";

    const PAGE_INFO = {
        overview: { title: "首页概览", loader: loadOverview },
        wellness: { title: "天时智养", loader: loadWellness },
        vital: { title: "生命镜像", loader: loadVitalTwin },
        quest: { title: "健康远征", loader: loadQuest },
        detection: { title: "实时检测", loader: loadDetections },
        medicine: { title: "药品管理", loader: loadMedicines },
        emotion: { title: "情绪分析", loader: loadEmotion },
        stats: { title: "数据统计", loader: loadStats },
        alerts: { title: "告警记录", loader: loadAlertCenter },
        chat: { title: "健康问答", loader: loadChatStatus },
        settings: { title: "系统设置", loader: loadHealth }
    };

    const LEVEL_META = {
        danger: { label: "严重", icon: "circle-alert" },
        warning: { label: "警告", icon: "triangle-alert" },
        info: { label: "提示", icon: "info" },
        success: { label: "正常", icon: "circle-check" }
    };

    const MODE_LABELS = {
        medicine: "药品检测",
        emotion: "情绪检测",
        both: "药品与情绪检测"
    };

    const state = {
        activePage: "overview",
        alerts: [],
        cameraActive: false,
        charts: new Map(),
        health: null,
        medicines: [],
        mode: "medicine",
        recognitionLoading: false,
        recognitionTimer: null,
        refreshTimer: null,
        wellness: null,
        wellnessAccuracy: null,
        wellnessCoordinates: null,
        wellnessProfile: null,
        wellnessScene: null,
        wellnessSceneFrame: null,
        wellnessLoading: false,
        wellnessLocationAttempted: false
    };

    const $ = (selector, root = document) => root.querySelector(selector);
    const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

    function refreshIcons(root = document) {
        if (window.lucide) {
            window.lucide.createIcons({ attrs: { "stroke-width": 1.8 }, root });
        }
    }

    function createElement(tag, className, text) {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (text !== undefined) element.textContent = text;
        return element;
    }

    function iconElement(name, className = "") {
        const wrapper = createElement("span", className);
        const icon = document.createElement("i");
        icon.dataset.lucide = name;
        wrapper.appendChild(icon);
        return wrapper;
    }

    async function apiRequest(url, options = {}) {
        const config = { ...options };
        config.headers = { Accept: "application/json", ...(options.headers || {}) };
        if (config.body && typeof config.body !== "string") {
            config.headers["Content-Type"] = "application/json";
            config.body = JSON.stringify(config.body);
        }

        const response = await fetch(url, config);
        let payload = null;
        try {
            payload = await response.json();
        } catch (_error) {
            throw new Error(`服务返回异常 (${response.status})`);
        }

        if (!response.ok || (payload.code && payload.code >= 400)) {
            throw new Error(payload.message || `请求失败 (${response.status})`);
        }
        return payload.data !== undefined ? payload.data : payload;
    }

    function showToast(title, message = "", type = "info") {
        const region = $("#toastRegion");
        const toast = createElement("div", `toast ${type}`);
        const iconName = type === "success" ? "circle-check" : type === "danger" ? "circle-alert" : type === "warning" ? "triangle-alert" : "info";
        toast.appendChild(iconElement(iconName));
        const copy = createElement("div");
        copy.appendChild(createElement("strong", "", title));
        if (message) copy.appendChild(createElement("span", "", message));
        toast.appendChild(copy);
        region.appendChild(toast);
        refreshIcons(toast);
        window.setTimeout(() => toast.remove(), 3600);
    }

    function formatDateTime(value) {
        if (!value) return "--";
        const normalized = value.includes("T") ? value : value.replace(" ", "T");
        const date = new Date(normalized);
        if (Number.isNaN(date.getTime())) return value;
        return new Intl.DateTimeFormat("zh-CN", {
            month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit"
        }).format(date);
    }

    function formatDate(value) {
        if (!value) return "未设置";
        const date = new Date(`${value}T00:00:00`);
        if (Number.isNaN(date.getTime())) return value;
        return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" }).format(date);
    }

    function setPageMeta() {
        const now = new Date();
        $("#pageMeta").textContent = new Intl.DateTimeFormat("zh-CN", {
            year: "numeric", month: "long", day: "numeric", weekday: "short"
        }).format(now);
    }

    function navigate(page, updateHash = true) {
        if (!PAGE_INFO[page]) page = "overview";
        if (state.activePage === "detection" && page !== "detection") stopCamera();
        if (state.activePage === "quest" && page !== "quest" && window.MedProQuest) {
            window.MedProQuest.pause();
        }
        if (state.activePage === "vital" && page !== "vital" && window.MedProVitalTwin) {
            window.MedProVitalTwin.pause();
        }
        state.activePage = page;

        $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.page === page));
        $$(".app-page").forEach((panel) => panel.classList.toggle("active", panel.dataset.pagePanel === page));
        $("#pageTitle").textContent = PAGE_INFO[page].title;
        $("#pageScroll").scrollTop = 0;
        document.body.classList.remove("sidebar-open");

        if (updateHash && window.location.hash !== `#${page}`) {
            window.history.replaceState(null, "", `#${page}`);
        }

        PAGE_INFO[page].loader().catch((error) => {
            showToast("页面加载失败", error.message, "danger");
        });
        resizeCharts();
    }

    function trendText(value) {
        const number = Number(value || 0);
        if (number === 0) return { text: "较昨日持平", className: "" };
        return {
            text: `较昨日 ${number > 0 ? "+" : ""}${number}%`,
            className: number > 0 ? "up" : "down"
        };
    }

    async function loadOverview() {
        const [overview, health, alerts, medicineStats, trend] = await Promise.all([
            apiRequest("/api/overview"),
            apiRequest("/api/health").catch(() => null),
            fetchAlerts({ limit: 4, status: "open" }),
            apiRequest("/api/stats/medicine"),
            apiRequest("/api/stats/trend")
        ]);

        $("#todayCount").textContent = overview.today_count ?? 0;
        $("#alertCount").textContent = overview.alert_count ?? 0;
        $("#llmStatus").textContent = overview.llm_status || "未连接";
        $("#llmDesc").textContent = overview.llm_desc || "未检测";
        $("#syncTime").textContent = overview.time || "--:--:--";
        $("#syncDate").textContent = overview.date || "----";

        const todayTrend = trendText(overview.today_count_rate);
        $("#todayTrend").textContent = todayTrend.text;
        $("#todayTrend").className = `metric-trend ${todayTrend.className}`.trim();
        const alertTrend = trendText(overview.alert_rate);
        $("#alertTrend").textContent = alertTrend.text;
        $("#alertTrend").className = `metric-trend ${alertTrend.className}`.trim();

        if (health) renderHealth(health);
        renderCompactAlerts(alerts);
        renderMedicineChart("overviewMedicineChart", medicineStats);
        renderTrendChart("overviewTrendChart", trend);
    }

    async function loadQuest() {
        if (window.MedProQuest) {
            await window.MedProQuest.load();
        }
    }

    async function loadVitalTwin() {
        if (window.MedProVitalTwin) {
            await window.MedProVitalTwin.load();
        }
    }

    function healthItems(health) {
        const detectorOk = health.detector && !["MOCK", "unavailable"].includes(health.detector);
        const emotionOk = health.emotion && !["MOCK", "unavailable"].includes(health.emotion);
        const databaseOk = health.database === "connected";
        return [
            { label: "药品检测", value: health.detector || "不可用", ok: detectorOk, icon: "scan-line" },
            { label: "情绪模型", value: health.emotion || "不可用", ok: emotionOk, icon: "activity" },
            { label: "本地问答", value: health.llm_backend || "未连接", ok: Boolean(health.llm_available), icon: "brain-circuit" },
            { label: "数据存储", value: databaseOk ? "已连接" : "未连接", ok: databaseOk, icon: "database" }
        ];
    }

    function renderHealth(health) {
        state.health = health;
        const items = healthItems(health);
        const availableCount = items.filter((item) => item.ok).length;
        const healthy = availableCount === items.length;
        const partial = availableCount > 0 && !healthy;
        const statusClass = healthy ? "success" : partial ? "warning" : "danger";
        const statusText = healthy ? "系统正常" : partial ? "部分组件受限" : "系统不可用";

        const systemChip = $("#systemChip");
        $(".status-dot", systemChip).className = `status-dot ${statusClass}`;
        $("#systemChipText").textContent = statusText;

        const sidebar = $("#sidebarConnection");
        $(".status-dot", sidebar).className = `status-dot ${statusClass}`;
        $("strong", sidebar).textContent = statusText;
        $(".connection-state div > span", sidebar).textContent = `${availableCount}/${items.length} 个组件可用`;

        const list = $("#overviewStatusList");
        list.replaceChildren(...items.slice(0, 3).map((item) => buildStatusRow(item)));
        const details = $("#healthDetails");
        details.replaceChildren(...items.map((item) => buildStatusRow(item)));
        $("#chatBackend").textContent = health.llm_backend || "模型未连接";
        refreshIcons(list);
        refreshIcons(details);
    }

    function buildStatusRow(item) {
        const row = createElement("div", "status-row");
        const label = createElement("div", "status-row-label");
        label.appendChild(iconElement(item.icon));
        label.appendChild(createElement("span", "", item.label));
        row.appendChild(label);
        const badge = createElement("span", `status-badge ${item.ok ? "success" : "warning"}`, item.value);
        row.appendChild(badge);
        return row;
    }

    async function loadHealth() {
        const health = await apiRequest("/api/health");
        renderHealth(health);
        try {
            const modeData = await apiRequest("/api/mode");
            state.mode = modeData.mode;
            $("#settingsMode").value = state.mode;
            updateModeButtons();
        } catch (_error) {
            // Health details remain useful if mode lookup fails.
        }
    }

    async function fetchAlerts({ limit = 50, status = "all", level = "all" } = {}) {
        const params = new URLSearchParams({ limit: String(limit), status, level });
        try {
            return await apiRequest(`/api/alerts?${params}`);
        } catch (_error) {
            return apiRequest(`/api/alerts/latest?limit=${limit}`);
        }
    }

    function renderCompactAlerts(alerts) {
        const container = $("#overviewAlertList");
        if (!alerts.length) {
            container.replaceChildren(createElement("div", "empty-inline", "暂无待处理告警"));
            return;
        }

        const nodes = alerts.map((alert) => {
            const item = createElement("article", "compact-alert");
            item.appendChild(createElement("span", `severity-bar ${alert.level || "info"}`));
            const copy = createElement("div", "compact-alert-copy");
            copy.appendChild(createElement("strong", "", alert.title || "未命名告警"));
            copy.appendChild(createElement("span", "", alert.content || ""));
            item.appendChild(copy);
            item.appendChild(createElement("time", "tabular", alert.time || formatDateTime(alert.created_at)));
            return item;
        });
        container.replaceChildren(...nodes);
    }

    async function loadAlertCenter() {
        const status = $("#alertStatusFilter").value;
        const level = $("#alertLevelFilter").value;
        const alerts = await fetchAlerts({ limit: 100, status, level });
        state.alerts = alerts;
        renderAlertCenter(alerts);
        updateAlertCount();
    }

    function renderAlertCenter(alerts) {
        const container = $("#alertCenterList");
        if (!alerts.length) {
            container.replaceChildren(buildEmptyState("bell-off", "暂无符合条件的告警"));
            return;
        }

        const nodes = alerts.map((alert) => {
            const meta = LEVEL_META[alert.level] || LEVEL_META.info;
            const item = createElement("article", "alert-record");
            item.dataset.alertId = alert.id || "";
            item.appendChild(iconElement(meta.icon, `alert-record-icon ${alert.level || "info"}`));
            const copy = createElement("div", "alert-record-copy");
            copy.appendChild(createElement("strong", "", alert.title || "未命名告警"));
            copy.appendChild(createElement("span", "", alert.content || ""));
            item.appendChild(copy);
            item.appendChild(createElement("time", "tabular", alert.time || formatDateTime(alert.created_at)));

            if (alert.status === "acknowledged") {
                item.appendChild(createElement("span", "status-badge success", "已处理"));
            } else if (alert.id) {
                const button = createElement("button", "secondary-button", "确认处理");
                button.type = "button";
                button.dataset.ackAlert = alert.id;
                item.appendChild(button);
            } else {
                item.appendChild(createElement("span", `status-badge ${alert.level || "info"}`, meta.label));
            }
            return item;
        });
        container.replaceChildren(...nodes);
        refreshIcons(container);
    }

    async function acknowledgeAlert(id) {
        const button = $(`[data-ack-alert="${CSS.escape(String(id))}"]`);
        if (button) button.disabled = true;
        try {
            await apiRequest(`/api/alerts/${id}/acknowledge`, { method: "POST" });
            showToast("告警已处理", "处置状态已更新", "success");
            await loadAlertCenter();
        } catch (error) {
            if (button) button.disabled = false;
            showToast("更新失败", error.message, "danger");
        }
    }

    async function updateAlertCount() {
        try {
            const openAlerts = await fetchAlerts({ limit: 100, status: "open" });
            const count = openAlerts.length;
            const badge = $("#navAlertCount");
            badge.textContent = count > 99 ? "99+" : String(count);
            badge.classList.toggle("hidden", count === 0);
        } catch (_error) {
            // Navigation remains usable without a badge.
        }
    }

    function chartInstance(id) {
        if (!window.echarts) return null;
        if (state.charts.has(id)) return state.charts.get(id);
        const element = document.getElementById(id);
        if (!element) return null;
        const instance = window.echarts.init(element, null, { renderer: "canvas" });
        state.charts.set(id, instance);
        return instance;
    }

    function emptyChartOption(message = "暂无数据") {
        return {
            animation: false,
            title: { text: message, left: "center", top: "middle", textStyle: { color: "#89939d", fontSize: 12, fontWeight: 400 } },
            series: []
        };
    }

    function renderMedicineChart(id, data) {
        const chart = chartInstance(id);
        if (!chart) return;
        chart.clear();
        if (!data || !data.length) {
            chart.setOption(emptyChartOption());
            return;
        }
        chart.setOption({
            color: ["#0f8b7a", "#3973e6", "#d38b18", "#2f9b63", "#d95d5d", "#2389a8", "#8367a8", "#64748b"],
            tooltip: { trigger: "item", formatter: "{b}<br/>{c} 次 · {d}%", borderWidth: 0, textStyle: { fontSize: 12 } },
            legend: { bottom: 8, left: "center", itemHeight: 8, itemWidth: 8, textStyle: { color: "#5f6c78", fontSize: 10 } },
            series: [{
                type: "pie", radius: ["48%", "70%"], center: ["50%", "43%"],
                itemStyle: { borderColor: "#ffffff", borderWidth: 2, borderRadius: 3 },
                label: { show: false }, emphasis: { scaleSize: 4 }, data
            }]
        });
    }

    function renderTrendChart(id, data) {
        const chart = chartInstance(id);
        if (!chart) return;
        chart.clear();
        if (!data || !data.values || !data.values.some((value) => Number(value) > 0)) {
            chart.setOption(emptyChartOption());
            return;
        }
        chart.setOption({
            color: ["#3973e6"],
            tooltip: { trigger: "axis", borderWidth: 0, textStyle: { fontSize: 12 }, axisPointer: { type: "line", lineStyle: { color: "#cbd4d9" } } },
            grid: { left: 46, right: 18, top: 25, bottom: 34 },
            xAxis: { type: "category", data: data.dates, boundaryGap: false, axisTick: { show: false }, axisLine: { lineStyle: { color: "#dfe5e8" } }, axisLabel: { color: "#89939d", fontSize: 10 } },
            yAxis: { type: "value", minInterval: 1, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { color: "#89939d", fontSize: 10 }, splitLine: { lineStyle: { color: "#edf1f3" } } },
            series: [{ type: "line", data: data.values, smooth: 0.25, symbol: "circle", symbolSize: 6, lineStyle: { width: 3 }, itemStyle: { color: "#3973e6", borderColor: "#ffffff", borderWidth: 2 }, areaStyle: { color: "rgba(57, 115, 230, 0.12)" } }]
        });
    }

    async function loadStats() {
        const [medicineStats, trend] = await Promise.all([
            apiRequest("/api/stats/medicine"), apiRequest("/api/stats/trend")
        ]);
        renderTrendChart("statsTrendChart", trend);
        renderMedicineChart("statsMedicineChart", medicineStats);
    }

    function resizeCharts() {
        window.requestAnimationFrame(() => {
            state.charts.forEach((chart) => chart.resize());
            resizeWellnessScene();
            if (window.MedProVitalTwin) window.MedProVitalTwin.resize();
        });
    }

    async function loadDetections() {
        let detections = [];
        try {
            detections = await apiRequest("/api/detections/latest?limit=30");
        } catch (_error) {
            detections = [];
        }
        renderDetections(detections);
        const modeData = await apiRequest("/api/mode").catch(() => ({ mode: state.mode }));
        state.mode = modeData.mode || state.mode;
        updateModeButtons();
    }

    function renderDetections(items) {
        const container = $("#detectionList");
        if (!items.length) {
            container.replaceChildren(buildEmptyState("scan-line", "暂无有效识别记录"));
            return;
        }
        const nodes = items.map((item) => {
            const row = createElement("article", "detection-item");
            row.appendChild(iconElement("pill", "detection-item-icon"));
            const copy = createElement("div", "detection-item-copy");
            copy.appendChild(createElement("strong", "", item.medicine_name || "未识别药品"));
            copy.appendChild(createElement("span", "", `${item.medicine_category || "其他"} · ${formatDateTime(item.created_at)}`));
            if (item.medicine_efficacy) {
                copy.appendChild(createElement("p", "detection-item-efficacy", item.medicine_efficacy));
            }
            row.appendChild(copy);
            row.appendChild(createElement("span", "confidence-value tabular", `${Math.round(Number(item.confidence || 0) * 100)}%`));
            return row;
        });
        container.replaceChildren(...nodes);
        refreshIcons(container);
    }

    function setRecognitionHeader(name, label, tone = "neutral") {
        $("#liveRecognitionName").textContent = name;
        const badge = $("#recognitionState");
        badge.className = `recognition-state ${tone}`;
        badge.replaceChildren(createElement("span", `status-dot ${tone}`), document.createTextNode(label));
    }

    function renderRecognitionEmpty(name, label, tone, message, icon = "scan-line") {
        setRecognitionHeader(name, label, tone);
        const empty = createElement("div", "live-recognition-empty");
        empty.appendChild(iconElement(icon));
        empty.appendChild(createElement("span", "", message));
        $("#liveRecognitionList").replaceChildren(empty);
        refreshIcons(empty);
    }

    function renderCurrentRecognition(payload) {
        if (!state.cameraActive) {
            renderRecognitionEmpty("等待检测结果", "未检测", "neutral", "暂无药品结果");
            return;
        }
        if (state.mode === "emotion") {
            renderRecognitionEmpty("药品识别已关闭", "情绪模式", "neutral", "当前仅输出情绪检测结果", "activity");
            return;
        }

        const detections = Array.isArray(payload?.detections) ? payload.detections : [];
        if (!payload?.active || !detections.length) {
            renderRecognitionEmpty("正在检测药盒", "检测中", "warning", "等待药品进入有效识别区域", "scan-line");
            return;
        }

        const known = detections.filter((item) => item.status === "known");
        const headline = known.length === 1
            ? known[0].name
            : known.length > 1
                ? `已识别 ${known.length} 种药品`
                : "正在读取包装文字";
        setRecognitionHeader(headline, known.length ? "已确认" : "识别中", known.length ? "success" : "warning");

        const nodes = detections.map((item) => {
            const article = createElement("article", `live-recognition-item ${item.status || "pending"}`);
            const heading = createElement("div", "live-recognition-item-head");
            const identity = createElement("div", "live-recognition-identity");
            identity.appendChild(iconElement(item.status === "known" ? "badge-check" : "scan-text"));
            const nameGroup = createElement("div");
            nameGroup.appendChild(createElement("strong", "", item.name || "识别中..."));
            nameGroup.appendChild(createElement("span", "", item.category || "待查询"));
            identity.appendChild(nameGroup);
            heading.appendChild(identity);
            const confidence = Math.max(0, Math.min(1, Number(item.recognition_confidence || 0)));
            heading.appendChild(createElement("span", "live-recognition-confidence tabular", `${Math.round(confidence * 100)}%`));
            article.appendChild(heading);
            article.appendChild(createElement("p", "live-recognition-efficacy", item.efficacy || "功效信息正在查询"));
            return article;
        });
        $("#liveRecognitionList").replaceChildren(...nodes);
        refreshIcons($("#liveRecognitionList"));
    }

    async function loadCurrentRecognition() {
        if (!state.cameraActive || state.recognitionLoading) return;
        state.recognitionLoading = true;
        try {
            renderCurrentRecognition(await apiRequest("/api/recognition/current"));
        } catch (_error) {
            renderRecognitionEmpty("识别服务暂不可用", "连接异常", "danger", "未能读取当前识别结果", "circle-alert");
        } finally {
            state.recognitionLoading = false;
        }
    }

    function startRecognitionPolling() {
        if (state.recognitionTimer) window.clearInterval(state.recognitionTimer);
        loadCurrentRecognition();
        state.recognitionTimer = window.setInterval(loadCurrentRecognition, 650);
    }

    function stopRecognitionPolling() {
        if (state.recognitionTimer) window.clearInterval(state.recognitionTimer);
        state.recognitionTimer = null;
        state.recognitionLoading = false;
    }

    function buildEmptyState(icon, text) {
        const empty = createElement("div", "empty-state");
        empty.appendChild(iconElement(icon, "empty-icon"));
        empty.appendChild(createElement("strong", "", text));
        refreshIcons(empty);
        return empty;
    }

    async function setMode(mode, notify = false) {
        const result = await apiRequest("/api/mode", { method: "POST", body: { mode } });
        state.mode = result.mode;
        updateModeButtons();
        if (state.cameraActive) loadCurrentRecognition();
        if (notify) showToast("检测模式已更新", MODE_LABELS[state.mode], "success");
    }

    function updateModeButtons() {
        $$("#detectModeControl button").forEach((button) => button.classList.toggle("active", button.dataset.mode === state.mode));
        $("#settingsMode").value = state.mode;
        $("#videoModeLabel").textContent = MODE_LABELS[state.mode] || "实时检测";
    }

    function startCamera() {
        if (state.cameraActive) return;
        state.cameraActive = true;
        const feed = $("#videoFeed");
        feed.hidden = false;
        $("#videoPlaceholder").hidden = true;
        feed.src = `/video_feed?t=${Date.now()}`;
        $("#cameraBadge").className = "status-badge warning";
        $("#cameraBadge").replaceChildren(createElement("span", "status-dot warning"), document.createTextNode("连接中"));
        $("#toggleCamera").replaceChildren(iconElement("video-off"), document.createTextNode("停止画面"));
        refreshIcons($("#toggleCamera"));
        renderRecognitionEmpty("正在启动识别", "连接中", "warning", "等待视频流与识别模型就绪", "loader-circle");
        startRecognitionPolling();
    }

    function stopCamera() {
        if (!state.cameraActive) return;
        state.cameraActive = false;
        stopRecognitionPolling();
        const feed = $("#videoFeed");
        feed.removeAttribute("src");
        feed.hidden = true;
        $("#videoPlaceholder").hidden = false;
        $("#cameraBadge").className = "status-badge neutral";
        $("#cameraBadge").replaceChildren(createElement("span", "status-dot neutral"), document.createTextNode("未启动"));
        $("#toggleCamera").replaceChildren(iconElement("video"), document.createTextNode("启动画面"));
        refreshIcons($("#toggleCamera"));
        renderRecognitionEmpty("等待检测结果", "未检测", "neutral", "暂无药品结果");
    }

    async function loadMedicines() {
        state.medicines = await apiRequest("/api/medicines");
        renderMedicines();
    }

    function medicineStatus(medicine) {
        const stock = Number(medicine.stock || 0);
        if (medicine.expire_date) {
            const today = new Date();
            today.setHours(0, 0, 0, 0);
            const expire = new Date(`${medicine.expire_date}T00:00:00`);
            const days = Math.ceil((expire - today) / 86400000);
            if (days < 0) return { key: "expired", label: "已过期", className: "danger" };
            if (days <= 30) return { key: "expiring", label: "即将过期", className: "warning" };
        }
        if (stock <= 2) return { key: "low", label: "库存不足", className: "warning" };
        return { key: "normal", label: "正常", className: "success" };
    }

    function filteredMedicines() {
        const query = $("#medicineSearch").value.trim().toLowerCase();
        const filter = $("#medicineFilter").value;
        return state.medicines.filter((medicine) => {
            const matchesText = !query || [medicine.name, medicine.category, medicine.description].some((value) => String(value || "").toLowerCase().includes(query));
            const matchesStatus = filter === "all" || medicineStatus(medicine).key === filter;
            return matchesText && matchesStatus;
        });
    }

    function renderMedicines() {
        const medicines = filteredMedicines();
        const empty = $("#medicineEmpty");
        empty.classList.toggle("hidden", medicines.length > 0);
        $(".table-wrap", $(".table-surface")).classList.toggle("hidden", medicines.length === 0);
        $("#medicineMobileList").classList.toggle("hidden", medicines.length === 0);

        const rows = medicines.map((medicine) => {
            const status = medicineStatus(medicine);
            const row = document.createElement("tr");
            row.appendChild(buildMedicineNameCell(medicine));
            row.appendChild(createTableCell(medicine.category || "未分类"));
            row.appendChild(createTableCell(formatDate(medicine.expire_date), "tabular"));
            row.appendChild(createTableCell(`${Number(medicine.stock || 0)} 件`, "tabular"));
            const statusCell = document.createElement("td");
            statusCell.appendChild(createElement("span", `status-badge ${status.className}`, status.label));
            row.appendChild(statusCell);
            const actions = document.createElement("td");
            actions.className = "actions-column";
            actions.appendChild(buildMedicineActions(medicine));
            row.appendChild(actions);
            return row;
        });
        $("#medicineTableBody").replaceChildren(...rows);

        const mobileItems = medicines.map((medicine) => {
            const status = medicineStatus(medicine);
            const item = createElement("article", "medicine-mobile-item");
            const head = createElement("div", "medicine-mobile-head");
            head.appendChild(createElement("strong", "", medicine.name));
            head.appendChild(createElement("span", `status-badge ${status.className}`, status.label));
            item.appendChild(head);
            const meta = createElement("div", "medicine-mobile-meta");
            meta.appendChild(createElement("span", "", medicine.category || "未分类"));
            meta.appendChild(createElement("span", "tabular", `库存 ${Number(medicine.stock || 0)} 件`));
            meta.appendChild(createElement("span", "tabular", formatDate(medicine.expire_date)));
            item.appendChild(meta);
            const actions = buildMedicineActions(medicine);
            actions.classList.add("medicine-mobile-actions");
            item.appendChild(actions);
            return item;
        });
        $("#medicineMobileList").replaceChildren(...mobileItems);
        refreshIcons($(".table-surface"));
    }

    function buildMedicineNameCell(medicine) {
        const cell = document.createElement("td");
        const content = createElement("div", "medicine-name");
        content.appendChild(createElement("strong", "", medicine.name));
        content.appendChild(createElement("span", "", medicine.description || "无备注"));
        cell.appendChild(content);
        return cell;
    }

    function createTableCell(text, className = "") {
        const cell = document.createElement("td");
        cell.className = className;
        cell.textContent = text;
        return cell;
    }

    function buildMedicineActions(medicine) {
        const actions = createElement("div", "table-actions");
        const edit = createElement("button", "icon-button compact");
        edit.type = "button";
        edit.title = "编辑药品";
        edit.setAttribute("aria-label", `编辑${medicine.name}`);
        edit.dataset.editMedicine = medicine.id;
        edit.appendChild(iconElement("pencil"));
        const remove = createElement("button", "icon-button compact");
        remove.type = "button";
        remove.title = "删除药品";
        remove.setAttribute("aria-label", `删除${medicine.name}`);
        remove.dataset.deleteMedicine = medicine.id;
        remove.appendChild(iconElement("trash-2"));
        actions.append(edit, remove);
        return actions;
    }

    function openMedicineDialog(medicine = null) {
        $("#medicineForm").reset();
        $("#medicineId").value = medicine?.id || "";
        $("#medicineName").value = medicine?.name || "";
        $("#medicineCategory").value = medicine?.category || "";
        $("#medicineStock").value = medicine?.stock ?? 0;
        $("#medicineExpireDate").value = medicine?.expire_date || "";
        $("#medicineDescription").value = medicine?.description || "";
        $("#medicineDialogTitle").textContent = medicine ? "编辑药品" : "新增药品";
        $("#medicineDialog").showModal();
        window.setTimeout(() => $("#medicineName").focus(), 50);
    }

    async function saveMedicine(event) {
        event.preventDefault();
        const id = $("#medicineId").value;
        const name = $("#medicineName").value.trim();
        if (!name) {
            $(".field-error", $("#medicineForm")).textContent = "请输入药品名称";
            $("#medicineName").focus();
            return;
        }
        $(".field-error", $("#medicineForm")).textContent = "";
        const payload = {
            name,
            category: $("#medicineCategory").value.trim() || null,
            stock: Number($("#medicineStock").value || 0),
            expire_date: $("#medicineExpireDate").value || null,
            description: $("#medicineDescription").value.trim() || null
        };

        const button = $("#saveMedicine");
        button.disabled = true;
        try {
            await apiRequest(id ? `/api/medicines/${id}` : "/api/medicines", { method: id ? "PUT" : "POST", body: payload });
            $("#medicineDialog").close();
            showToast(id ? "药品已更新" : "药品已添加", name, "success");
            await loadMedicines();
        } catch (error) {
            showToast("保存失败", error.message, "danger");
        } finally {
            button.disabled = false;
        }
    }

    function requestConfirmation(title, message) {
        return new Promise((resolve) => {
            const dialog = $("#confirmDialog");
            $("#confirmTitle").textContent = title;
            $("#confirmMessage").textContent = message;
            const onClose = () => {
                dialog.removeEventListener("close", onClose);
                resolve(dialog.returnValue === "confirm");
            };
            dialog.addEventListener("close", onClose);
            dialog.showModal();
        });
    }

    async function deleteMedicine(id) {
        const medicine = state.medicines.find((item) => String(item.id) === String(id));
        if (!medicine) return;
        const confirmed = await requestConfirmation("删除药品", `确认删除“${medicine.name}”？`);
        if (!confirmed) return;
        try {
            await apiRequest(`/api/medicines/${id}`, { method: "DELETE" });
            showToast("药品已删除", medicine.name, "success");
            await loadMedicines();
        } catch (error) {
            showToast("删除失败", error.message, "danger");
        }
    }

    async function loadEmotion() {
        let summary = { latest: null, distribution: [] };
        try {
            summary = await apiRequest("/api/emotions/summary");
        } catch (_error) {
            // Render a truthful empty state when the endpoint is unavailable.
        }
        renderEmotion(summary);
    }

    function renderEmotion(summary) {
        const reading = $("#emotionReading");
        const latest = summary.latest;
        reading.replaceChildren(iconElement("activity", "emotion-symbol"));
        reading.appendChild(createElement("strong", "", latest?.emotion || "暂无记录"));
        const confidence = Number(latest?.emotion_confidence || 0);
        reading.appendChild(createElement("span", "tabular", latest ? `可信度 ${Math.round(confidence * 100)}% · ${formatDateTime(latest.created_at)}` : "--"));
        $("#emotionConfidenceBar").style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
        refreshIcons(reading);

        const chart = chartInstance("emotionChart");
        if (!chart) return;
        chart.clear();
        if (!summary.distribution?.length) {
            chart.setOption(emptyChartOption());
            return;
        }
        const emotionColors = {
            "开心": "#2f9b63", "平静": "#2389a8", "惊讶": "#d38b18",
            "悲伤": "#3973e6", "恐惧": "#8367a8", "愤怒": "#d95d5d",
            "厌恶": "#75677f"
        };
        chart.setOption({
            tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, borderWidth: 0 },
            grid: { left: 70, right: 24, top: 22, bottom: 32 },
            xAxis: { type: "value", minInterval: 1, splitLine: { lineStyle: { color: "#edf1f3" } }, axisLabel: { color: "#89939d" } },
            yAxis: { type: "category", data: summary.distribution.map((item) => item.name), axisTick: { show: false }, axisLine: { show: false }, axisLabel: { color: "#5f6c78" } },
            series: [{
                type: "bar",
                data: summary.distribution.map((item) => ({
                    value: item.value,
                    itemStyle: { color: emotionColors[item.name] || "#8367a8" }
                })),
                barWidth: 16,
                itemStyle: { borderRadius: [0, 3, 3, 0] }
            }]
        });
    }

    function initializeChat() {
        const box = $("#chatMessages");
        if (box.children.length) return;
        appendChatMessage("您好，我是 MedPro 健康科普助手。", "assistant");
    }

    async function loadChatStatus() {
        initializeChat();
        if (!state.health) {
            const health = await apiRequest("/api/health").catch(() => null);
            if (health) renderHealth(health);
        }
    }

    function appendChatMessage(text, role, loading = false) {
        const message = createElement("article", `chat-message ${role === "user" ? "user" : "assistant"}${loading ? " loading" : ""}`);
        message.appendChild(iconElement(role === "user" ? "user" : "bot", "message-avatar"));
        message.appendChild(createElement("div", "message-bubble", text));
        $("#chatMessages").appendChild(message);
        $("#chatMessages").scrollTop = $("#chatMessages").scrollHeight;
        refreshIcons(message);
        return message;
    }

    async function sendQuestion(event) {
        event.preventDefault();
        const input = $("#questionInput");
        const question = input.value.trim();
        if (!question) return;
        appendChatMessage(question, "user");
        input.value = "";
        input.style.height = "auto";
        $("#sendBtn").disabled = true;
        const loading = appendChatMessage("正在生成回答...", "assistant", true);
        try {
            const result = await apiRequest("/api/chat", { method: "POST", body: { question } });
            loading.remove();
            appendChatMessage(result.answer || "暂无回答。", "assistant");
        } catch (error) {
            loading.remove();
            appendChatMessage(`请求失败：${error.message}`, "assistant");
        } finally {
            $("#sendBtn").disabled = false;
            input.focus();
        }
    }

    async function loadWellness(force = false) {
        initializeWellnessScene();
        if (!state.wellnessProfile) {
            state.wellnessProfile = await apiRequest("/api/wellness/profile").catch(() => null);
        }

        if (state.wellness && !force) {
            renderWellness(state.wellness);
            resizeWellnessScene();
            return;
        }

        if (!state.wellnessCoordinates) {
            const sessionCoordinates = readStoredCoordinates(sessionStorage, "medpro-wellness-session");
            const retainedCoordinates = state.wellnessProfile?.retain_location
                ? readStoredCoordinates(localStorage, "medpro-wellness-location")
                : null;
            state.wellnessCoordinates = sessionCoordinates || retainedCoordinates;
        }

        if (state.wellnessCoordinates) {
            await fetchWellness(state.wellnessCoordinates, force);
        } else if (!state.wellnessLocationAttempted) {
            await requestBrowserLocation(true);
        }
    }

    function readStoredCoordinates(storage, key) {
        try {
            const value = JSON.parse(storage.getItem(key) || "null");
            if (!value || !Number.isFinite(value.latitude) || !Number.isFinite(value.longitude)) return null;
            return value;
        } catch (_error) {
            return null;
        }
    }

    function requestBrowserLocation(automatic = false) {
        state.wellnessLocationAttempted = true;
        if (!navigator.geolocation) {
            setWellnessStandby("浏览器不支持定位，请搜索城市");
            return Promise.resolve();
        }
        if (!window.isSecureContext) {
            setWellnessStandby("当前地址无法申请定位，请搜索城市");
            showToast("定位需要安全连接", "localhost 可直接使用；局域网或公网请启用 HTTPS", "warning");
            return Promise.resolve();
        }

        startWellnessLoader("正在获取位置");
        return new Promise((resolve) => {
            navigator.geolocation.getCurrentPosition(async (position) => {
                const coordinates = {
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude,
                    accuracy: position.coords.accuracy,
                };
                state.wellnessAccuracy = position.coords.accuracy;
                state.wellnessCoordinates = coordinates;
                try {
                    await fetchWellness(coordinates, true);
                } finally {
                    resolve();
                }
            }, (error) => {
                stopWellnessLoader();
                const messages = {
                    1: "定位权限未开启，可搜索城市继续",
                    2: "暂时无法确定位置，可搜索城市继续",
                    3: "定位超时，可重试或搜索城市",
                };
                const message = messages[error.code] || "定位失败，可搜索城市继续";
                setWellnessStandby(message);
                if (!automatic || error.code !== 1) showToast("未获取当前位置", message, "warning");
                resolve();
            }, { enableHighAccuracy: true, timeout: 12000, maximumAge: 300000 });
        });
    }

    async function fetchWellness(coordinates, force = false) {
        if (state.wellnessLoading) return;
        state.wellnessLoading = true;
        startWellnessLoader("同步实时天气");
        const progress = animateWellnessProgress();
        const params = new URLSearchParams({
            latitude: String(coordinates.latitude),
            longitude: String(coordinates.longitude),
        });
        if (coordinates.name) params.set("location_name", coordinates.name);
        if (force) params.set("force", "true");

        try {
            const data = await apiRequest(`/api/wellness/daily?${params.toString()}`);
            state.wellness = data;
            state.wellnessCoordinates = {
                latitude: data.weather.location.latitude,
                longitude: data.weather.location.longitude,
                accuracy: coordinates.accuracy || null,
                name: coordinates.name || data.weather.location.name,
            };
            state.wellnessAccuracy = coordinates.accuracy || null;
            sessionStorage.setItem("medpro-wellness-session", JSON.stringify(state.wellnessCoordinates));
            if (state.wellnessProfile?.retain_location) {
                localStorage.setItem("medpro-wellness-location", JSON.stringify({
                    latitude: Number(state.wellnessCoordinates.latitude.toFixed(2)),
                    longitude: Number(state.wellnessCoordinates.longitude.toFixed(2)),
                    name: state.wellnessCoordinates.name,
                }));
            }
            renderWellness(data);
        } catch (error) {
            setWellnessStandby(error.message);
            showToast("环境数据连接失败", `${error.message}；可稍后刷新重试`, "danger");
        } finally {
            window.clearInterval(progress);
            stopWellnessLoader();
            state.wellnessLoading = false;
        }
    }

    function startWellnessLoader(title) {
        const loader = $("#wellnessLoader");
        loader.classList.remove("hidden");
        $("#wellnessLoaderTitle").textContent = title;
        $$("span", $("#wellnessLoaderSteps")).forEach((step, index) => step.classList.toggle("active", index === 0));
    }

    function animateWellnessProgress() {
        let index = 0;
        const titles = ["校准位置坐标", "读取天气与空气", "匹配健康档案", "完成安全过滤"];
        return window.setInterval(() => {
            index = Math.min(index + 1, 3);
            $("#wellnessLoaderTitle").textContent = titles[index];
            $$("span", $("#wellnessLoaderSteps")).forEach((step, stepIndex) => {
                step.classList.toggle("active", stepIndex <= index);
            });
        }, 480);
    }

    function stopWellnessLoader() {
        $("#wellnessLoader").classList.add("hidden");
    }

    function setWellnessStandby(message) {
        $("#wellnessCondition").textContent = "等待环境信号";
        $("#wellnessFocus").textContent = message;
        $("#wellnessLiveBadge").className = "weather-live-badge waiting";
        $("#wellnessLiveBadge").textContent = "STANDBY";
    }

    function renderWellness(data) {
        const { weather, recommendation } = data;
        const current = weather.current;
        const today = weather.today;
        const provenance = weather.provenance;
        const stateLabels = { live: "LIVE", cache: "CACHE", stale: "STALE" };
        const locationParts = [weather.location.name, weather.location.admin1]
            .filter((value, index, values) => value && values.indexOf(value) === index);

        $("#wellnessLocationName").textContent = locationParts.join(" · ") || "当前位置";
        $("#wellnessCondition").textContent = current.condition;
        $("#wellnessFocus").textContent = recommendation.summary.focus;
        $("#wellnessTemperature").textContent = Math.round(current.temperature ?? 0);
        $("#wellnessFeels").textContent = `体感 ${Math.round(current.feels_like ?? current.temperature ?? 0)}°`;
        $("#wellnessHumidity").textContent = `${Math.round(current.humidity ?? 0)}%`;
        $("#wellnessWind").textContent = `${Number(current.wind_speed ?? 0).toFixed(1)} km/h`;
        $("#wellnessAqi").textContent = current.aqi === null ? "US AQI --" : `US AQI ${Math.round(current.aqi)} · ${current.aqi_label}`;
        $("#wellnessUv").textContent = `UV ${Number(today.uv_index_max ?? 0).toFixed(1)}`;
        $("#wellnessRain").textContent = `${Math.round(today.precipitation_probability_max ?? 0)}%`;
        $("#wellnessSource").textContent = "Open-Meteo · OpenStreetMap";
        $("#wellnessUpdated").textContent = `更新 ${formatDateTime(provenance.fetched_at)}`;
        $("#wellnessAccuracy").textContent = state.wellnessAccuracy
            ? `定位精度 ±${Math.round(state.wellnessAccuracy)} m`
            : "城市中心点";

        const liveBadge = $("#wellnessLiveBadge");
        liveBadge.className = `weather-live-badge ${provenance.state}`;
        liveBadge.textContent = stateLabels[provenance.state] || "DATA";
        const hero = $("#wellnessHero");
        hero.dataset.weather = current.weather_group || "clear";
        hero.dataset.day = current.is_day ? "true" : "false";

        $("#wellnessSummaryTitle").textContent = recommendation.summary.title;
        $("#wellnessSummaryText").textContent = `今日重点：${recommendation.summary.focus}`;
        $("#hydrationValue").textContent = recommendation.summary.hydration.target;
        $("#hydrationNote").textContent = recommendation.summary.hydration.note;
        const hydrationAmount = recommendation.summary.hydration.amount_ml;
        $("#hydrationMeter").style.width = hydrationAmount ? `${Math.min(100, hydrationAmount / 24)}%` : "52%";
        $("#playBriefing").disabled = false;
        $("#exportWellness").disabled = false;

        renderWellnessMeals(recommendation.meals, recommendation.excluded);
        renderWellnessTimeline(recommendation.timeline);
        renderWellnessRisks(recommendation.risks);
        renderWellnessProtections(recommendation.protections);
        renderWellnessDisclosure(recommendation, weather.sources);
        renderWellnessHourly(weather.hourly);
        updateWellnessScene(current.weather_group, current.is_day, current.aqi_level);
        refreshIcons($("[data-page-panel='wellness']"));
        window.setTimeout(() => {
            resizeCharts();
            resizeWellnessScene();
        }, 40);
    }

    function renderWellnessMeals(meals, excluded) {
        const cards = meals.map((meal) => {
            const card = createElement("article", `meal-card ${meal.tone}`);
            card.dataset.recipeId = meal.recipe_id;
            const media = createElement("div", "meal-media");
            const image = document.createElement("img");
            image.src = meal.image;
            image.alt = meal.image_alt || meal.title;
            image.loading = "lazy";
            media.appendChild(image);
            const slot = createElement("span", "meal-slot", `${meal.slot} · ${meal.time}`);
            media.appendChild(slot);
            media.appendChild(createElement("span", "meal-fit", `${meal.fit_score}% MATCH`));
            card.appendChild(media);
            const body = createElement("div", "meal-body");
            body.appendChild(createElement("span", "meal-energy", meal.energy));
            body.appendChild(createElement("h3", "", meal.title));
            body.appendChild(createElement("p", "", meal.description));
            const footer = createElement("div", "meal-card-footer");
            const meta = createElement("span", "meal-card-meta");
            meta.appendChild(iconElement("clock-3"));
            meta.appendChild(document.createTextNode(`${meal.duration_minutes} 分钟 · ${meal.difficulty}`));
            footer.appendChild(meta);
            const openButton = createElement("button", "meal-open-button");
            openButton.type = "button";
            openButton.setAttribute("aria-label", `查看${meal.title}详细做法`);
            openButton.appendChild(document.createTextNode("查看做法"));
            openButton.appendChild(iconElement("arrow-up-right"));
            openButton.addEventListener("click", () => {
                window.dispatchEvent(new CustomEvent("wellness:open-recipe", { detail: { meal } }));
            });
            footer.appendChild(openButton);
            body.appendChild(footer);
            card.appendChild(body);
            return card;
        });
        $("#wellnessMealGrid").replaceChildren(...cards);

        const exclusions = $("#wellnessExclusions");
        exclusions.classList.toggle("hidden", !excluded.length);
        exclusions.replaceChildren(...excluded.map((label) => {
            const chip = createElement("span", "");
            chip.appendChild(iconElement("shield-check"));
            chip.appendChild(document.createTextNode(`已过滤 ${label}`));
            return chip;
        }));
    }

    function renderWellnessTimeline(items) {
        const nodes = items.map((item, index) => {
            const node = createElement("article", "rhythm-node");
            node.style.setProperty("--node-index", index);
            const marker = createElement("div", "rhythm-marker");
            marker.appendChild(iconElement(item.icon));
            node.appendChild(marker);
            const copy = createElement("div", "rhythm-copy");
            const meta = createElement("div", "rhythm-meta");
            meta.appendChild(createElement("time", "tabular", item.time));
            meta.appendChild(createElement("span", "", item.phase));
            copy.appendChild(meta);
            copy.appendChild(createElement("strong", "", item.title));
            copy.appendChild(createElement("p", "", item.detail));
            node.appendChild(copy);
            return node;
        });
        $("#wellnessTimeline").replaceChildren(...nodes);
    }

    function renderWellnessRisks(items) {
        const levelNames = { low: "低", guarded: "关注", elevated: "升高", high: "高" };
        const cards = items.map((item) => {
            const card = createElement("article", `risk-item ${item.level}`);
            const head = createElement("div", "risk-head");
            head.appendChild(iconElement(item.icon));
            const title = createElement("div", "");
            title.appendChild(createElement("span", "", item.label));
            title.appendChild(createElement("strong", "tabular", item.value));
            head.appendChild(title);
            head.appendChild(createElement("em", "", levelNames[item.level] || "关注"));
            card.appendChild(head);
            const meter = createElement("div", "risk-meter");
            const fill = createElement("span", "");
            fill.style.width = `${Math.max(4, Math.min(100, item.score))}%`;
            meter.appendChild(fill);
            card.appendChild(meter);
            card.appendChild(createElement("p", "", item.guidance));
            return card;
        });
        $("#wellnessRiskGrid").replaceChildren(...cards);
    }

    function renderWellnessProtections(items) {
        const cards = items.map((item, index) => {
            const card = createElement("article", `protection-item ${item.level}`);
            const indexLabel = createElement("span", "protection-index tabular", String(index + 1).padStart(2, "0"));
            card.appendChild(indexLabel);
            const icon = createElement("div", "protection-icon");
            icon.appendChild(iconElement(item.icon));
            card.appendChild(icon);
            const copy = createElement("div", "protection-copy");
            copy.appendChild(createElement("h3", "", item.title));
            const list = document.createElement("ul");
            item.actions.forEach((action) => list.appendChild(createElement("li", "", action)));
            copy.appendChild(list);
            card.appendChild(copy);
            return card;
        });
        $("#wellnessProtectionGrid").replaceChildren(...cards);
    }

    function renderWellnessDisclosure(recommendation, weatherSources) {
        const notices = $("#wellnessNotices");
        notices.replaceChildren(iconElement("shield-check"));
        const noticeCopy = createElement("div", "");
        recommendation.notices.forEach((notice) => noticeCopy.appendChild(createElement("span", "", notice)));
        notices.appendChild(noticeCopy);

        const sources = [...weatherSources, ...recommendation.references];
        $("#wellnessSources").replaceChildren(...sources.map((source) => {
            const link = document.createElement("a");
            link.href = source.url;
            link.target = "_blank";
            link.rel = "noopener noreferrer";
            link.textContent = source.name;
            return link;
        }));
    }

    function renderWellnessHourly(hourly) {
        const chart = chartInstance("wellnessHourlyChart");
        if (!chart) return;
        const labels = hourly.map((item) => String(item.time || "").slice(11, 16));
        const temperatures = hourly.map((item) => item.temperature);
        const precipitation = hourly.map((item) => item.precipitation_probability);
        chart.setOption({
            animationDuration: 700,
            color: ["#ffcf5c", "#58c7ed"],
            tooltip: { trigger: "axis", borderWidth: 0, backgroundColor: "rgba(12, 30, 45, .94)", textStyle: { color: "#fff", fontSize: 11 } },
            grid: { left: 42, right: 42, top: 28, bottom: 34 },
            xAxis: { type: "category", boundaryGap: false, data: labels, axisTick: { show: false }, axisLine: { lineStyle: { color: "#d9e4e8" } }, axisLabel: { color: "#71808d", fontSize: 10, interval: 3 } },
            yAxis: [
                { type: "value", axisLabel: { color: "#71808d", fontSize: 10, formatter: "{value}°" }, splitLine: { lineStyle: { color: "#edf2f4" } } },
                { type: "value", min: 0, max: 100, axisLabel: { color: "#71808d", fontSize: 10, formatter: "{value}%" }, splitLine: { show: false } },
            ],
            series: [
                { name: "温度", type: "line", data: temperatures, smooth: 0.35, symbol: "none", lineStyle: { width: 3, color: "#f09a3e" }, areaStyle: { color: "rgba(255, 207, 92, .22)" } },
                { name: "降水", type: "bar", yAxisIndex: 1, data: precipitation, barWidth: 6, itemStyle: { color: "#58c7ed", borderRadius: [3, 3, 0, 0] } },
            ],
        }, true);
    }

    async function searchWellnessCity(event) {
        event.preventDefault();
        const query = $("#citySearchInput").value.trim();
        if (query.length < 2) {
            showToast("请输入城市名称", "至少输入 2 个字符", "warning");
            return;
        }
        const resultsBox = $("#citySearchResults");
        resultsBox.classList.remove("hidden");
        resultsBox.replaceChildren(createElement("div", "city-result-loading", "正在搜索..."));
        try {
            const results = await apiRequest(`/api/wellness/locations?q=${encodeURIComponent(query)}`);
            if (!results.length) {
                resultsBox.replaceChildren(createElement("div", "city-result-loading", "未找到匹配城市"));
                return;
            }
            resultsBox.replaceChildren(...results.map((location) => {
                const button = createElement("button", "city-result");
                button.type = "button";
                button.setAttribute("role", "option");
                const copy = createElement("span", "");
                copy.appendChild(createElement("strong", "", location.name));
                copy.appendChild(createElement("small", "", [location.admin1, location.country].filter(Boolean).join(" · ")));
                button.appendChild(iconElement("map-pin"));
                button.appendChild(copy);
                button.addEventListener("click", () => selectWellnessCity(location));
                return button;
            }));
            refreshIcons(resultsBox);
        } catch (error) {
            resultsBox.replaceChildren(createElement("div", "city-result-loading danger-text", error.message));
        }
    }

    async function selectWellnessCity(location) {
        $("#citySearchInput").value = location.name;
        $("#citySearchResults").classList.add("hidden");
        state.wellnessAccuracy = null;
        const coordinates = {
            latitude: Number(location.latitude),
            longitude: Number(location.longitude),
            name: [location.name, location.admin1].filter(Boolean).join(" · "),
            accuracy: null,
        };
        state.wellnessCoordinates = coordinates;
        await fetchWellness(coordinates, true);
    }

    async function openWellnessProfile() {
        if (!state.wellnessProfile) {
            state.wellnessProfile = await apiRequest("/api/wellness/profile");
        }
        const profile = state.wellnessProfile;
        $("#profileAge").value = profile.age_group;
        $("#profileSex").value = profile.sex;
        $("#profileMedications").value = profile.medications || "";
        $("#profileRetainLocation").checked = Boolean(profile.retain_location);
        ["conditions", "allergies", "dietary_preferences", "health_goals"].forEach((name) => {
            const selected = new Set(profile[name] || []);
            $$(`input[name='${name}']`, $("#profileForm")).forEach((input) => {
                input.checked = selected.has(input.value);
            });
        });
        $("#profileDialog").showModal();
    }

    async function saveWellnessProfile(event) {
        event.preventDefault();
        const selected = (name) => $$(`input[name='${name}']:checked`, $("#profileForm")).map((input) => input.value);
        const payload = {
            age_group: $("#profileAge").value,
            sex: $("#profileSex").value,
            conditions: selected("conditions"),
            allergies: selected("allergies"),
            dietary_preferences: selected("dietary_preferences"),
            health_goals: selected("health_goals"),
            medications: $("#profileMedications").value.trim(),
            retain_location: $("#profileRetainLocation").checked,
        };
        const button = $("#saveProfile");
        button.disabled = true;
        try {
            state.wellnessProfile = await apiRequest("/api/wellness/profile", { method: "PUT", body: payload });
            if (!state.wellnessProfile.retain_location) {
                localStorage.removeItem("medpro-wellness-location");
            }
            $("#profileDialog").close();
            showToast("健康档案已更新", "推荐已按新的约束重新生成", "success");
            if (state.wellnessCoordinates) await fetchWellness(state.wellnessCoordinates, false);
        } catch (error) {
            showToast("档案保存失败", error.message, "danger");
        } finally {
            button.disabled = false;
        }
    }

    function playWellnessBriefing() {
        if (!state.wellness) return;
        if (!("speechSynthesis" in window)) {
            showToast("当前浏览器不支持语音播报", "请使用新版 Chrome 或 Edge", "warning");
            return;
        }
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(state.wellness.recommendation.summary.briefing);
        utterance.lang = "zh-CN";
        utterance.rate = 0.92;
        utterance.pitch = 1.02;
        window.speechSynthesis.speak(utterance);
        showToast("语音简报已开始", "再次点击可重新播报", "success");
    }

    function exportWellnessReport() {
        if (!state.wellness) return;
        const { weather, recommendation } = state.wellness;
        const canvas = document.createElement("canvas");
        canvas.width = 1080;
        canvas.height = 1440;
        const context = canvas.getContext("2d");
        context.fillStyle = "#0b2730";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = "#18b8a0";
        context.fillRect(0, 0, 24, canvas.height);
        context.fillStyle = "#ffffff";
        context.font = "700 34px Microsoft YaHei";
        context.fillText("MEDPRO · 天时智养", 78, 92);
        context.fillStyle = "#82e5d1";
        context.font = "500 22px Microsoft YaHei";
        context.fillText(new Date().toLocaleDateString("zh-CN"), 78, 132);
        context.fillStyle = "#ffffff";
        context.font = "700 76px Microsoft YaHei";
        context.fillText(weather.location.name, 78, 250);
        context.font = "700 152px Microsoft YaHei";
        context.fillText(`${Math.round(weather.current.temperature)}°`, 70, 430);
        context.fillStyle = "#ffcf5c";
        context.font = "700 38px Microsoft YaHei";
        context.fillText(weather.current.condition, 390, 360);
        context.fillStyle = "#b7d2d9";
        context.font = "500 24px Microsoft YaHei";
        context.fillText(`体感 ${Math.round(weather.current.feels_like)}°  ·  US AQI ${Math.round(weather.current.aqi ?? 0)}  ·  UV ${Number(weather.today.uv_index_max ?? 0).toFixed(1)}`, 390, 408);

        context.fillStyle = "#123943";
        context.fillRect(70, 505, 940, 150);
        context.fillStyle = "#65e2cb";
        context.font = "700 22px Microsoft YaHei";
        context.fillText("TODAY'S SIGNAL", 104, 553);
        context.fillStyle = "#ffffff";
        context.font = "700 34px Microsoft YaHei";
        drawWrappedText(context, recommendation.summary.title, 104, 605, 830, 48, 2);

        context.font = "700 26px Microsoft YaHei";
        context.fillStyle = "#ffcf5c";
        context.fillText("气候适配三餐", 78, 735);
        recommendation.meals.forEach((meal, index) => {
            const top = 780 + index * 130;
            const colors = ["#d95473", "#e7952d", "#2fb481"];
            context.fillStyle = colors[index];
            context.fillRect(78, top, 10, 92);
            context.fillStyle = "#82aab3";
            context.font = "500 20px Microsoft YaHei";
            context.fillText(`${meal.slot}  ${meal.time}`, 112, top + 25);
            context.fillStyle = "#ffffff";
            context.font = "700 29px Microsoft YaHei";
            context.fillText(meal.title.slice(0, 20), 112, top + 67);
        });

        context.fillStyle = "#59c8ef";
        context.font = "700 24px Microsoft YaHei";
        context.fillText(`今日补水  ${recommendation.summary.hydration.target}`, 78, 1220);
        context.fillStyle = "#91adb4";
        context.font = "500 19px Microsoft YaHei";
        drawWrappedText(context, recommendation.notices[0], 78, 1282, 900, 30, 3);
        context.fillStyle = "#56747b";
        context.font = "500 17px Microsoft YaHei";
        context.fillText("Weather: Open-Meteo · Guidance: deterministic rules · Not medical diagnosis", 78, 1380);

        canvas.toBlob((blob) => {
            if (!blob) return;
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = `MedPro-天时智养-${new Date().toISOString().slice(0, 10)}.png`;
            link.click();
            window.setTimeout(() => URL.revokeObjectURL(link.href), 1000);
        }, "image/png");
    }

    function drawWrappedText(context, text, x, y, maxWidth, lineHeight, maxLines) {
        const characters = Array.from(text || "");
        let line = "";
        let lineIndex = 0;
        characters.forEach((character, index) => {
            const testLine = line + character;
            if (context.measureText(testLine).width > maxWidth && line) {
                if (lineIndex < maxLines) context.fillText(line, x, y + lineIndex * lineHeight);
                line = character;
                lineIndex += 1;
            } else {
                line = testLine;
            }
            if (index === characters.length - 1 && lineIndex < maxLines) {
                context.fillText(line, x, y + lineIndex * lineHeight);
            }
        });
    }

    function initializeWellnessScene() {
        if (state.wellnessScene || !window.THREE) return;
        const canvas = $("#wellnessScene");
        try {
            const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: true });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x176b81);
            scene.fog = new THREE.FogExp2(0x176b81, 0.045);
            const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 100);
            camera.position.set(0, 4.2, 10.5);

            scene.add(new THREE.HemisphereLight(0xc8f5ff, 0x17333b, 2.3));
            const sunLight = new THREE.DirectionalLight(0xffe2a1, 3.4);
            sunLight.position.set(-5, 8, 4);
            scene.add(sunLight);

            const terrainGeometry = new THREE.PlaneGeometry(42, 28, 50, 32);
            const terrain = new THREE.Mesh(terrainGeometry, new THREE.MeshStandardMaterial({
                color: 0x2ac5b0, wireframe: true, transparent: true, opacity: 0.3, roughness: 0.8,
            }));
            terrain.rotation.x = -Math.PI / 2;
            terrain.position.y = -2.6;
            scene.add(terrain);

            const skyline = new THREE.Group();
            const palette = [0x26ad9b, 0x3e83dd, 0xf0a43c, 0xd75978, 0x7c68ae];
            for (let index = 0; index < 28; index += 1) {
                const width = 0.38 + (index % 4) * 0.12;
                const height = 0.7 + ((index * 7) % 13) * 0.18;
                const tower = new THREE.Mesh(
                    new THREE.BoxGeometry(width, height, width),
                    new THREE.MeshStandardMaterial({ color: palette[index % palette.length], roughness: 0.65, metalness: 0.1 })
                );
                tower.position.set((index - 13.5) * 0.72, -2.25 + height / 2, -2 - (index % 3) * 0.7);
                skyline.add(tower);
            }
            scene.add(skyline);

            const sun = new THREE.Mesh(
                new THREE.SphereGeometry(0.58, 32, 32),
                new THREE.MeshBasicMaterial({ color: 0xffd46b })
            );
            sun.position.set(-5.7, 3.8, -4);
            scene.add(sun);

            const clouds = new THREE.Group();
            for (let index = 0; index < 7; index += 1) {
                const cloud = new THREE.Group();
                for (let part = 0; part < 4; part += 1) {
                    const puff = new THREE.Mesh(
                        new THREE.SphereGeometry(0.5 + (part % 2) * 0.18, 16, 12),
                        new THREE.MeshLambertMaterial({ color: 0xe8f4f5, transparent: true, opacity: 0.72 })
                    );
                    puff.scale.y = 0.55;
                    puff.position.set(part * 0.65, Math.sin(part) * 0.18, 0);
                    cloud.add(puff);
                }
                cloud.position.set(-9 + index * 3.1, 1.8 + (index % 2) * 1.25, -2.8 - (index % 3));
                clouds.add(cloud);
            }
            scene.add(clouds);

            const weatherPositions = new Float32Array(900 * 3);
            for (let index = 0; index < 900; index += 1) {
                weatherPositions[index * 3] = (Math.sin(index * 12.9898) * 12) % 12;
                weatherPositions[index * 3 + 1] = ((index * 37) % 120) / 10 - 4;
                weatherPositions[index * 3 + 2] = -((index * 17) % 100) / 12;
            }
            const weatherGeometry = new THREE.BufferGeometry();
            weatherGeometry.setAttribute("position", new THREE.BufferAttribute(weatherPositions, 3));
            const weatherMaterial = new THREE.PointsMaterial({ color: 0x8ee4ff, size: 0.045, transparent: true, opacity: 0.9 });
            const particles = new THREE.Points(weatherGeometry, weatherMaterial);
            particles.visible = false;
            scene.add(particles);

            const starPositions = new Float32Array(260 * 3);
            for (let index = 0; index < 260; index += 1) {
                starPositions[index * 3] = (Math.sin(index * 4.71) * 14) % 14;
                starPositions[index * 3 + 1] = 1 + ((index * 29) % 70) / 10;
                starPositions[index * 3 + 2] = -3 - ((index * 11) % 70) / 10;
            }
            const starGeometry = new THREE.BufferGeometry();
            starGeometry.setAttribute("position", new THREE.BufferAttribute(starPositions, 3));
            const stars = new THREE.Points(starGeometry, new THREE.PointsMaterial({ color: 0xdcefff, size: 0.06, transparent: true, opacity: 0.9 }));
            stars.visible = false;
            scene.add(stars);

            const pointer = { x: 0, y: 0 };
            $("#wellnessHero").addEventListener("pointermove", (event) => {
                const rect = $("#wellnessHero").getBoundingClientRect();
                pointer.x = ((event.clientX - rect.left) / rect.width - 0.5) * 0.7;
                pointer.y = ((event.clientY - rect.top) / rect.height - 0.5) * 0.3;
            });
            $("#wellnessHero").addEventListener("pointerleave", () => { pointer.x = 0; pointer.y = 0; });

            state.wellnessScene = { renderer, scene, camera, terrain, skyline, sun, sunLight, clouds, particles, stars, pointer, mode: "clear" };
            resizeWellnessScene();
            const animate = (time) => {
                if (!state.wellnessScene) return;
                const model = state.wellnessScene;
                const positions = model.terrain.geometry.attributes.position;
                for (let index = 0; index < positions.count; index += 1) {
                    const x = positions.getX(index);
                    const y = positions.getY(index);
                    positions.setZ(index, Math.sin(x * 0.42 + time * 0.00035) * 0.18 + Math.cos(y * 0.31 + time * 0.00022) * 0.12);
                }
                positions.needsUpdate = true;
                model.clouds.children.forEach((cloud, index) => {
                    cloud.position.x += 0.0015 * (index + 1);
                    if (cloud.position.x > 12) cloud.position.x = -11;
                });
                if (model.particles.visible) {
                    const particlePositions = model.particles.geometry.attributes.position;
                    const snow = model.mode === "snow";
                    for (let index = 0; index < particlePositions.count; index += 1) {
                        let y = particlePositions.getY(index) - (snow ? 0.008 : 0.04);
                        if (y < -3.5) y = 7.5;
                        particlePositions.setY(index, y);
                        if (snow) particlePositions.setX(index, particlePositions.getX(index) + Math.sin(time * 0.001 + index) * 0.0015);
                    }
                    particlePositions.needsUpdate = true;
                }
                model.camera.position.x += (model.pointer.x - model.camera.position.x) * 0.025;
                model.camera.position.y += (4.2 - model.pointer.y - model.camera.position.y) * 0.025;
                model.camera.lookAt(0, -0.4, -2.5);
                model.renderer.render(model.scene, model.camera);
                state.wellnessSceneFrame = window.requestAnimationFrame(animate);
            };
            state.wellnessSceneFrame = window.requestAnimationFrame(animate);
        } catch (error) {
            console.warn("Weather scene unavailable", error);
            canvas.classList.add("scene-unavailable");
        }
    }

    function updateWellnessScene(mode = "clear", isDay = true, aqiLevel = "low") {
        const model = state.wellnessScene;
        if (!model) return;
        const colors = {
            clear: isDay ? 0x167b91 : 0x101b3e,
            cloud: isDay ? 0x3b7180 : 0x162640,
            rain: 0x274f66,
            storm: 0x282545,
            snow: 0x688997,
        };
        const color = colors[mode] || colors.clear;
        model.scene.background.setHex(color);
        model.scene.fog.color.setHex(color);
        model.mode = mode;
        model.sun.visible = mode === "clear" && isDay;
        model.sunLight.intensity = isDay ? (mode === "clear" ? 3.4 : 1.7) : 0.45;
        model.clouds.visible = mode !== "clear" || aqiLevel === "high";
        model.particles.visible = ["rain", "storm", "snow"].includes(mode);
        model.particles.material.color.setHex(mode === "snow" ? 0xffffff : 0x7ed9ff);
        model.particles.material.size = mode === "snow" ? 0.095 : 0.038;
        model.stars.visible = !isDay && !["rain", "storm"].includes(mode);
        model.terrain.material.color.setHex(mode === "storm" ? 0xa379db : mode === "snow" ? 0x82d8e8 : 0x2ac5b0);
    }

    function resizeWellnessScene() {
        const model = state.wellnessScene;
        const canvas = $("#wellnessScene");
        if (!model || !canvas) return;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        if (!width || !height) return;
        model.renderer.setSize(width, height, false);
        model.camera.aspect = width / height;
        model.camera.updateProjectionMatrix();
    }

    async function refreshActivePage() {
        const loader = PAGE_INFO[state.activePage]?.loader;
        if (!loader) return;
        const button = $("#refreshPage");
        button.classList.add("is-loading");
        try {
            if (state.activePage === "wellness" && state.wellnessCoordinates) {
                await fetchWellness(state.wellnessCoordinates, true);
            } else {
                await loader();
            }
        } catch (error) {
            showToast("刷新失败", error.message, "danger");
        } finally {
            button.classList.remove("is-loading");
        }
    }

    function bindEvents() {
        $$(".nav-item").forEach((item) => item.addEventListener("click", () => navigate(item.dataset.page)));
        document.addEventListener("click", (event) => {
            const navigateTarget = event.target.closest("[data-navigate]");
            if (navigateTarget) navigate(navigateTarget.dataset.navigate);

            const addTarget = event.target.closest("[data-action='add-medicine']");
            if (addTarget) openMedicineDialog();

            const editTarget = event.target.closest("[data-edit-medicine]");
            if (editTarget) {
                const medicine = state.medicines.find((item) => String(item.id) === editTarget.dataset.editMedicine);
                if (medicine) openMedicineDialog(medicine);
            }

            const deleteTarget = event.target.closest("[data-delete-medicine]");
            if (deleteTarget) deleteMedicine(deleteTarget.dataset.deleteMedicine);

            const ackTarget = event.target.closest("[data-ack-alert]");
            if (ackTarget) acknowledgeAlert(ackTarget.dataset.ackAlert);
        });

        $("#mobileMenu").addEventListener("click", () => document.body.classList.add("sidebar-open"));
        $("#sidebarClose").addEventListener("click", () => document.body.classList.remove("sidebar-open"));
        $("#sidebarScrim").addEventListener("click", () => document.body.classList.remove("sidebar-open"));
        $("#refreshPage").addEventListener("click", refreshActivePage);
        $("#refreshHealth").addEventListener("click", loadHealth);
        $("#refreshAlerts").addEventListener("click", loadAlertCenter);
        $("#refreshDetections").addEventListener("click", loadDetections);
        $("#alertLevelFilter").addEventListener("change", loadAlertCenter);
        $("#alertStatusFilter").addEventListener("change", loadAlertCenter);

        $("#wellnessLocate").addEventListener("click", () => requestBrowserLocation(false));
        $("#citySearchForm").addEventListener("submit", searchWellnessCity);
        $("#editProfile").addEventListener("click", openWellnessProfile);
        $("#profileForm").addEventListener("submit", saveWellnessProfile);
        $("#playBriefing").addEventListener("click", playWellnessBriefing);
        $("#exportWellness").addEventListener("click", exportWellnessReport);

        $$("#detectModeControl button").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode, true)));
        $("#settingsMode").addEventListener("change", (event) => setMode(event.target.value, true));
        $("#toggleCamera").addEventListener("click", () => state.cameraActive ? stopCamera() : startCamera());
        $("#videoFeed").addEventListener("load", () => {
            $("#cameraBadge").className = "status-badge success";
            $("#cameraBadge").replaceChildren(createElement("span", "status-dot success"), document.createTextNode("画面正常"));
        });
        $("#videoFeed").addEventListener("error", () => {
            stopCamera();
            showToast("摄像头连接失败", "请检查设备占用和权限", "warning");
        });

        $("#addMedicine").addEventListener("click", () => openMedicineDialog());
        $("#medicineSearch").addEventListener("input", renderMedicines);
        $("#medicineFilter").addEventListener("change", renderMedicines);
        $("#medicineForm").addEventListener("submit", saveMedicine);
        $$('[data-close-dialog]').forEach((button) => button.addEventListener("click", () => button.closest("dialog").close()));

        $("#chatForm").addEventListener("submit", sendQuestion);
        $("#clearChat").addEventListener("click", () => {
            $("#chatMessages").replaceChildren();
            initializeChat();
        });
        $$("[data-question]").forEach((button) => button.addEventListener("click", () => {
            $("#questionInput").value = button.dataset.question;
            $("#questionInput").focus();
        }));
        $("#questionInput").addEventListener("input", (event) => {
            event.target.style.height = "auto";
            event.target.style.height = `${Math.min(event.target.scrollHeight, 120)}px`;
        });
        $("#questionInput").addEventListener("keydown", (event) => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                $("#chatForm").requestSubmit();
            }
        });

        $("#compactMode").addEventListener("change", (event) => {
            document.body.classList.toggle("compact", event.target.checked);
            localStorage.setItem("medpro-compact", event.target.checked ? "1" : "0");
        });

        window.addEventListener("resize", resizeCharts);
        window.addEventListener("hashchange", () => {
            const route = window.location.hash.slice(1);
            if (route.startsWith("wellness/recipe/")) return;
            navigate(route, false);
        });
        document.addEventListener("visibilitychange", () => {
            if (document.hidden && state.cameraActive) stopCamera();
        });
        document.addEventListener("click", (event) => {
            if (!event.target.closest(".city-search")) $("#citySearchResults").classList.add("hidden");
        });
    }

    async function initialize() {
        refreshIcons();
        setPageMeta();
        bindEvents();
        initializeChat();
        const compact = localStorage.getItem("medpro-compact") === "1";
        $("#compactMode").checked = compact;
        document.body.classList.toggle("compact", compact);

        window.setInterval(() => {
            const now = new Date();
            $("#videoClock").textContent = now.toLocaleTimeString("zh-CN", { hour12: false });
        }, 1000);

        if (window.ResizeObserver) {
            const observer = new ResizeObserver(resizeCharts);
            $$(".chart-box").forEach((element) => observer.observe(element));
        }

        const initialRoute = window.location.hash.slice(1) || "overview";
        const initialPage = initialRoute.startsWith("wellness/recipe/") ? "wellness" : initialRoute;
        navigate(initialPage, false);
        updateAlertCount();
        state.refreshTimer = window.setInterval(() => {
            if (!document.hidden && ["overview", "detection", "alerts", "emotion"].includes(state.activePage)) {
                PAGE_INFO[state.activePage].loader().catch(() => {});
            }
        }, 15000);
    }

    initialize();
})();
