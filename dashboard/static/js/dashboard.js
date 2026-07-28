/**
 * 多模态智能医疗家庭助手·成果大屏 — 前端逻辑
 * 负责：ECharts 图表、API 轮询、聊天交互、导航切换
 */

/* ═══════════════════════════════════════════════════════════
   ECharts 初始化
   ═══════════════════════════════════════════════════════════ */
const pieChartDom = document.getElementById("pieChart");
const barChartDom = document.getElementById("barChart");

const pieChart = echarts.init(pieChartDom);
const barChart = echarts.init(barChartDom);

// 通用图表颜色
const CHART_COLORS = ["#1f7ae0", "#f39c12", "#27ae60", "#9b59b6", "#95a5a6"];

/* ═══════════════════════════════════════════════════════════
   API 请求封装
   ═══════════════════════════════════════════════════════════ */
async function apiGet(url) {
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        return json.data || json;
    } catch (err) {
        console.warn(`[API] GET ${url} 失败:`, err.message);
        return null;
    }
}

async function apiPost(url, body) {
    try {
        const res = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        const json = await res.json();
        return json;
    } catch (err) {
        console.warn(`[API] POST ${url} 失败:`, err.message);
        return { code: 500, message: "网络请求失败", data: null };
    }
}

/* ═══════════════════════════════════════════════════════════
   顶部概览数据
   ═══════════════════════════════════════════════════════════ */
async function loadOverview() {
    const data = await apiGet("/api/overview");
    if (!data) return;

    document.getElementById("todayCount").innerText = data.today_count ?? "--";
    document.getElementById("todayRate").innerText = data.today_count_rate ?? "--";
    document.getElementById("alertCount").innerText = data.alert_count ?? "--";
    document.getElementById("alertRate").innerText = data.alert_rate ?? "--";
    document.getElementById("llmStatus").innerText = data.llm_status ?? "--";
    document.getElementById("llmDesc").innerText = data.llm_desc ?? "--";

    // 日期时间
    document.getElementById("dateTime").innerText =
        `${data.date || "--"} ${data.time || "--"}`;

    // 天气
    const weatherEmoji = { "晴": "☀️", "多云": "⛅", "阴": "☁️", "雨": "🌧️", "雪": "❄️" };
    const w = weatherEmoji[data.weather] || "🌤️";
    document.getElementById("weatherInfo").innerText =
        `${w} ${data.weather || "--"}  ${data.temperature || "--"}  湿度 ${data.humidity || "--"}`;

    // LLM 状态指示灯（后端返回 "已连接" / "未连接"）
    const indicator = document.getElementById("llmIndicator");
    if (indicator) {
        indicator.className = "status-indicator " +
            (data.llm_status === "已连接" ? "online" : "offline");
    }
}

/* ═══════════════════════════════════════════════════════════
   药品类别饼图
   ═══════════════════════════════════════════════════════════ */
async function loadMedicineStats() {
    const data = await apiGet("/api/stats/medicine");
    if (!data || !data.length) return;

    pieChart.setOption({
        color: CHART_COLORS,
        tooltip: {
            trigger: "item",
            formatter: "{b}: {c} 次 ({d}%)",
            backgroundColor: "rgba(31,45,61,0.92)",
            borderColor: "transparent",
            textStyle: { color: "#fff", fontSize: 13 }
        },
        legend: {
            orient: "vertical",
            right: 8,
            top: "center",
            textStyle: { fontSize: 11, color: "#5c7086" },
            itemWidth: 10,
            itemHeight: 10,
            itemGap: 12
        },
        series: [{
            type: "pie",
            radius: ["42%", "72%"],
            center: ["32%", "50%"],
            avoidLabelOverlap: false,
            itemStyle: {
                borderRadius: 4,
                borderColor: "#fff",
                borderWidth: 3
            },
            label: { show: false },
            emphasis: {
                label: {
                    show: true,
                    fontSize: 16,
                    fontWeight: "bold"
                },
                scaleSize: 12
            },
            data: data
        }]
    });
}

/* ═══════════════════════════════════════════════════════════
   近7日趋势柱状图
   ═══════════════════════════════════════════════════════════ */
async function loadTrendStats() {
    const data = await apiGet("/api/stats/trend");
    if (!data || !data.dates) return;

    barChart.setOption({
        color: CHART_COLORS[0],
        tooltip: {
            trigger: "axis",
            backgroundColor: "rgba(31,45,61,0.92)",
            borderColor: "transparent",
            textStyle: { color: "#fff", fontSize: 13 },
            axisPointer: { type: "shadow" }
        },
        grid: {
            left: 40,
            right: 16,
            top: 16,
            bottom: 28
        },
        xAxis: {
            type: "category",
            data: data.dates,
            axisLine: { lineStyle: { color: "#d9e8fa" } },
            axisTick: { show: false },
            axisLabel: { color: "#8a9bad", fontSize: 10 },
            boundaryGap: true
        },
        yAxis: {
            type: "value",
            name: "次",
            nameTextStyle: { color: "#8a9bad", fontSize: 11 },
            axisLine: { show: false },
            axisTick: { show: false },
            splitLine: { lineStyle: { color: "#eef3fa", type: "dashed" } },
            axisLabel: { color: "#8a9bad", fontSize: 10 }
        },
        series: [{
            data: data.values,
            type: "bar",
            barWidth: "50%",
            itemStyle: {
                borderRadius: [4, 4, 0, 0],
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: "#4da3ff" },
                    { offset: 1, color: "#1f7ae0" }
                ])
            },
            emphasis: {
                itemStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: "#66b3ff" },
                        { offset: 1, color: "#2d8cff" }
                    ])
                }
            },
            animationDelay: function (idx) { return idx * 80; }
        }]
    });
}

/* ═══════════════════════════════════════════════════════════
   告警列表
   ═══════════════════════════════════════════════════════════ */
function getAlertIcon(level) {
    const map = { danger: "🔴", warning: "🟠", info: "🔵", success: "🟢" };
    return map[level] || "⚪";
}

async function loadAlerts() {
    const data = await apiGet("/api/alerts/latest");
    const container = document.getElementById("alertList");

    if (!data || !data.length) {
        container.innerHTML = '<div class="alert-empty">暂无告警记录</div>';
        return;
    }

    container.innerHTML = data.map(item => `
        <div class="alert-item">
            <span class="alert-icon">${getAlertIcon(item.level)}</span>
            <div class="alert-body">
                <h4>${item.title}</h4>
                <p>${item.content}</p>
            </div>
            <span class="alert-time">${item.time}</span>
        </div>
    `).join("");
}

/* ═══════════════════════════════════════════════════════════
   健康问答聊天
   ═══════════════════════════════════════════════════════════ */
function appendMessage(text, type, isLoading = false) {
    const box = document.getElementById("chatMessages");
    const div = document.createElement("div");

    const avatar = type === "user-msg" ? "👤" : "🤖";
    const cssClass = isLoading ? "chat-msg msg-loading" : `chat-msg ${type}`;

    div.className = cssClass;
    div.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-bubble">${escapeHtml(text)}</div>
    `;

    box.appendChild(div);
    box.scrollTop = box.scrollHeight;
    return div;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML.replace(/\n/g, "<br>");
}

async function sendQuestion() {
    const input = document.getElementById("questionInput");
    const btn = document.getElementById("sendBtn");
    const question = input.value.trim();

    if (!question) return;

    // 添加用户消息
    appendMessage(question, "user-msg");
    input.value = "";
    btn.disabled = true;

    // 加载提示
    const loadingMsg = appendMessage("正在思考中...", "bot-msg", true);

    const result = await apiPost("/api/chat", { question });

    // 移除加载提示
    loadingMsg.remove();

    if (result.code === 200 && result.data) {
        appendMessage(result.data.answer, "bot-msg");
    } else {
        appendMessage(result.message || "抱歉，暂时无法回答该问题。", "bot-msg");
    }

    btn.disabled = false;
    input.focus();
}

// 发送按钮
document.getElementById("sendBtn").addEventListener("click", sendQuestion);

// 回车发送
document.getElementById("questionInput").addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});

// 清空对话
document.getElementById("clearChat").addEventListener("click", function () {
    document.getElementById("chatMessages").innerHTML = `
        <div class="chat-msg bot-msg">
            <div class="msg-avatar">🤖</div>
            <div class="msg-bubble">对话已清空。您可以继续询问健康科普问题。</div>
        </div>
    `;
});

// 手动刷新告警
document.getElementById("refreshAlerts").addEventListener("click", loadAlerts);

/* ═══════════════════════════════════════════════════════════
   检测模式切换
   ═══════════════════════════════════════════════════════════ */
const modeTitles = {
    medicine: "📹 实时视频检测 — 药盒识别",
    emotion: "📹 实时视频检测 — 表情识别",
    both: "📹 实时视频检测 — 药盒 + 表情"
};

async function switchMode(mode) {
    // 更新按钮状态
    document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
    document.getElementById("btn" + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add("active");

    // 更新标题
    document.getElementById("videoTitle").innerText = modeTitles[mode];

    // 发送到后端
    try {
        await fetch("/api/mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: mode })
        });
    } catch (e) {
        console.warn("Mode switch failed:", e);
    }
}

document.querySelectorAll(".mode-btn").forEach(btn => {
    btn.addEventListener("click", function () {
        switchMode(this.dataset.mode);
    });
});

/* ═══════════════════════════════════════════════════════════
   导航菜单切换
   ═══════════════════════════════════════════════════════════ */
document.querySelectorAll(".menu").forEach(menu => {
    menu.addEventListener("click", function () {
        document.querySelectorAll(".menu").forEach(m => m.classList.remove("active"));
        this.classList.add("active");

        const page = this.dataset.page;
        console.log(`[导航] 切换到: ${page}`);

        // 第一版仅首页概览完整实现，其他菜单显示提示
        // 后续可扩展为多页面路由
    });
});

/* ═══════════════════════════════════════════════════════════
   图表自适应
   ═══════════════════════════════════════════════════════════ */
window.addEventListener("resize", function () {
    pieChart.resize();
    barChart.resize();
});

// ResizeObserver 监听图表容器变化
if (window.ResizeObserver) {
    const ro = new ResizeObserver(() => {
        pieChart.resize();
        barChart.resize();
    });
    ro.observe(pieChartDom);
    ro.observe(barChartDom);
}

/* ═══════════════════════════════════════════════════════════
   定时刷新
   ═══════════════════════════════════════════════════════════ */
const REFRESH_INTERVALS = {
    overview: 1000,    // 概览数据每秒刷新
    alerts: 5000,      // 告警每5秒刷新
    charts: 10000      // 图表每10秒刷新
};

let refreshTimers = [];

function startAutoRefresh() {
    // 初始化检测模式 — 读取默认选中按钮并同步到后端
    const activeBtn = document.querySelector(".mode-btn.active");
    if (activeBtn && activeBtn.dataset.mode) {
        switchMode(activeBtn.dataset.mode);
    }

    // 立即加载一次
    loadOverview();
    loadMedicineStats();
    loadTrendStats();
    loadAlerts();

    // 定时刷新
    refreshTimers.push(setInterval(loadOverview, REFRESH_INTERVALS.overview));
    refreshTimers.push(setInterval(loadAlerts, REFRESH_INTERVALS.alerts));
    refreshTimers.push(setInterval(loadMedicineStats, REFRESH_INTERVALS.charts));
    refreshTimers.push(setInterval(loadTrendStats, REFRESH_INTERVALS.charts));
}

function stopAutoRefresh() {
    refreshTimers.forEach(clearInterval);
    refreshTimers = [];
}

// 页面隐藏时暂停刷新，可见时恢复
document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
        stopAutoRefresh();
    } else {
        startAutoRefresh();
    }
});

/* ═══════════════════════════════════════════════════════════
   启动
   ═══════════════════════════════════════════════════════════ */
console.log("🚀 多模态智能医疗家庭助手·成果大屏 初始化中...");
startAutoRefresh();
console.log("✅ 大屏初始化完成");
