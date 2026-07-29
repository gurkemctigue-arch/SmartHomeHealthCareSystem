(() => {
    "use strict";

    const MEMBER_COLORS = {
        mint: 0x55f2ba,
        cyan: 0x4ed8f2,
        coral: 0xff7468,
        amber: 0xf1c75b,
        violet: 0xb99cff
    };

    const STATUS_COLORS = {
        baseline: 0x60736d,
        good: 0x55f2ba,
        warning: 0xf1c75b,
        danger: 0xff7468
    };

    const SIGNAL_META = {
        mind: { icon: "brain", index: "NEURAL / 01" },
        cardio: { icon: "heart-pulse", index: "CARDIO / 02" },
        respiratory: { icon: "wind", index: "RESPIRATORY / 03" },
        metabolic: { icon: "flame", index: "METABOLIC / 04" },
        thermal: { icon: "thermometer", index: "THERMAL / 05" }
    };

    const METRIC_UI = {
        blood_pressure: { label: "收缩压", unit: "mmHg", min: 40, max: 300, step: 1, secondary: true },
        heart_rate: { label: "心率", unit: "bpm", min: 25, max: 250, step: 1 },
        spo2: { label: "血氧", unit: "%", min: 50, max: 100, step: 1 },
        temperature: { label: "体温", unit: "°C", min: 30, max: 45, step: 0.1 },
        blood_glucose: { label: "血糖", unit: "mmol/L", min: 1, max: 40, step: 0.1 },
        weight: { label: "体重", unit: "kg", min: 1, max: 400, step: 0.1 }
    };

    const METRIC_SIGNAL = {
        blood_pressure: "cardio",
        heart_rate: "cardio",
        spo2: "respiratory",
        temperature: "thermal",
        blood_glucose: "metabolic",
        weight: "metabolic"
    };

    const state = {
        bound: false,
        data: null,
        memberId: null,
        scene: null,
        sceneFrame: null,
        sceneRunning: false,
        selectedSignal: "thermal"
    };

    const $ = (selector, root = document) => root.querySelector(selector);
    const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

    function makeElement(tag, className = "", text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function icon(name) {
        const node = document.createElement("i");
        node.dataset.lucide = name;
        return node;
    }

    function refreshIcons(root = document) {
        if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.7 }, root });
    }

    async function apiRequest(url, options = {}) {
        const config = { ...options, headers: { Accept: "application/json", ...(options.headers || {}) } };
        if (config.body && typeof config.body !== "string") {
            config.headers["Content-Type"] = "application/json";
            config.body = JSON.stringify(config.body);
        }
        const response = await fetch(url, config);
        let payload;
        try {
            payload = await response.json();
        } catch (_error) {
            throw new Error(`服务返回异常 (${response.status})`);
        }
        if (!response.ok || payload.code >= 400) throw new Error(payload.message || `请求失败 (${response.status})`);
        return payload.data;
    }

    function showToast(title, message = "", type = "info") {
        const region = $("#toastRegion");
        if (!region) return;
        const toast = makeElement("div", `toast ${type}`);
        toast.appendChild(icon(type === "success" ? "circle-check" : type === "danger" ? "circle-alert" : type === "warning" ? "triangle-alert" : "info"));
        const copy = makeElement("div");
        copy.appendChild(makeElement("strong", "", title));
        if (message) copy.appendChild(makeElement("span", "", message));
        toast.appendChild(copy);
        region.appendChild(toast);
        refreshIcons(toast);
        window.setTimeout(() => toast.remove(), 3800);
    }

    function localInputValue(value = new Date()) {
        const adjusted = new Date(value.getTime() - value.getTimezoneOffset() * 60000);
        return adjusted.toISOString().slice(0, 16);
    }

    function formatEventTime(value) {
        if (!value) return "--";
        const parsed = new Date(value.includes("T") ? value : value.replace(" ", "T"));
        if (Number.isNaN(parsed.getTime())) return value;
        const today = new Date();
        const sameDay = parsed.toDateString() === today.toDateString();
        return new Intl.DateTimeFormat("zh-CN", sameDay
            ? { hour: "2-digit", minute: "2-digit", hour12: false }
            : { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }
        ).format(parsed);
    }

    async function load(memberId = state.memberId) {
        bindEvents();
        initializeScene();
        const requested = memberId || localStorage.getItem("medpro-vital-member") || "";
        const query = requested ? `?member_id=${encodeURIComponent(requested)}` : "";
        const data = await apiRequest(`/api/vital-twin/overview${query}`);
        state.data = data;
        state.memberId = data.member.id;
        localStorage.setItem("medpro-vital-member", String(state.memberId));
        render(data);
        applyMemberToScene(data.member);
        updateSceneSignals(data.signals);
        startScene();
        window.requestAnimationFrame(resizeScene);
        return data;
    }

    function render(data) {
        renderMembers(data.members, data.member.id);
        renderIdentity(data);
        renderSignals(data.signals);
        renderActions(data.actions);
        renderTimeline(data.timeline);
        populateMemberOptions(data.members);
        $("#vitalDisclaimer").textContent = data.disclaimer;
        $("#vitalSyncLabel").textContent = `LOCAL SYNC · ${formatEventTime(data.updated_at)}`;
        refreshIcons($("[data-page-panel='vital']"));
        focusSignal(state.selectedSignal, false);
    }

    function renderMembers(members, selectedId) {
        const list = $("#vitalMemberList");
        list.replaceChildren();
        members.forEach((member) => {
            const button = makeElement("button", `vital-member-chip${member.id === selectedId ? " active" : ""}`);
            button.type = "button";
            button.dataset.memberId = member.id;
            button.title = `${member.name} · ${member.relation_label}`;
            button.appendChild(makeElement("b", "", member.initial));
            button.appendChild(makeElement("span", "", member.name));
            list.appendChild(button);
        });
    }

    function renderIdentity(data) {
        const { member, score, coverage } = data;
        $("#vitalStage").dataset.memberColor = member.color || "mint";
        $("#vitalMemberRelation").textContent = member.relation_label;
        $("#vitalMemberAge").textContent = member.age === null ? "年龄未录入" : `${member.age} 岁`;
        $("#vitalMemberName").textContent = member.name;
        $("#vitalScore").textContent = score === null ? "--" : score;
        $("#vitalCoverage").textContent = `${coverage}%`;
        $("#vitalCoverageBar").style.width = `${coverage}%`;
        let context = "正在建立个人健康基线";
        if (score !== null && score >= 90) context = "今日已记录信号总体平稳";
        else if (score !== null && score >= 75) context = "存在需要复核的健康变化";
        else if (score !== null) context = "今日有重要信号需要优先关注";
        $("#vitalMemberContext").textContent = context;
    }

    function renderSignals(signals) {
        const deck = $("#vitalSignalDeck");
        deck.replaceChildren();
        signals.forEach((signal) => {
            const meta = SIGNAL_META[signal.key] || SIGNAL_META.thermal;
            const button = makeElement("button", `vital-signal-button ${signal.status}`);
            button.type = "button";
            button.dataset.signalKey = signal.key;
            const iconBox = makeElement("i");
            iconBox.appendChild(icon(meta.icon));
            const copy = makeElement("span");
            copy.appendChild(makeElement("b", "", signal.label));
            copy.appendChild(makeElement("small", "", signal.summary));
            button.append(iconBox, copy, makeElement("em"));
            deck.appendChild(button);
        });
    }

    function renderActions(actions) {
        const list = $("#vitalActionList");
        list.replaceChildren();
        if (!actions.length) {
            list.appendChild(makeElement("div", "vital-loading-line", "今日没有待处理事项"));
            return;
        }
        actions.forEach((action) => {
            const button = makeElement("button", `vital-action-item ${action.priority || "info"}`);
            button.type = "button";
            button.dataset.vitalAction = action.action;
            if (action.metric_type) button.dataset.metricType = action.metric_type;
            if (action.task_id) button.dataset.taskId = action.task_id;
            const iconBox = makeElement("i");
            iconBox.appendChild(icon(action.icon || "circle-dot"));
            const copy = makeElement("span");
            copy.appendChild(makeElement("b", "", action.title));
            copy.appendChild(makeElement("small", "", action.detail));
            button.append(iconBox, copy, icon(action.action === "complete" ? "check" : "chevron-right"));
            list.appendChild(button);
        });
    }

    function renderTimeline(events) {
        const track = $("#vitalTimelineTrack");
        $("#vitalTimelineCount").textContent = events.length;
        track.replaceChildren();
        if (!events.length) {
            track.appendChild(makeElement("div", "vital-timeline-empty", "第一条健康记录将从这里开始"));
            return;
        }
        events.forEach((event) => {
            const item = makeElement("button", `vital-timeline-event ${event.status || "baseline"}`);
            item.type = "button";
            if (event.metric_type) item.dataset.timelineMetric = event.metric_type;
            item.appendChild(makeElement("time", "", formatEventTime(event.event_at)));
            item.appendChild(makeElement("b", "", event.title));
            item.appendChild(makeElement("span", "", event.detail));
            track.appendChild(item);
        });
    }

    function populateMemberOptions(members) {
        [$("#vitalMeasurementMember"), $("#vitalTaskMember")].forEach((select) => {
            select.replaceChildren();
            members.forEach((member) => {
                const option = document.createElement("option");
                option.value = member.id;
                option.textContent = `${member.name} · ${member.relation_label}`;
                option.selected = member.id === state.memberId;
                select.appendChild(option);
            });
        });
    }

    function focusSignal(signalKey, animate = true) {
        const signal = state.data?.signals.find((item) => item.key === signalKey);
        if (!signal) return;
        state.selectedSignal = signalKey;
        $$(".vital-signal-button").forEach((button) => button.classList.toggle("active", button.dataset.signalKey === signalKey));
        const meta = SIGNAL_META[signalKey] || SIGNAL_META.thermal;
        $("#vitalZoneIndex").textContent = meta.index;
        $("#vitalZoneTitle").textContent = signal.label;
        $("#vitalZoneSummary").textContent = signal.summary;
        $("#vitalStage").dataset.focusZone = signal.region;
        if (state.scene) setSceneFocus(signalKey, animate);
    }

    function navigateTo(page) {
        const target = $(`.nav-item[data-page='${page}']`);
        if (target) target.click();
    }

    function openMemberDialog(member = null) {
        const form = $("#vitalMemberForm");
        form.reset();
        $("#vitalMemberId").value = member?.id || "";
        $("#vitalMemberDialogTitle").textContent = member ? "编辑成员档案" : "新增家庭成员";
        $("#vitalMemberInputName").value = member?.name || "";
        $("#vitalMemberRelationInput").value = member?.relation || "other";
        $("#vitalMemberSex").value = member?.sex || "unspecified";
        $("#vitalMemberBirthDate").value = member?.birth_date || "";
        $("#vitalMemberHeight").value = member?.height_cm || "";
        $("#vitalMemberConditions").value = (member?.conditions || []).join("，");
        $("#vitalMemberAllergies").value = (member?.allergies || []).join("，");
        const color = member?.color || "mint";
        const colorInput = $(`input[name='vitalMemberColor'][value='${color}']`);
        if (colorInput) colorInput.checked = true;
        $("#vitalDeleteMember").classList.toggle("hidden", !member || state.data.members.length <= 1);
        $("#vitalMemberDialog").showModal();
        window.setTimeout(() => $("#vitalMemberInputName").focus(), 30);
    }

    function openMeasurementDialog(metricType = "blood_pressure") {
        $("#vitalMeasurementForm").reset();
        populateMemberOptions(state.data.members);
        $("#vitalMeasurementMember").value = String(state.memberId);
        $("#vitalMeasurementType").value = metricType;
        $("#vitalMeasurementTime").value = localInputValue();
        updateMeasurementFields();
        $("#vitalMeasurementDialog").showModal();
        window.setTimeout(() => $("#vitalMeasurementPrimary").focus(), 30);
    }

    function openTaskDialog() {
        $("#vitalTaskForm").reset();
        populateMemberOptions(state.data.members);
        $("#vitalTaskMember").value = String(state.memberId);
        $("#vitalTaskDue").value = localInputValue(new Date(Date.now() + 60 * 60 * 1000));
        $("#vitalTaskDialog").showModal();
        window.setTimeout(() => $("#vitalTaskTitle").focus(), 30);
    }

    function updateMeasurementFields() {
        const type = $("#vitalMeasurementType").value;
        const meta = METRIC_UI[type];
        const primary = $("#vitalMeasurementPrimary");
        $("#vitalPrimaryLabel").textContent = meta.label;
        $("#vitalMeasurementUnit").textContent = meta.unit;
        primary.min = meta.min;
        primary.max = meta.max;
        primary.step = meta.step;
        $("#vitalSecondaryField").classList.toggle("hidden", !meta.secondary);
        $("#vitalMeasurementSecondary").required = Boolean(meta.secondary);
    }

    function splitList(value) {
        return value.split(/[,，]/).map((item) => item.trim()).filter(Boolean);
    }

    async function saveMember(event) {
        event.preventDefault();
        const id = $("#vitalMemberId").value;
        const color = $("input[name='vitalMemberColor']:checked")?.value || "mint";
        const payload = {
            name: $("#vitalMemberInputName").value.trim(),
            relation: $("#vitalMemberRelationInput").value,
            sex: $("#vitalMemberSex").value,
            birth_date: $("#vitalMemberBirthDate").value || null,
            height_cm: $("#vitalMemberHeight").value || null,
            conditions: splitList($("#vitalMemberConditions").value),
            allergies: splitList($("#vitalMemberAllergies").value),
            color
        };
        try {
            const member = await apiRequest(id ? `/api/vital-twin/members/${id}` : "/api/vital-twin/members", {
                method: id ? "PUT" : "POST", body: payload
            });
            $("#vitalMemberDialog").close();
            showToast(id ? "成员档案已更新" : "家庭成员已创建", member.name, "success");
            await load(member.id);
        } catch (error) {
            showToast("档案保存失败", error.message, "danger");
        }
    }

    async function saveMeasurement(event) {
        event.preventDefault();
        const type = $("#vitalMeasurementType").value;
        const payload = {
            member_id: Number($("#vitalMeasurementMember").value),
            metric_type: type,
            value_primary: $("#vitalMeasurementPrimary").value,
            value_secondary: METRIC_UI[type].secondary ? $("#vitalMeasurementSecondary").value : null,
            measured_at: $("#vitalMeasurementTime").value,
            note: $("#vitalMeasurementNote").value.trim()
        };
        try {
            const measurement = await apiRequest("/api/vital-twin/measurements", { method: "POST", body: payload });
            $("#vitalMeasurementDialog").close();
            showToast("生命体征已记录", `${measurement.label} ${measurement.display_value} ${measurement.unit}`, "success");
            await load(payload.member_id);
            focusSignal(METRIC_SIGNAL[type]);
        } catch (error) {
            showToast("记录失败", error.message, "danger");
        }
    }

    async function saveTask(event) {
        event.preventDefault();
        const payload = {
            member_id: Number($("#vitalTaskMember").value),
            title: $("#vitalTaskTitle").value.trim(),
            priority: $("#vitalTaskPriority").value,
            due_at: $("#vitalTaskDue").value,
            detail: $("#vitalTaskDetail").value.trim()
        };
        try {
            await apiRequest("/api/vital-twin/tasks", { method: "POST", body: payload });
            $("#vitalTaskDialog").close();
            showToast("照护任务已添加", payload.title, "success");
            await load(payload.member_id);
        } catch (error) {
            showToast("任务保存失败", error.message, "danger");
        }
    }

    function confirmAction(title, message) {
        const dialog = $("#confirmDialog");
        $("#confirmTitle").textContent = title;
        $("#confirmMessage").textContent = message;
        return new Promise((resolve) => {
            const onClose = () => {
                dialog.removeEventListener("close", onClose);
                resolve(dialog.returnValue === "confirm");
            };
            dialog.addEventListener("close", onClose);
            dialog.showModal();
        });
    }

    async function deleteCurrentMember() {
        const member = state.data.member;
        const confirmed = await confirmAction("删除家庭成员", `将删除 ${member.name} 的体征与照护记录，此操作无法撤销。`);
        if (!confirmed) return;
        try {
            await apiRequest(`/api/vital-twin/members/${member.id}`, { method: "DELETE" });
            $("#vitalMemberDialog").close();
            localStorage.removeItem("medpro-vital-member");
            showToast("成员已删除", member.name, "success");
            state.memberId = null;
            await load();
        } catch (error) {
            showToast("删除失败", error.message, "danger");
        }
    }

    async function handleAction(button) {
        const action = button.dataset.vitalAction;
        if (action === "record") openMeasurementDialog(button.dataset.metricType || "blood_pressure");
        else if (action === "medicine") navigateTo("medicine");
        else if (action === "alerts") navigateTo("alerts");
        else if (action === "complete") {
            try {
                await apiRequest(`/api/vital-twin/tasks/${button.dataset.taskId}/complete`, { method: "POST" });
                showToast("照护任务已完成", "健康轨迹已同步", "success");
                await load();
            } catch (error) {
                showToast("任务更新失败", error.message, "danger");
            }
        }
    }

    function bindEvents() {
        if (state.bound) return;
        state.bound = true;
        $("#vitalAddMember").addEventListener("click", () => openMemberDialog());
        $("#vitalEditMember").addEventListener("click", () => openMemberDialog(state.data.member));
        $("#vitalAddMeasurement").addEventListener("click", () => openMeasurementDialog());
        $("#vitalAddTask").addEventListener("click", openTaskDialog);
        $("#vitalResetView").addEventListener("click", resetSceneView);
        $("#vitalMeasurementType").addEventListener("change", updateMeasurementFields);
        $("#vitalMemberForm").addEventListener("submit", saveMember);
        $("#vitalMeasurementForm").addEventListener("submit", saveMeasurement);
        $("#vitalTaskForm").addEventListener("submit", saveTask);
        $("#vitalDeleteMember").addEventListener("click", deleteCurrentMember);
        $("#vitalMemberList").addEventListener("click", (event) => {
            const button = event.target.closest("[data-member-id]");
            if (button) load(Number(button.dataset.memberId)).catch((error) => showToast("成员切换失败", error.message, "danger"));
        });
        $("#vitalSignalDeck").addEventListener("click", (event) => {
            const button = event.target.closest("[data-signal-key]");
            if (button) focusSignal(button.dataset.signalKey);
        });
        $("#vitalActionList").addEventListener("click", (event) => {
            const button = event.target.closest("[data-vital-action]");
            if (button) handleAction(button);
        });
        $("#vitalTimelineTrack").addEventListener("click", (event) => {
            const item = event.target.closest("[data-timeline-metric]");
            if (item) focusSignal(METRIC_SIGNAL[item.dataset.timelineMetric]);
        });
        $("#vitalLayerSwitch").addEventListener("click", (event) => {
            const button = event.target.closest("[data-vital-layer]");
            if (!button) return;
            $$("#vitalLayerSwitch button").forEach((item) => item.classList.toggle("active", item === button));
            setSceneLayer(button.dataset.vitalLayer);
            if (button.dataset.vitalLayer !== "overview") focusSignal(button.dataset.vitalLayer);
        });
    }

    function initializeScene() {
        if (state.scene || !window.THREE) return;
        const THREE = window.THREE;
        const canvas = $("#vitalTwinCanvas");
        try {
            const renderer = new THREE.WebGLRenderer({
                canvas,
                antialias: true,
                alpha: true,
                powerPreference: "high-performance",
                preserveDrawingBuffer: true
            });
            renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8));
            renderer.outputColorSpace = THREE.SRGBColorSpace;
            renderer.setClearColor(0x080b0c, 0);

            const scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x080b0c, 0.035);
            const camera = new THREE.PerspectiveCamera(37, 1, 0.1, 40);
            camera.position.set(0, 0.2, 8.6);

            scene.add(new THREE.HemisphereLight(0xd9fff1, 0x08110e, 1.45));
            const keyLight = new THREE.DirectionalLight(0x8fffd8, 3.2);
            keyLight.position.set(3.5, 4.5, 5);
            scene.add(keyLight);
            const rimLight = new THREE.DirectionalLight(0xff7468, 1.5);
            rimLight.position.set(-4, 1, -3);
            scene.add(rimLight);

            const bodyRoot = new THREE.Group();
            bodyRoot.position.y = 0.05;
            scene.add(bodyRoot);

            const shellMaterial = new THREE.MeshPhysicalMaterial({
                color: MEMBER_COLORS.mint,
                emissive: 0x12392c,
                transparent: true,
                opacity: 0.26,
                roughness: 0.32,
                metalness: 0.28,
                clearcoat: 0.7,
                clearcoatRoughness: 0.28,
                depthWrite: false,
                side: THREE.DoubleSide
            });
            const edgeMaterial = new THREE.LineBasicMaterial({ color: MEMBER_COLORS.mint, transparent: true, opacity: 0.25 });
            const innerMaterial = new THREE.MeshStandardMaterial({ color: 0x9cd6c2, emissive: 0x17372d, transparent: true, opacity: 0.26, roughness: 0.5, depthWrite: false });
            const shellMeshes = [];
            const rayTargets = [];

            function addShell(geometry, position, scale = [1, 1, 1], rotation = [0, 0, 0], signal = null) {
                const mesh = new THREE.Mesh(geometry, shellMaterial);
                mesh.position.set(...position);
                mesh.scale.set(...scale);
                mesh.rotation.set(...rotation);
                if (signal) {
                    mesh.userData.signal = signal;
                    rayTargets.push(mesh);
                }
                const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, 24), edgeMaterial);
                mesh.add(edges);
                bodyRoot.add(mesh);
                shellMeshes.push(mesh);
                return mesh;
            }

            const torsoPoints = [
                new THREE.Vector2(0.42, 0), new THREE.Vector2(0.64, 0.18),
                new THREE.Vector2(0.76, 0.72), new THREE.Vector2(0.68, 1.28),
                new THREE.Vector2(0.43, 1.72)
            ];
            addShell(new THREE.LatheGeometry(torsoPoints, 48), [0, 0.27, 0], [1, 1, 0.58], [0, 0, 0], "cardio");
            addShell(new THREE.SphereGeometry(0.47, 32, 32), [0, 2.5, 0], [0.94, 1.12, 0.92], [0, 0, 0], "mind");
            addShell(new THREE.CylinderGeometry(0.2, 0.25, 0.38, 24), [0, 2.02, 0]);
            addShell(new THREE.SphereGeometry(0.58, 28, 22), [0, 0.16, 0], [1, 0.68, 0.72], [0, 0, 0], "metabolic");

            [-1, 1].forEach((side) => {
                addShell(new THREE.CylinderGeometry(0.18, 0.16, 1.25, 18), [side * 0.83, 1.28, 0], [1, 1, 1], [0, 0, side * -0.13], "thermal");
                addShell(new THREE.SphereGeometry(0.18, 18, 14), [side * 0.92, 0.64, 0]);
                addShell(new THREE.CylinderGeometry(0.15, 0.11, 1.18, 18), [side * 1.0, 0.08, 0], [1, 1, 1], [0, 0, side * -0.08], "thermal");
                addShell(new THREE.SphereGeometry(0.14, 18, 14), [side * 1.04, -0.55, 0], [0.75, 1.25, 0.65]);
                addShell(new THREE.CylinderGeometry(0.25, 0.2, 1.48, 20), [side * 0.32, -0.75, 0], [1, 1, 1], [0, 0, side * -0.035], "thermal");
                addShell(new THREE.SphereGeometry(0.23, 18, 14), [side * 0.35, -1.53, 0], [1, 0.78, 0.9]);
                addShell(new THREE.CylinderGeometry(0.19, 0.13, 1.25, 20), [side * 0.38, -2.16, 0.02], [1, 1, 1], [0, 0, side * -0.02], "thermal");
                addShell(new THREE.BoxGeometry(0.34, 0.2, 0.62), [side * 0.4, -2.82, 0.16], [1, 1, 1], [0, 0, 0]);
            });

            const spine = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, 3.5, 12), innerMaterial);
            spine.position.set(0, 0.18, -0.22);
            bodyRoot.add(spine);

            function organ(geometry, color, position, scale, signal) {
                const material = new THREE.MeshStandardMaterial({
                    color, emissive: color, emissiveIntensity: 0.42,
                    transparent: true, opacity: 0.78, roughness: 0.34, depthWrite: false
                });
                const mesh = new THREE.Mesh(geometry, material);
                mesh.position.set(...position);
                mesh.scale.set(...scale);
                mesh.userData.signal = signal;
                mesh.userData.baseScale = mesh.scale.clone();
                bodyRoot.add(mesh);
                rayTargets.push(mesh);
                return mesh;
            }

            const brain = organ(new THREE.IcosahedronGeometry(0.31, 2), 0xb99cff, [0, 2.52, 0.08], [1, 0.76, 0.78], "mind");
            const heart = organ(new THREE.DodecahedronGeometry(0.2, 1), 0xff7468, [0.16, 1.2, 0.38], [0.82, 1.05, 0.72], "cardio");
            const leftLung = organ(new THREE.SphereGeometry(0.29, 24, 20), 0x4ed8f2, [-0.28, 1.35, 0.2], [0.72, 1.25, 0.5], "respiratory");
            const rightLung = organ(new THREE.SphereGeometry(0.29, 24, 20), 0x4ed8f2, [0.29, 1.35, 0.2], [0.72, 1.25, 0.5], "respiratory");
            const metabolic = organ(new THREE.TorusKnotGeometry(0.2, 0.055, 72, 10), 0xf1c75b, [0, 0.42, 0.34], [1.05, 0.8, 0.65], "metabolic");

            const rings = {};
            const ringSpecs = {
                mind: [0.67, 2.5], cardio: [0.86, 1.23], respiratory: [0.78, 1.42],
                metabolic: [0.73, 0.36], thermal: [1.14, -0.5]
            };
            Object.entries(ringSpecs).forEach(([signal, [radius, y]]) => {
                const material = new THREE.MeshBasicMaterial({ color: STATUS_COLORS.baseline, transparent: true, opacity: 0.16, depthWrite: false });
                const mesh = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.012, 8, 96), material);
                mesh.position.set(0, y, 0);
                mesh.rotation.x = Math.PI / 2;
                bodyRoot.add(mesh);
                rings[signal] = mesh;
            });

            const scanMaterial = new THREE.LineBasicMaterial({ color: MEMBER_COLORS.mint, transparent: true, opacity: 0.38 });
            const scanGeometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(-1.35, 0, 0.78), new THREE.Vector3(1.35, 0, 0.78)
            ]);
            const scanLine = new THREE.Line(scanGeometry, scanMaterial);
            bodyRoot.add(scanLine);

            const grid = new THREE.GridHelper(8, 20, MEMBER_COLORS.mint, 0x173129);
            grid.position.y = -2.94;
            grid.material.transparent = true;
            grid.material.opacity = 0.13;
            scene.add(grid);

            const axisGeometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, -2.9, -0.7), new THREE.Vector3(0, 3.2, -0.7)
            ]);
            const axis = new THREE.Line(axisGeometry, new THREE.LineDashedMaterial({ color: 0x345a4d, dashSize: 0.06, gapSize: 0.09, transparent: true, opacity: 0.38 }));
            axis.computeLineDistances();
            scene.add(axis);

            state.scene = {
                THREE, renderer, scene, camera, bodyRoot, shellMaterial, edgeMaterial, scanMaterial,
                shellMeshes, rayTargets, rings, scanLine, grid, keyLight,
                organs: { mind: [brain], cardio: [heart], respiratory: [leftLung, rightLung], metabolic: [metabolic] },
                heart, lungs: [leftLung, rightLung], metabolic,
                selectedSignal: "thermal", layer: "overview", targetCameraY: 0.2,
                baseCameraZ: 8.6, drag: null, reducedMotion: window.matchMedia("(prefers-reduced-motion: reduce)").matches,
                raycaster: new THREE.Raycaster(), pointer: new THREE.Vector2()
            };
            bindSceneControls(canvas);
            if (window.ResizeObserver) {
                const observer = new ResizeObserver(resizeScene);
                observer.observe(canvas);
                state.scene.observer = observer;
            }
            resizeScene();
        } catch (error) {
            console.error("Vital Twin WebGL initialization failed", error);
            $("#vitalZoneTitle").textContent = "3D 渲染不可用";
            $("#vitalZoneSummary").textContent = "请确认浏览器已启用 WebGL 硬件加速";
        }
    }

    function bindSceneControls(canvas) {
        canvas.addEventListener("pointerdown", (event) => {
            if (!state.scene) return;
            canvas.setPointerCapture(event.pointerId);
            state.scene.drag = { x: event.clientX, y: event.clientY, rotation: state.scene.bodyRoot.rotation.y, moved: 0 };
        });
        canvas.addEventListener("pointermove", (event) => {
            const drag = state.scene?.drag;
            if (!drag) return;
            const dx = event.clientX - drag.x;
            const dy = event.clientY - drag.y;
            drag.moved = Math.max(drag.moved, Math.hypot(dx, dy));
            state.scene.bodyRoot.rotation.y = drag.rotation + dx * 0.008;
            state.scene.bodyRoot.rotation.x = Math.max(-0.08, Math.min(0.08, -dy * 0.0015));
        });
        const finishPointer = (event) => {
            const model = state.scene;
            if (!model?.drag) return;
            const wasClick = model.drag.moved < 6;
            model.drag = null;
            if (wasClick) pickBodySignal(event);
        };
        canvas.addEventListener("pointerup", finishPointer);
        canvas.addEventListener("pointercancel", () => { if (state.scene) state.scene.drag = null; });
        canvas.addEventListener("wheel", (event) => {
            if (!state.scene) return;
            event.preventDefault();
            state.scene.baseCameraZ = Math.max(7.2, Math.min(10.4, state.scene.baseCameraZ + event.deltaY * 0.004));
        }, { passive: false });
    }

    function pickBodySignal(event) {
        const model = state.scene;
        const rect = model.renderer.domElement.getBoundingClientRect();
        model.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        model.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        model.raycaster.setFromCamera(model.pointer, model.camera);
        const hit = model.raycaster.intersectObjects(model.rayTargets, false)[0];
        if (hit?.object.userData.signal) focusSignal(hit.object.userData.signal);
    }

    function applyMemberToScene(member) {
        if (!state.scene) return;
        const model = state.scene;
        const color = MEMBER_COLORS[member.color] || MEMBER_COLORS.mint;
        model.shellMaterial.color.setHex(color);
        model.shellMaterial.emissive.setHex(color);
        model.shellMaterial.emissive.multiplyScalar(0.16);
        model.edgeMaterial.color.setHex(color);
        model.scanMaterial.color.setHex(color);
        model.grid.material.color.setHex(color);
        model.keyLight.color.setHex(color);
    }

    function updateSceneSignals(signals) {
        if (!state.scene) return;
        const model = state.scene;
        signals.forEach((signal) => {
            const color = STATUS_COLORS[signal.status] || STATUS_COLORS.baseline;
            const ring = model.rings[signal.key];
            if (ring) ring.material.color.setHex(color);
            (model.organs[signal.key] || []).forEach((mesh) => {
                mesh.material.color.setHex(color);
                mesh.material.emissive.setHex(color);
            });
        });
    }

    function setSceneFocus(signalKey, animate = true) {
        const model = state.scene;
        if (!model) return;
        model.selectedSignal = signalKey;
        const focusY = { mind: 1.35, cardio: 0.65, respiratory: 0.75, metabolic: 0.15, thermal: 0.2 };
        model.targetCameraY = focusY[signalKey] ?? 0.2;
        Object.entries(model.rings).forEach(([key, ring]) => {
            ring.material.opacity = key === signalKey ? 0.82 : 0.13;
            const scale = key === signalKey ? 1.08 : 1;
            if (!animate) ring.scale.setScalar(scale);
            ring.userData.targetScale = scale;
        });
    }

    function setSceneLayer(layer) {
        const model = state.scene;
        if (!model) return;
        model.layer = layer;
        Object.entries(model.organs).forEach(([key, meshes]) => {
            meshes.forEach((mesh) => {
                mesh.visible = layer === "overview" || key === layer;
            });
        });
        model.shellMaterial.opacity = layer === "overview" ? 0.26 : 0.13;
    }

    function resetSceneView() {
        if (!state.scene) return;
        state.scene.bodyRoot.rotation.set(0, 0, 0);
        state.scene.baseCameraZ = window.innerWidth < 700 ? 9.4 : 8.6;
        state.scene.targetCameraY = 0.2;
        $$("#vitalLayerSwitch button").forEach((button) => button.classList.toggle("active", button.dataset.vitalLayer === "overview"));
        setSceneLayer("overview");
        focusSignal("thermal");
    }

    function resizeScene() {
        const model = state.scene;
        const canvas = $("#vitalTwinCanvas");
        if (!model || !canvas) return;
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        if (!width || !height) return;
        model.renderer.setSize(width, height, false);
        model.camera.aspect = width / height;
        model.camera.updateProjectionMatrix();
        const mobile = width < 700;
        model.bodyRoot.position.x = mobile ? 0.65 : width < 1100 ? 0.2 : 0.08;
        if (mobile && model.baseCameraZ < 9) model.baseCameraZ = 9.4;
    }

    function startScene() {
        if (!state.scene || state.sceneRunning) return;
        state.sceneRunning = true;
        const animate = (time) => {
            if (!state.sceneRunning || !state.scene) return;
            const model = state.scene;
            const seconds = time * 0.001;
            const motion = model.reducedMotion ? 0 : 1;
            model.bodyRoot.position.y = 0.05 + Math.sin(seconds * 1.25) * 0.014 * motion;
            model.camera.position.y += (model.targetCameraY - model.camera.position.y) * 0.045;
            model.camera.position.z += (model.baseCameraZ - model.camera.position.z) * 0.055;
            model.camera.lookAt(model.bodyRoot.position.x * 0.16, model.camera.position.y * 0.38, 0);

            const beat = 1 + Math.max(0, Math.sin(seconds * 5.6)) * 0.08 * motion;
            model.heart.scale.copy(model.heart.userData.baseScale).multiplyScalar(beat);
            const breath = 1 + Math.sin(seconds * 1.8) * 0.035 * motion;
            model.lungs.forEach((lung) => {
                lung.scale.copy(lung.userData.baseScale);
                lung.scale.x *= breath;
                lung.scale.y *= breath;
            });
            model.metabolic.rotation.x = seconds * 0.18 * motion;
            model.metabolic.rotation.y = seconds * 0.25 * motion;
            model.scanLine.position.y = -2.7 + ((seconds * 0.28) % 1) * 5.7;
            Object.values(model.rings).forEach((ring) => {
                const target = ring.userData.targetScale || 1;
                ring.scale.x += (target - ring.scale.x) * 0.08;
                ring.scale.y += (target - ring.scale.y) * 0.08;
                ring.scale.z += (target - ring.scale.z) * 0.08;
                ring.rotation.z += 0.0006 * motion;
            });
            model.renderer.render(model.scene, model.camera);
            state.sceneFrame = window.requestAnimationFrame(animate);
        };
        state.sceneFrame = window.requestAnimationFrame(animate);
    }

    function pause() {
        state.sceneRunning = false;
        if (state.sceneFrame) window.cancelAnimationFrame(state.sceneFrame);
        state.sceneFrame = null;
    }

    window.MedProVitalTwin = { load, pause, resize: resizeScene };
})();
