(() => {
    "use strict";

    const state = {
        recipe: null,
        meal: null,
        servings: 2,
        originalServings: 2,
        nutritionChart: null,
        kitchenIndex: 0,
        timerId: null,
        timerRemaining: 0,
        timerRunning: false,
        wakeLock: null,
        wakeDesired: false,
        lastFocus: null,
    };

    const $ = (selector, root = document) => root.querySelector(selector);
    const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

    function createElement(tag, className = "", text = "") {
        const element = document.createElement(tag);
        if (className) element.className = className;
        if (text !== "") element.textContent = text;
        return element;
    }

    function iconElement(name) {
        const icon = document.createElement("i");
        icon.dataset.lucide = name;
        return icon;
    }

    function refreshIcons(root = document) {
        if (window.lucide) window.lucide.createIcons({ attrs: { "stroke-width": 1.8 }, root });
    }

    async function apiRequest(url) {
        const response = await fetch(url, { headers: { Accept: "application/json" } });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok || payload.code >= 400) throw new Error(payload.message || "食谱加载失败");
        return payload.data;
    }

    function recipeRoute(recipeId) {
        return `wellness/recipe/${encodeURIComponent(recipeId)}`;
    }

    function routeRecipeId() {
        const route = window.location.hash.slice(1);
        if (!route.startsWith("wellness/recipe/")) return "";
        return decodeURIComponent(route.split("/").pop() || "");
    }

    async function openRecipe(meal, { fromRoute = false } = {}) {
        const recipeId = meal?.recipe_id;
        if (!recipeId) return;
        state.meal = meal;
        state.lastFocus = document.activeElement;

        const experience = $("#recipeExperience");
        experience.hidden = false;
        experience.setAttribute("aria-hidden", "false");
        experience.classList.remove("is-scrolled");
        experience.scrollTop = 0;
        document.body.classList.add("recipe-open");
        $(".app-shell")?.setAttribute("inert", "");
        $("#recipeLoading").classList.remove("hidden");
        $("#recipeView").classList.add("hidden");

        if (!fromRoute && window.location.hash !== `#${recipeRoute(recipeId)}`) {
            window.history.pushState({ wellnessRecipe: recipeId }, "", `#${recipeRoute(recipeId)}`);
        }

        try {
            const recipe = await apiRequest(`/api/wellness/recipes/${encodeURIComponent(recipeId)}`);
            if (routeRecipeId() && routeRecipeId() !== recipeId) return;
            state.recipe = recipe;
            state.originalServings = recipe.servings;
            state.servings = recipe.servings;
            renderRecipe();
            $("#recipeLoading").classList.add("hidden");
            $("#recipeView").classList.remove("hidden");
            $("#closeRecipe").focus({ preventScroll: true });
        } catch (error) {
            $("#recipeLoading strong").textContent = "暂时无法打开这份食谱";
            $("#recipeLoading small").textContent = error.message;
            window.setTimeout(() => requestCloseRecipe(), 1800);
        }
    }

    function renderRecipe() {
        const recipe = state.recipe;
        const match = recipe.profile_match;
        const experience = $("#recipeExperience");
        experience.classList.remove("theme-berry", "theme-citrus", "theme-leaf");
        experience.classList.add(`theme-${recipe.theme}`);

        $("#recipeHeroImage").src = recipe.image;
        $("#recipeHeroImage").alt = recipe.image_alt;
        $("#recipeSlot").textContent = `${recipe.slot} · 今日推荐`;
        $("#recipeMatchLabel").textContent = match.label;
        $("#recipeMatchLabel").className = match.status === "eligible" ? "" : `is-${match.status}`;
        $("#recipeTitle").textContent = recipe.title;
        $("#recipeSubtitle").textContent = recipe.subtitle;
        $("#recipeDuration").textContent = `${recipe.prep_minutes + recipe.cook_minutes} 分钟`;
        $("#recipeDifficulty").textContent = recipe.difficulty;
        $("#recipeCalories").textContent = `${recipe.nutrition.calories_kcal} kcal / 份`;
        $("#recipeFitScore").textContent = match.score;
        $("#recipeFitOrbit").style.setProperty("--fit", `${match.score}%`);
        $("#recipeImageNote").textContent = recipe.image_note;

        renderEvidence();
        renderProfileMatch();
        renderServings();
        renderIngredients();
        renderSteps();
        renderNutritionMetrics();
        renderSwaps();

        $("#recipeReviewStatus").textContent = `${recipe.source.status} · ${recipe.source.reviewed_on}`;
        $("#recipeMedicalNotice").textContent = recipe.medical_notice;
        $("#recipeSourceLink").href = recipe.source.url;
        $("#nutritionNote").textContent = recipe.nutrition_note;
        $("#kitchenRecipeTitle").textContent = recipe.title;
        $("#kitchenBackdrop").src = recipe.image;
        $("#kitchenBackdrop").alt = recipe.image_alt;
        $("#launchKitchen").disabled = !match.eligible;
        $("#launchKitchen").title = match.eligible ? "进入逐步烹饪模式" : "当前食谱与健康档案冲突，无法启动";
        activateTab("method");
        refreshIcons(experience);
    }

    function renderEvidence() {
        const evidence = state.meal?.evidence?.length ? state.meal.evidence : [
            { label: "今日选择", icon: "cloud-sun", detail: "这份食谱来自今日气候适配三餐，环境详情需回到实时推荐查看。" },
            { label: "健康档案", icon: "shield-check", detail: `已核对 ${state.recipe.profile_match.checked_fields.join("、")}。` },
            { label: "内容来源", icon: "badge-check", detail: `食谱由${state.recipe.source.name}维护，营养数值均标记为估算。` },
        ];
        const nodes = evidence.map((item, index) => {
            const article = createElement("article", "recipe-evidence-item");
            article.appendChild(createElement("small", "", `0${index + 1}`));
            const head = createElement("div", "recipe-evidence-head");
            const iconWrap = createElement("span");
            iconWrap.appendChild(iconElement(item.icon || "sparkles"));
            head.append(iconWrap, createElement("strong", "", item.label));
            article.append(head, createElement("p", "", item.detail));
            return article;
        });
        $("#recipeEvidenceFlow").replaceChildren(...nodes);
    }

    function renderProfileMatch() {
        const match = state.recipe.profile_match;
        const band = $("#recipeProfileBand");
        band.className = `recipe-profile-band ${match.status === "eligible" ? "" : match.status}`.trim();
        $("#recipeProfileTitle").textContent = match.label;
        const details = match.blockers.length ? match.blockers : [...match.warnings, ...match.adjustments];
        $("#recipeProfileText").textContent = details.length
            ? details.slice(0, 3).join("；")
            : `已核对 ${match.checked_fields.join("、")}，未发现已记录的冲突。`;
        const icon = match.status === "ineligible" ? "shield-alert" : match.status === "caution" ? "triangle-alert" : "shield-check";
        const wrap = $(".recipe-profile-icon", band);
        wrap.replaceChildren(iconElement(icon));
    }

    function renderServings() {
        $("#servingCount").textContent = state.servings;
        $("#servingMinus").disabled = state.servings <= 1;
        $("#servingPlus").disabled = state.servings >= 8;
    }

    function amountText(ingredient) {
        const scaled = ingredient.amount * state.servings / state.originalServings;
        const rounded = Math.abs(scaled - Math.round(scaled)) < 0.01 ? Math.round(scaled) : Number(scaled.toFixed(1));
        return `${rounded} ${ingredient.unit}`;
    }

    function checklistKey() {
        return `medpro-recipe-list:${state.recipe.id}`;
    }

    function readChecklist() {
        try {
            const value = JSON.parse(localStorage.getItem(checklistKey()) || "[]");
            return new Set(Array.isArray(value) ? value : []);
        } catch (_error) {
            return new Set();
        }
    }

    function writeChecklist(checked) {
        try { localStorage.setItem(checklistKey(), JSON.stringify([...checked])); } catch (_error) { /* Storage can be unavailable. */ }
    }

    function renderIngredients() {
        const checked = readChecklist();
        const rows = state.recipe.ingredients.map((ingredient) => {
            const label = createElement("label", `ingredient-row ${checked.has(ingredient.id) ? "checked" : ""}`.trim());
            const input = document.createElement("input");
            input.type = "checkbox";
            input.checked = checked.has(ingredient.id);
            input.dataset.ingredientId = ingredient.id;
            input.addEventListener("change", () => {
                const current = readChecklist();
                if (input.checked) current.add(ingredient.id);
                else current.delete(ingredient.id);
                writeChecklist(current);
                label.classList.toggle("checked", input.checked);
                updateIngredientProgress();
            });
            const copy = createElement("span");
            copy.append(createElement("strong", "", ingredient.name), createElement("small", "", ingredient.note));
            label.append(input, copy, createElement("b", "tabular", amountText(ingredient)));
            return label;
        });
        $("#ingredientList").replaceChildren(...rows);
        $("#ingredientTotal").textContent = state.recipe.ingredients.length;
        updateIngredientProgress();
    }

    function updateIngredientProgress() {
        $("#ingredientDone").textContent = $$("#ingredientList input:checked").length;
    }

    function renderSteps() {
        const rows = state.recipe.steps.map((step, index) => {
            const item = createElement("li", "recipe-step-item");
            const copy = createElement("div", "recipe-step-copy");
            copy.append(createElement("h3", "", step.title), createElement("p", "", step.body));
            const meta = createElement("div", "recipe-step-meta");
            const heat = createElement("span");
            heat.append(iconElement("flame"), document.createTextNode(step.heat));
            meta.appendChild(heat);
            if (step.timer_seconds) {
                const duration = createElement("span");
                duration.append(iconElement("timer"), document.createTextNode(formatDuration(step.timer_seconds)));
                meta.appendChild(duration);
            }
            copy.append(meta, createElement("div", "recipe-step-tip", step.tip));
            item.appendChild(copy);
            if (step.timer_seconds) {
                const timer = createElement("button", "step-timer-button");
                timer.type = "button";
                timer.title = `在厨房模式中启动${step.title}计时`;
                timer.setAttribute("aria-label", timer.title);
                timer.appendChild(iconElement("timer"));
                timer.addEventListener("click", () => openKitchen(index));
                item.appendChild(timer);
            } else {
                item.appendChild(createElement("span"));
            }
            return item;
        });
        $("#recipeStepList").replaceChildren(...rows);
    }

    function renderNutritionMetrics() {
        const nutrition = state.recipe.nutrition;
        const metrics = [
            ["能量", nutrition.calories_kcal, "kcal"],
            ["蛋白质", nutrition.protein_g, "g"],
            ["碳水化合物", nutrition.carbohydrates_g, "g"],
            ["脂肪", nutrition.fat_g, "g"],
            ["膳食纤维", nutrition.fiber_g, "g"],
            ["钠", nutrition.sodium_mg, "mg"],
        ];
        $("#nutritionMetrics").replaceChildren(...metrics.map(([label, value, unit]) => {
            const item = createElement("div", "nutrition-metric");
            item.append(createElement("span", "", label), createElement("strong", "tabular", value), createElement("small", "", `${unit} / 每份估算`));
            return item;
        }));
    }

    function renderNutritionChart() {
        if (!window.echarts || !state.recipe || $("[data-recipe-panel='nutrition']").hidden) return;
        if (state.nutritionChart) state.nutritionChart.dispose();
        const nutrition = state.recipe.nutrition;
        state.nutritionChart = window.echarts.init($("#recipeNutritionChart"), null, { renderer: "canvas" });
        state.nutritionChart.setOption({
            animationDuration: 650,
            color: ["#ef557c", "#f1aa2f", "#2aae84", "#765ed0"],
            tooltip: { trigger: "item", formatter: "{b}<br>{c} g · {d}%" },
            legend: { bottom: 0, itemHeight: 8, itemWidth: 8, textStyle: { color: "#66777c", fontSize: 10 } },
            series: [{
                name: "宏量营养素",
                type: "pie",
                radius: ["53%", "76%"],
                center: ["50%", "45%"],
                avoidLabelOverlap: true,
                label: { color: "#31474d", fontSize: 10, formatter: "{b}\n{c}g" },
                labelLine: { lineStyle: { color: "#aab6b8" } },
                data: [
                    { name: "蛋白质", value: nutrition.protein_g },
                    { name: "碳水", value: nutrition.carbohydrates_g },
                    { name: "脂肪", value: nutrition.fat_g },
                    { name: "膳食纤维", value: nutrition.fiber_g },
                ],
            }],
            graphic: [{ type: "text", left: "center", top: "39%", style: { text: `${nutrition.calories_kcal}\nkcal`, fill: "#17343b", fontSize: 22, fontWeight: 800, lineHeight: 28, textAlign: "center" } }],
        });
    }

    function renderSwaps() {
        const nodes = state.recipe.substitutions.map((swap) => {
            const item = createElement("article", "swap-item");
            const icon = createElement("span");
            icon.appendChild(iconElement("repeat-2"));
            const copy = createElement("div");
            copy.appendChild(createElement("h3", "", swap.title));
            const route = createElement("div", "swap-route");
            route.append(createElement("span", "", swap.replace), iconElement("arrow-right"), createElement("b", "", swap.with));
            copy.append(route, createElement("p", "", swap.note));
            item.append(icon, copy);
            return item;
        });
        $("#recipeSwapList").replaceChildren(...nodes);
    }

    function activateTab(name) {
        $$("[data-recipe-tab]").forEach((button) => {
            const active = button.dataset.recipeTab === name;
            button.classList.toggle("active", active);
            button.setAttribute("aria-selected", active ? "true" : "false");
        });
        $$("[data-recipe-panel]").forEach((panel) => {
            const active = panel.dataset.recipePanel === name;
            panel.hidden = !active;
            panel.classList.toggle("active", active);
        });
        if (name === "nutrition") window.setTimeout(renderNutritionChart, 30);
    }

    function formatDuration(seconds) {
        if (seconds < 60) return `${seconds} 秒`;
        const minutes = Math.round(seconds / 60);
        return `${minutes} 分钟`;
    }

    function speak(text) {
        if (!("speechSynthesis" in window)) return;
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = "zh-CN";
        utterance.rate = 0.92;
        window.speechSynthesis.speak(utterance);
    }

    function speakRecipeOverview() {
        if (!state.recipe) return;
        const match = state.recipe.profile_match;
        speak(`${state.recipe.title}。${state.recipe.subtitle}。预计需要${state.recipe.prep_minutes + state.recipe.cook_minutes}分钟。健康档案核对结果：${match.label}。`);
    }

    function openKitchen(index = 0) {
        if (!state.recipe?.profile_match.eligible) return;
        state.kitchenIndex = Math.max(0, Math.min(index, state.recipe.steps.length - 1));
        $("#kitchenMode").classList.remove("hidden");
        $("#kitchenMode").setAttribute("aria-hidden", "false");
        renderKitchenStep();
        state.wakeDesired = true;
        requestWakeLock();
        $("#closeKitchen").focus({ preventScroll: true });
    }

    function closeKitchen() {
        stopTimer();
        state.wakeDesired = false;
        releaseWakeLock();
        window.speechSynthesis?.cancel();
        $("#kitchenMode").classList.add("hidden");
        $("#kitchenMode").setAttribute("aria-hidden", "true");
        $("#launchKitchen").focus({ preventScroll: true });
    }

    function renderKitchenStep() {
        stopTimer();
        const step = state.recipe.steps[state.kitchenIndex];
        const current = state.kitchenIndex + 1;
        const total = state.recipe.steps.length;
        $("#kitchenStepCurrent").textContent = String(current).padStart(2, "0");
        $("#kitchenStepTotal").textContent = `/ ${String(total).padStart(2, "0")}`;
        $("#kitchenHeat").textContent = step.heat;
        $("#kitchenStepTitle").textContent = step.title;
        $("#kitchenStepBody").textContent = step.body;
        $("#kitchenStepTip").textContent = step.tip;
        $("#kitchenProgressBar").style.width = `${current / total * 100}%`;
        $("#kitchenPrevious").disabled = state.kitchenIndex === 0;
        $("#kitchenNext span").textContent = state.kitchenIndex === total - 1 ? "完成" : "下一步";
        $("#kitchenNext [data-lucide]")?.setAttribute("data-lucide", state.kitchenIndex === total - 1 ? "check" : "arrow-right");
        state.timerRemaining = step.timer_seconds || 0;
        $("#kitchenTimer").classList.toggle("hidden", !step.timer_seconds);
        updateTimerDisplay();
        refreshIcons($("#kitchenMode"));
    }

    function updateTimerDisplay() {
        const minutes = Math.floor(state.timerRemaining / 60);
        const seconds = state.timerRemaining % 60;
        $("#kitchenTimerValue").textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
        const button = $("#kitchenTimerToggle");
        button.querySelector("span").textContent = state.timerRunning ? "暂停计时" : state.timerRemaining === 0 ? "重新计时" : "开始计时";
        button.querySelector("[data-lucide]")?.setAttribute("data-lucide", state.timerRunning ? "pause" : "play");
        refreshIcons(button);
    }

    function toggleTimer() {
        if (state.timerRunning) {
            stopTimer(false);
            return;
        }
        const step = state.recipe.steps[state.kitchenIndex];
        if (state.timerRemaining <= 0) state.timerRemaining = step.timer_seconds;
        state.timerRunning = true;
        state.timerId = window.setInterval(() => {
            state.timerRemaining -= 1;
            if (state.timerRemaining <= 0) {
                state.timerRemaining = 0;
                stopTimer(false);
                speak(`${step.title}计时完成。`);
            }
            updateTimerDisplay();
        }, 1000);
        updateTimerDisplay();
    }

    function stopTimer(reset = true) {
        if (state.timerId) window.clearInterval(state.timerId);
        state.timerId = null;
        state.timerRunning = false;
        if (reset && state.recipe) state.timerRemaining = state.recipe.steps[state.kitchenIndex]?.timer_seconds || 0;
        if ($("#kitchenTimerValue")) updateTimerDisplay();
    }

    async function requestWakeLock() {
        if (!("wakeLock" in navigator) || !state.wakeDesired || document.hidden) {
            updateWakeButton();
            return;
        }
        try {
            state.wakeLock = await navigator.wakeLock.request("screen");
            state.wakeLock.addEventListener("release", () => {
                state.wakeLock = null;
                updateWakeButton();
            });
        } catch (_error) {
            state.wakeLock = null;
        }
        updateWakeButton();
    }

    async function releaseWakeLock() {
        if (state.wakeLock) await state.wakeLock.release().catch(() => {});
        state.wakeLock = null;
        updateWakeButton();
    }

    function toggleWakeLock() {
        state.wakeDesired = !state.wakeDesired;
        if (state.wakeDesired) requestWakeLock();
        else releaseWakeLock();
        updateWakeButton();
    }

    function updateWakeButton() {
        const button = $("#toggleWakeLock");
        if (!button) return;
        button.classList.toggle("active", state.wakeDesired && Boolean(state.wakeLock));
        button.title = state.wakeDesired && state.wakeLock ? "屏幕常亮已开启" : "保持屏幕常亮";
        button.setAttribute("aria-label", button.title);
    }

    function hideRecipe() {
        if ($("#kitchenMode") && !$("#kitchenMode").classList.contains("hidden")) closeKitchen();
        stopTimer();
        releaseWakeLock();
        window.speechSynthesis?.cancel();
        if (state.nutritionChart) state.nutritionChart.dispose();
        state.nutritionChart = null;
        const experience = $("#recipeExperience");
        experience.hidden = true;
        experience.setAttribute("aria-hidden", "true");
        experience.classList.remove("is-scrolled");
        document.body.classList.remove("recipe-open");
        $(".app-shell")?.removeAttribute("inert");
        state.recipe = null;
        state.meal = null;
        if (state.lastFocus?.isConnected) state.lastFocus.focus({ preventScroll: true });
    }

    function requestCloseRecipe() {
        const currentState = window.history.state;
        hideRecipe();
        if (routeRecipeId() && currentState?.wellnessRecipe) {
            window.history.back();
        } else if (routeRecipeId()) {
            window.history.replaceState(null, "", "#wellness");
        }
    }

    function handleRouteChange() {
        const recipeId = routeRecipeId();
        if (!recipeId) {
            if (!$("#recipeExperience").hidden) hideRecipe();
            return;
        }
        if (state.recipe?.id === recipeId || (!$("#recipeExperience").hidden && !state.recipe)) return;
        openRecipe({ recipe_id: recipeId }, { fromRoute: true });
    }

    function bindEvents() {
        window.addEventListener("wellness:open-recipe", (event) => openRecipe(event.detail.meal));
        window.addEventListener("popstate", handleRouteChange);
        window.addEventListener("hashchange", handleRouteChange);
        window.addEventListener("resize", () => state.nutritionChart?.resize());
        document.addEventListener("visibilitychange", () => {
            if (!document.hidden && state.wakeDesired && !$("#kitchenMode").classList.contains("hidden")) requestWakeLock();
        });
        $("#recipeExperience").addEventListener("scroll", (event) => {
            event.currentTarget.classList.toggle("is-scrolled", event.currentTarget.scrollTop > 90);
        }, { passive: true });

        $("#closeRecipe").addEventListener("click", requestCloseRecipe);
        $("#speakRecipe").addEventListener("click", speakRecipeOverview);
        $("#servingMinus").addEventListener("click", () => {
            state.servings = Math.max(1, state.servings - 1);
            renderServings();
            renderIngredients();
        });
        $("#servingPlus").addEventListener("click", () => {
            state.servings = Math.min(8, state.servings + 1);
            renderServings();
            renderIngredients();
        });
        $("#clearIngredients").addEventListener("click", () => {
            writeChecklist(new Set());
            renderIngredients();
        });
        $$("[data-recipe-tab]").forEach((button) => button.addEventListener("click", () => activateTab(button.dataset.recipeTab)));
        $("#recipeTabs").addEventListener("keydown", (event) => {
            if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
            event.preventDefault();
            const tabs = $$("[data-recipe-tab]");
            const current = tabs.indexOf(document.activeElement);
            const next = (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
            tabs[next].focus();
            activateTab(tabs[next].dataset.recipeTab);
        });

        $("#launchKitchen").addEventListener("click", () => openKitchen(0));
        $("#closeKitchen").addEventListener("click", closeKitchen);
        $("#toggleWakeLock").addEventListener("click", toggleWakeLock);
        $("#kitchenTimerToggle").addEventListener("click", toggleTimer);
        $("#kitchenSpeak").addEventListener("click", () => {
            const step = state.recipe.steps[state.kitchenIndex];
            speak(`第${state.kitchenIndex + 1}步，${step.title}。${step.body}。提示：${step.tip}`);
        });
        $("#kitchenPrevious").addEventListener("click", () => {
            if (state.kitchenIndex > 0) {
                state.kitchenIndex -= 1;
                renderKitchenStep();
            }
        });
        $("#kitchenNext").addEventListener("click", () => {
            if (state.kitchenIndex >= state.recipe.steps.length - 1) closeKitchen();
            else {
                state.kitchenIndex += 1;
                renderKitchenStep();
            }
        });

        document.addEventListener("keydown", (event) => {
            if (event.key !== "Escape" || $("#recipeExperience").hidden) return;
            event.preventDefault();
            if (!$("#kitchenMode").classList.contains("hidden")) closeKitchen();
            else requestCloseRecipe();
        });
    }

    bindEvents();
    refreshIcons($("#recipeExperience"));
    if (routeRecipeId()) handleRouteChange();
})();
