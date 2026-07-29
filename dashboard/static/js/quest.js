(() => {
    "use strict";

    const state = {
        bound: false,
        loaded: false,
        loading: null,
        overview: null,
        articles: [],
        categories: [],
        currentArticle: null,
        activeView: "mission",
        progressTimer: null,
        progressSent: 0,
        constellationFrame: null,
        gameSession: null,
        gameRuntime: null,
        gameIndex: 0,
        gameTimer: null,
        gameRemaining: 120,
        gameBusy: false,
    };

    const $ = (selector, root = document) => root.querySelector(selector);
    const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

    function element(tag, className = "", text) {
        const node = document.createElement(tag);
        if (className) node.className = className;
        if (text !== undefined) node.textContent = text;
        return node;
    }

    function icon(name, className = "") {
        const wrapper = element("span", className);
        const node = document.createElement("i");
        node.dataset.lucide = name;
        wrapper.appendChild(node);
        return wrapper;
    }

    function refreshIcons(root = document) {
        if (window.lucide) {
            window.lucide.createIcons({ attrs: { "stroke-width": 1.8 }, root });
        }
    }

    async function api(url, options = {}) {
        const config = { ...options };
        config.headers = { Accept: "application/json", ...(options.headers || {}) };
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
        if (!response.ok || Number(payload.code) >= 400) {
            throw new Error(payload.message || `请求失败 (${response.status})`);
        }
        return payload.data;
    }

    function notify(title, message = "", type = "info") {
        const region = $("#toastRegion");
        if (!region) return;
        const toast = element("div", `toast ${type}`);
        toast.appendChild(icon(type === "success" ? "circle-check" : type === "danger" ? "circle-alert" : "info"));
        const copy = element("div");
        copy.appendChild(element("strong", "", title));
        if (message) copy.appendChild(element("span", "", message));
        toast.appendChild(copy);
        region.appendChild(toast);
        refreshIcons(toast);
        window.setTimeout(() => toast.remove(), 4200);
    }

    async function load(force = false) {
        bindEvents();
        if (state.loading) return state.loading;
        if (state.loaded && !force) {
            startConstellation();
            drawKnowledgeMap();
            return;
        }
        state.loading = Promise.all([
            api("/api/quest/overview"),
            api("/api/quest/articles"),
        ]).then(([overview, articleData]) => {
            state.overview = overview;
            state.articles = articleData.items || [];
            state.categories = articleData.categories || [];
            renderOverview();
            renderArticleFilters();
            renderArticles();
            renderAtlas();
            startConstellation();
            drawKnowledgeMap();
            state.loaded = true;
        }).finally(() => {
            state.loading = null;
        });
        return state.loading;
    }

    function pause() {
        if (state.constellationFrame) {
            window.cancelAnimationFrame(state.constellationFrame);
            state.constellationFrame = null;
        }
        if (state.gameRuntime) state.gameRuntime.pause();
    }

    function bindEvents() {
        if (state.bound) return;
        state.bound = true;

        $("#questSwitcher").addEventListener("click", (event) => {
            const button = event.target.closest("[data-quest-view-target]");
            if (button) switchView(button.dataset.questViewTarget);
        });

        document.addEventListener("click", (event) => {
            const jump = event.target.closest("[data-quest-jump]");
            if (jump) switchView(jump.dataset.questJump);
            const articleButton = event.target.closest("[data-quest-article]");
            if (articleButton) openArticle(articleButton.dataset.questArticle);
        });

        let searchTimer = null;
        $("#questArticleSearch").addEventListener("input", () => {
            window.clearTimeout(searchTimer);
            searchTimer = window.setTimeout(renderArticles, 180);
        });
        $("#questArticleCategory").addEventListener("change", renderArticles);

        $("#closeQuestReader").addEventListener("click", closeArticle);
        $("#questReaderDialog").addEventListener("cancel", (event) => {
            event.preventDefault();
            closeArticle();
        });
        $("#questReaderScroll").addEventListener("scroll", trackReadingProgress, { passive: true });
        $("#questQuizForm").addEventListener("submit", submitQuiz);
        $("#questCommentForm").addEventListener("submit", submitComment);

        $("#questProfileButton").addEventListener("click", openProfiles);
        $("#closeQuestProfile").addEventListener("click", () => $("#questProfileDialog").close());
        $("#questProfileForm").addEventListener("submit", createProfile);

        $("#startHospitalGame").addEventListener("click", startGame);
        $("#restartHospitalGame").addEventListener("click", startGame);
        $("#gameFeedbackContinue").addEventListener("click", nextGameCase);
        $("#gameExit").addEventListener("click", finishGame);

        window.addEventListener("resize", () => {
            drawKnowledgeMap();
            if (state.gameRuntime) state.gameRuntime.resize();
        });
    }

    function switchView(view) {
        if (!$( `[data-quest-view="${view}"]` )) return;
        state.activeView = view;
        $$("#questSwitcher [data-quest-view-target]").forEach((button) => {
            const active = button.dataset.questViewTarget === view;
            button.classList.toggle("active", active);
            button.setAttribute("aria-selected", String(active));
        });
        $$("[data-quest-view]").forEach((panel) => {
            const active = panel.dataset.questView === view;
            panel.hidden = !active;
            panel.classList.toggle("active", active);
        });
        if (view === "mission") drawKnowledgeMap();
        if (view === "game" && state.gameRuntime) {
            state.gameRuntime.resize();
            state.gameRuntime.resume();
        } else if (state.gameRuntime) {
            state.gameRuntime.pause();
        }
        refreshIcons($( `[data-quest-view="${view}"]` ));
    }

    function renderOverview() {
        const { profile, stats, missions, achievements, recent_activity: recent } = state.overview;
        $("#questLevelLabel").textContent = `LEVEL ${String(profile.level).padStart(2, "0")}`;
        $("#questRankName").textContent = profile.level_name;
        $("#questNextLevel").textContent = profile.level_progress >= 100
            ? "最高等级信号已稳定"
            : `距离 ${profile.next_level_name} 还需 ${Math.max(0, profile.next_level_xp - profile.xp)} VP`;
        $("#questXpValue").textContent = profile.xp;
        $("#questXpOrbit").style.setProperty("--quest-progress", profile.level_progress);
        $("#questDisplayName").textContent = profile.display_name;
        $("#questAvatar").textContent = initials(profile.display_name);
        $("#questArticleStat").textContent = `${stats.articles_completed} / ${stats.articles_total}`;
        $("#questShiftStat").textContent = stats.game_shifts;
        $("#questStreakStat").textContent = profile.active_days;
        $("#questUnreadCount").textContent = Math.max(0, stats.articles_total - stats.articles_completed);

        const missionTotal = missions.reduce((sum, item) => sum + item.target, 0);
        const missionDone = missions.reduce((sum, item) => sum + item.current, 0);
        $("#questMissionSummary").textContent = `${missionDone} / ${missionTotal} 个行动节点已稳定`;
        $("#questMissionList").replaceChildren(...missions.map((mission) => {
            const item = element("div", `quest-mission-item ${mission.current >= mission.target ? "complete" : ""}`);
            item.appendChild(icon(mission.icon, "quest-mission-icon"));
            const copy = element("div");
            copy.appendChild(element("strong", "", mission.label));
            copy.appendChild(element("small", "", `${mission.current} / ${mission.target}`));
            item.appendChild(copy);
            const progress = element("div", "quest-mission-progress");
            const fill = element("span");
            fill.style.width = `${Math.min(100, mission.current / mission.target * 100)}%`;
            progress.appendChild(fill);
            item.appendChild(progress);
            return item;
        }));

        const unlocked = achievements.filter((item) => item.unlocked);
        $("#questAchievementCount").textContent = unlocked.length;
        if (unlocked.length) {
            const latest = unlocked.slice().sort((a, b) => String(b.unlocked_at).localeCompare(String(a.unlocked_at)))[0];
            const block = $("#questLatestUnlock");
            $("strong", block).textContent = latest.name;
            $("small", block).textContent = latest.description;
        }

        const nextArticle = state.articles.find((item) => !item.completed_at) || state.articles[0];
        if (nextArticle) {
            $("#questContinueTitle").textContent = nextArticle.title;
            $("#questContinueMeta").textContent = nextArticle.progress
                ? `${nextArticle.progress}% · 继续阅读`
                : `+5 VP · ${nextArticle.reading_minutes} 分钟`;
            $("#questContinueTitle").closest("button").dataset.questArticle = nextArticle.slug;
        }

        renderActivity(recent);
        refreshIcons($(".quest-page"));
    }

    function renderArticleFilters() {
        const select = $("#questArticleCategory");
        const previous = select.value || "all";
        const all = element("option", "", "全部领域");
        all.value = "all";
        const options = state.categories.map((category) => {
            const option = element("option", "", category);
            option.value = category;
            return option;
        });
        select.replaceChildren(all, ...options);
        select.value = state.categories.includes(previous) ? previous : "all";
    }

    function filteredArticles() {
        const query = $("#questArticleSearch").value.trim().toLowerCase();
        const category = $("#questArticleCategory").value;
        return state.articles.filter((article) => {
            const categoryMatch = category === "all" || article.category === category;
            const queryMatch = !query || `${article.title} ${article.dek} ${article.category}`.toLowerCase().includes(query);
            return categoryMatch && queryMatch;
        });
    }

    function renderArticles() {
        const items = filteredArticles();
        const grid = $("#questArticleGrid");
        if (!items.length) {
            grid.replaceChildren(element("div", "quest-loading-line", "没有匹配的知识档案"));
            return;
        }
        grid.replaceChildren(...items.map((article, index) => {
            const card = element("button", "quest-article-card");
            card.type = "button";
            card.dataset.questArticle = article.slug;
            const media = element("div", "quest-article-media");
            const image = document.createElement("img");
            image.src = article.cover_asset;
            image.alt = "";
            image.loading = "lazy";
            media.appendChild(image);
            media.appendChild(element("span", "quest-article-category", article.category));
            const status = element("span", "quest-article-state");
            status.appendChild(icon(article.completed_at ? "circle-check" : article.progress ? "bookmark" : "sparkles"));
            status.appendChild(document.createTextNode(article.completed_at ? "已完成" : article.progress ? `${article.progress}%` : "+15 VP"));
            media.appendChild(status);
            card.appendChild(media);
            const copy = element("div", "quest-article-copy");
            copy.appendChild(element("span", "", `CODEX / ${String(index + 1).padStart(3, "0")}`));
            copy.appendChild(element("h3", "", article.title));
            copy.appendChild(element("p", "", article.dek));
            const meta = element("div", "quest-article-meta");
            meta.appendChild(metaItem("clock-3", `${article.reading_minutes} 分钟`));
            meta.appendChild(metaItem("message-circle", `${article.comment_count} 讨论`));
            meta.appendChild(metaItem("badge-check", article.reviewer));
            copy.appendChild(meta);
            const track = element("div", "quest-card-progress");
            const fill = element("span");
            fill.style.width = `${article.progress || 0}%`;
            track.appendChild(fill);
            copy.appendChild(track);
            card.appendChild(copy);
            return card;
        }));
        refreshIcons(grid);
    }

    function metaItem(iconName, text) {
        const item = element("span");
        item.appendChild(icon(iconName));
        item.appendChild(document.createTextNode(text));
        return item;
    }

    async function openArticle(slug) {
        try {
            const article = await api(`/api/quest/articles/${encodeURIComponent(slug)}`);
            state.currentArticle = article;
            state.progressSent = article.progress || 0;
            renderArticle(article);
            const dialog = $("#questReaderDialog");
            dialog.showModal();
            $("#questReaderScroll").scrollTop = 0;
            document.body.classList.add("quest-reader-open");
        } catch (error) {
            notify("知识档案读取失败", error.message, "danger");
        }
    }

    function renderArticle(article) {
        $("#questReaderImage").src = article.cover_asset;
        $("#questReaderImage").alt = article.title;
        $("#questReaderCategory").textContent = article.category;
        $("#questReaderReview").textContent = `${article.reviewer} · ${article.reviewed_at}`;
        $("#questReaderSourceLink").href = article.source_url;
        $("#questReaderNumber").textContent = String(article.id).padStart(3, "0");
        $("#questReaderTitle").textContent = article.title;
        $("#questReaderDek").textContent = article.dek;
        $("#questReaderMinutes").textContent = `${article.reading_minutes} 分钟`;
        $("#questReaderReviewer").textContent = article.reviewer;
        $("#questReaderProgressBar").style.width = `${article.progress || 0}%`;

        $("#questReaderBody").replaceChildren(...article.body.map((section) => {
            const block = element("section", "quest-reader-section");
            block.appendChild(element("h2", "", section.heading));
            section.paragraphs.forEach((paragraph) => block.appendChild(element("p", "", paragraph)));
            return block;
        }));
        $("#questTakeawayList").replaceChildren(...article.takeaways.map((takeaway, index) => {
            const item = element("div", "quest-takeaway-item");
            item.appendChild(element("b", "", String(index + 1).padStart(2, "0")));
            item.appendChild(element("span", "", takeaway));
            return item;
        }));
        renderQuiz(article.quiz, article.best_quiz_score);
        renderComments(article.comments || []);
        refreshIcons($("#questReaderDialog"));
    }

    function closeArticle() {
        flushProgress();
        $("#questReaderDialog").close();
        document.body.classList.remove("quest-reader-open");
        state.currentArticle = null;
    }

    function trackReadingProgress() {
        if (!state.currentArticle) return;
        const scroll = $("#questReaderScroll");
        const available = Math.max(1, scroll.scrollHeight - scroll.clientHeight);
        const calculated = Math.min(100, Math.round(scroll.scrollTop / available * 100));
        const progress = Math.max(state.currentArticle.progress || 0, calculated);
        state.currentArticle.progress = progress;
        $("#questReaderProgressBar").style.width = `${progress}%`;
        if (progress - state.progressSent >= 10 || progress >= 90) {
            window.clearTimeout(state.progressTimer);
            state.progressTimer = window.setTimeout(flushProgress, 320);
        }
    }

    async function flushProgress() {
        window.clearTimeout(state.progressTimer);
        const article = state.currentArticle;
        if (!article || article.progress <= state.progressSent) return;
        const progress = article.progress;
        state.progressSent = progress;
        try {
            const result = await api(`/api/quest/articles/${encodeURIComponent(article.slug)}/progress`, {
                method: "POST",
                body: { progress },
            });
            if (result.points_awarded) notify("阅读完成", `+${result.points_awarded} Vital Points`, "success");
            handleUnlocks(result.unlocked);
            await refreshQuestData();
        } catch (error) {
            state.progressSent = Math.max(0, state.progressSent - 10);
            notify("阅读进度保存失败", error.message, "danger");
        }
    }

    function renderQuiz(questions, bestScore) {
        const container = $("#questQuizQuestions");
        container.replaceChildren(...questions.map((question, index) => {
            const fieldset = element("fieldset", "quest-quiz-question");
            const legend = element("h3", "", `${String(index + 1).padStart(2, "0")} · ${question.question}`);
            fieldset.appendChild(legend);
            const options = element("div", "quest-quiz-options");
            question.options.forEach((option, optionIndex) => {
                const label = document.createElement("label");
                const input = document.createElement("input");
                input.type = "radio";
                input.name = `quest-${question.index}`;
                input.value = String(optionIndex);
                input.required = true;
                label.appendChild(input);
                label.appendChild(element("span", "", option));
                options.appendChild(label);
            });
            fieldset.appendChild(options);
            return fieldset;
        }));
        const result = $("#questQuizResult");
        result.classList.toggle("hidden", !bestScore);
        result.textContent = bestScore ? `历史最佳 ${bestScore} 分` : "";
    }

    async function submitQuiz(event) {
        event.preventDefault();
        if (!state.currentArticle) return;
        const form = event.currentTarget;
        const answers = state.currentArticle.quiz.map((question) => {
            const checked = form.querySelector(`input[name='quest-${question.index}']:checked`);
            return checked ? Number(checked.value) : null;
        });
        if (answers.some((answer) => answer === null)) {
            notify("还有题目未完成", "完成全部判断后再提交", "warning");
            return;
        }
        const button = $("button[type='submit']", form);
        button.disabled = true;
        try {
            const result = await api(`/api/quest/articles/${encodeURIComponent(state.currentArticle.slug)}/quiz`, {
                method: "POST",
                body: { answers },
            });
            const resultBox = $("#questQuizResult");
            resultBox.classList.remove("hidden");
            resultBox.textContent = `${result.score} 分 · ${result.correct_count} / ${result.total} 项判断正确${result.points_awarded ? ` · +${result.points_awarded} VP` : ""}`;
            $$(".quest-quiz-question", form).forEach((question, index) => {
                question.dataset.result = result.correctness[index] ? "correct" : "wrong";
            });
            handleUnlocks(result.unlocked);
            await refreshQuestData();
        } catch (error) {
            notify("知识校验失败", error.message, "danger");
        } finally {
            button.disabled = false;
        }
    }

    async function submitComment(event) {
        event.preventDefault();
        if (!state.currentArticle) return;
        const input = $("#questCommentInput");
        const button = $("button", event.currentTarget);
        button.disabled = true;
        try {
            const result = await api(`/api/quest/articles/${encodeURIComponent(state.currentArticle.slug)}/comments`, {
                method: "POST",
                body: { content: input.value },
            });
            input.value = "";
            notify(result.status === "published" ? "评论已发布" : "评论进入复核", result.points_awarded ? `+${result.points_awarded} Vital Points` : result.message, result.status === "published" ? "success" : "warning");
            handleUnlocks(result.unlocked);
            const refreshed = await api(`/api/quest/articles/${encodeURIComponent(state.currentArticle.slug)}`);
            state.currentArticle.comments = refreshed.comments;
            renderComments(refreshed.comments);
            await refreshQuestData();
        } catch (error) {
            notify("评论发布失败", error.message, "danger");
        } finally {
            button.disabled = false;
        }
    }

    function renderComments(comments) {
        const list = $("#questCommentList");
        if (!comments.length) {
            list.replaceChildren(element("div", "quest-loading-line", "成为第一位参与讨论的探索者"));
            return;
        }
        list.replaceChildren(...comments.map((comment) => {
            const item = element("article", "quest-comment");
            item.appendChild(element("span", "quest-comment-avatar", initials(comment.display_name)));
            const copy = element("div");
            const head = element("div", "quest-comment-head");
            head.appendChild(element("strong", "", comment.display_name));
            if (comment.status === "review") head.appendChild(element("span", "", "审核中"));
            const time = document.createElement("time");
            time.textContent = formatDateTime(comment.created_at);
            head.appendChild(time);
            copy.appendChild(head);
            copy.appendChild(element("p", "", comment.content));
            item.appendChild(copy);
            return item;
        }));
    }

    function renderAtlas() {
        if (!state.overview) return;
        const achievements = state.overview.achievements || [];
        const unlocked = achievements.filter((item) => item.unlocked);
        $("#questAtlasCount").textContent = `${unlocked.length} / ${achievements.length}`;
        $("#questAtlasSummary").textContent = `${unlocked.length} 枚信号已稳定，${achievements.length - unlocked.length} 枚等待解锁`;
        $("#questAchievementGrid").replaceChildren(...achievements.map((achievement) => {
            const item = element("article", `quest-achievement ${achievement.unlocked ? "" : "locked"}`);
            item.dataset.tone = achievement.tone;
            item.appendChild(icon(achievement.unlocked ? achievement.icon : "lock-keyhole", "quest-medal"));
            const copy = element("div");
            copy.appendChild(element("strong", "", achievement.name));
            copy.appendChild(element("p", "", achievement.description));
            copy.appendChild(element("small", "", achievement.unlocked ? `UNLOCKED · ${formatDate(achievement.unlocked_at)}` : "SIGNAL LOCKED"));
            item.appendChild(copy);
            return item;
        }));
        renderActivity(state.overview.recent_activity || []);
        refreshIcons($("[data-quest-view='atlas']"));
    }

    function renderActivity(items) {
        const list = $("#questActivityList");
        if (!list) return;
        if (!items.length) {
            list.replaceChildren(element("div", "quest-loading-line", "完成行动后生成积分记录"));
            return;
        }
        list.replaceChildren(...items.map((activity) => {
            const item = element("div", "quest-activity-item");
            item.appendChild(element("strong", "", activity.label));
            item.appendChild(element("span", "", `+${activity.points} VP`));
            item.appendChild(element("small", "", formatDateTime(activity.created_at)));
            return item;
        }));
    }

    async function openProfiles() {
        try {
            const profiles = await api("/api/quest/profiles");
            const list = $("#questProfileList");
            list.replaceChildren(...profiles.map((profile) => {
                const button = element("button", `quest-profile-item ${profile.active ? "active" : ""}`);
                button.type = "button";
                button.appendChild(element("span", "quest-avatar", initials(profile.display_name)));
                const copy = element("span");
                copy.appendChild(element("strong", "", profile.display_name));
                copy.appendChild(element("small", "", `@${profile.username}`));
                button.appendChild(copy);
                button.appendChild(icon(profile.active ? "circle-check" : "arrow-right"));
                button.addEventListener("click", () => activateProfile(profile.id));
                return button;
            }));
            refreshIcons(list);
            $("#questProfileDialog").showModal();
        } catch (error) {
            notify("档案读取失败", error.message, "danger");
        }
    }

    async function activateProfile(id) {
        try {
            await api(`/api/quest/profiles/${id}/activate`, { method: "POST" });
            $("#questProfileDialog").close();
            await refreshQuestData(true);
            notify("探索者档案已切换", "进度与积分已经同步", "success");
        } catch (error) {
            notify("档案切换失败", error.message, "danger");
        }
    }

    async function createProfile(event) {
        event.preventDefault();
        const input = $("#questProfileName");
        const button = $("button", event.currentTarget);
        button.disabled = true;
        try {
            await api("/api/quest/profiles", { method: "POST", body: { display_name: input.value } });
            input.value = "";
            $("#questProfileDialog").close();
            await refreshQuestData(true);
            notify("探索者档案已建立", "新的健康远征从这里开始", "success");
        } catch (error) {
            notify("档案创建失败", error.message, "danger");
        } finally {
            button.disabled = false;
        }
    }

    function drawKnowledgeMap() {
        const canvas = $("#questKnowledgeMap");
        if (!canvas || canvas.offsetParent === null || !state.articles.length) return;
        const rect = canvas.getBoundingClientRect();
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        canvas.width = Math.max(1, Math.round(rect.width * ratio));
        canvas.height = Math.max(1, Math.round(rect.height * ratio));
        const context = canvas.getContext("2d");
        context.scale(ratio, ratio);
        const width = rect.width;
        const height = rect.height;
        context.clearRect(0, 0, width, height);

        const palette = ["#39bca4", "#e16d68", "#e1a33d", "#4f8ddd", "#9174c2", "#3f9eb7", "#6aa45e", "#c3688b"];
        const categories = state.categories.map((category, index) => {
            const articles = state.articles.filter((item) => item.category === category);
            const completion = articles.length ? articles.filter((item) => item.completed_at).length / articles.length : 0;
            const angle = -Math.PI / 2 + index / Math.max(state.categories.length, 1) * Math.PI * 2;
            const radiusX = width * 0.36;
            const radiusY = height * 0.34;
            return {
                name: category,
                completion,
                color: palette[index % palette.length],
                x: width / 2 + Math.cos(angle) * radiusX,
                y: height / 2 + Math.sin(angle) * radiusY,
            };
        });
        context.strokeStyle = "#d8e4e5";
        context.lineWidth = 1;
        categories.forEach((node) => {
            context.beginPath();
            context.moveTo(width / 2, height / 2);
            context.lineTo(node.x, node.y);
            context.stroke();
        });
        categories.forEach((node) => {
            context.fillStyle = node.color;
            context.globalAlpha = 0.18 + node.completion * 0.72;
            context.beginPath();
            context.arc(node.x, node.y, 12 + node.completion * 8, 0, Math.PI * 2);
            context.fill();
            context.globalAlpha = 1;
            context.strokeStyle = node.color;
            context.lineWidth = 2;
            context.stroke();
            context.fillStyle = "#41535b";
            context.font = "700 9px Microsoft YaHei";
            context.textAlign = "center";
            context.fillText(node.name, node.x, node.y + 33);
        });
        context.fillStyle = "#10272e";
        context.beginPath();
        context.arc(width / 2, height / 2, 31, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = "#57dcc5";
        context.font = "800 13px Microsoft YaHei";
        context.textAlign = "center";
        const completed = state.articles.filter((item) => item.completed_at).length;
        context.fillText(`${completed}/${state.articles.length}`, width / 2, height / 2 + 4);

        $("#questMapLegend").replaceChildren(...categories.map((node) => {
            const item = element("span", "", `${node.name} ${Math.round(node.completion * 100)}%`);
            item.style.color = node.color;
            item.prepend(element("i"));
            return item;
        }));
    }

    function startConstellation() {
        if (state.constellationFrame || document.hidden) return;
        const canvas = $("#questConstellation");
        if (!canvas || canvas.offsetParent === null) return;
        const points = Array.from({ length: 36 }, (_, index) => ({
            x: ((index * 47) % 101) / 100,
            y: ((index * 29) % 97) / 100,
            phase: index * 0.7,
        }));
        const draw = (time) => {
            if (!canvas.offsetParent) {
                state.constellationFrame = null;
                return;
            }
            const rect = canvas.getBoundingClientRect();
            const ratio = Math.min(window.devicePixelRatio || 1, 2);
            if (canvas.width !== Math.round(rect.width * ratio) || canvas.height !== Math.round(rect.height * ratio)) {
                canvas.width = Math.round(rect.width * ratio);
                canvas.height = Math.round(rect.height * ratio);
            }
            const context = canvas.getContext("2d");
            context.setTransform(ratio, 0, 0, ratio, 0, 0);
            context.clearRect(0, 0, rect.width, rect.height);
            const coords = points.map((point) => ({
                x: point.x * rect.width,
                y: point.y * rect.height + Math.sin(time * 0.0004 + point.phase) * 4,
            }));
            context.strokeStyle = "rgba(86, 214, 199, .12)";
            context.lineWidth = 1;
            coords.forEach((a, index) => {
                coords.slice(index + 1).forEach((b) => {
                    const distance = Math.hypot(a.x - b.x, a.y - b.y);
                    if (distance < 112) {
                        context.globalAlpha = 1 - distance / 112;
                        context.beginPath();
                        context.moveTo(a.x, a.y);
                        context.lineTo(b.x, b.y);
                        context.stroke();
                    }
                });
            });
            context.globalAlpha = 1;
            coords.forEach((point, index) => {
                context.fillStyle = index % 7 === 0 ? "#f0b84c" : "#4ddbc1";
                context.globalAlpha = 0.24 + (Math.sin(time * 0.001 + index) + 1) * 0.16;
                context.fillRect(point.x - 1.5, point.y - 1.5, 3, 3);
            });
            context.globalAlpha = 1;
            state.constellationFrame = window.requestAnimationFrame(draw);
        };
        state.constellationFrame = window.requestAnimationFrame(draw);
    }

    async function startGame() {
        if (state.gameBusy) return;
        state.gameBusy = true;
        try {
            const gameSession = await api("/api/quest/game/sessions", { method: "POST" });
            state.gameSession = gameSession;
            state.gameIndex = 0;
            state.gameRemaining = gameSession.scenario.duration_seconds;
            $("#gameLaunchPanel").classList.add("hidden");
            $("#gameResultPanel").classList.add("hidden");
            $("#gameFeedback").classList.add("hidden");
            $("#gameCaseFile").classList.remove("hidden");
            $("#gameExit").classList.remove("hidden");
            $("#gameShiftStatus").textContent = "执勤中";
            updateGameMetrics({ score: 0, safety_score: 100, completed_cases: 0, total_cases: gameSession.cases.length });
            if (!state.gameRuntime) {
                if (!window.VitalHospitalGame) throw new Error("游戏运行组件未加载");
                state.gameRuntime = new window.VitalHospitalGame({
                    parent: "hospitalGameCanvas",
                    actions: gameSession.actions,
                    onAction: handleGameAction,
                });
            }
            await state.gameRuntime.start(gameSession);
            showGameCase();
            startGameTimer();
        } catch (error) {
            notify("班次启动失败", error.message, "danger");
            $("#gameLaunchPanel").classList.remove("hidden");
        } finally {
            state.gameBusy = false;
        }
    }

    function showGameCase() {
        const session = state.gameSession;
        if (!session) return;
        const current = session.cases[state.gameIndex];
        if (!current) {
            finishGame();
            return;
        }
        $("#gameCaseTag").textContent = current.tag;
        $("#gameCaseIndex").textContent = `${String(state.gameIndex + 1).padStart(2, "0")} / ${String(session.cases.length).padStart(2, "0")}`;
        $("#gamePatientName").textContent = `${current.patient} · ${current.age}`;
        $("#gamePatientBrief").textContent = current.brief;
        $("#gameCaseFile").classList.remove("hidden");
        state.gameRuntime.setCase(current, state.gameIndex);
    }

    async function handleGameAction(action) {
        if (state.gameBusy || !state.gameSession) return;
        state.gameBusy = true;
        state.gameRuntime.lock();
        const current = state.gameSession.cases[state.gameIndex];
        try {
            const result = await api(`/api/quest/game/sessions/${state.gameSession.session_id}/action`, {
                method: "POST",
                body: { case_id: current.id, action },
            });
            updateGameMetrics(result);
            showGameFeedback(result);
        } catch (error) {
            notify("处置提交失败", error.message, "danger");
            state.gameRuntime.unlock();
        } finally {
            state.gameBusy = false;
        }
    }

    function showGameFeedback(result) {
        const feedback = $("#gameFeedback");
        feedback.classList.remove("hidden", "unsafe");
        feedback.classList.toggle("unsafe", !result.correct);
        $("#gameFeedbackIcon").replaceChildren(icon(result.correct ? "shield-check" : result.unsafe ? "shield-alert" : "rotate-ccw"));
        $("#gameFeedbackLabel").textContent = result.correct ? `SAFE ACTION · +${result.score_delta}` : result.unsafe ? `SAFETY BREACH · ${result.score_delta}` : `RECHECK · ${result.score_delta}`;
        $("#gameFeedbackTitle").textContent = result.correct ? "核对正确" : `正确节点：${result.correct_action.label}`;
        $("#gameFeedbackText").textContent = result.explanation;
        refreshIcons(feedback);
        state.gameRuntime.feedback(result.correct);
    }

    function nextGameCase() {
        $("#gameFeedback").classList.add("hidden");
        state.gameIndex += 1;
        if (!state.gameSession || state.gameIndex >= state.gameSession.cases.length) {
            finishGame();
            return;
        }
        state.gameRuntime.unlock();
        showGameCase();
    }

    function updateGameMetrics(result) {
        $("#gameScore").textContent = String(Math.max(0, result.score || 0)).padStart(4, "0");
        $("#gameSafety").textContent = `${result.safety_score ?? 100}%`;
        $("#gameQueue").textContent = `${result.completed_cases || 0} / ${result.total_cases || state.gameSession?.cases.length || 8}`;
    }

    function startGameTimer() {
        window.clearInterval(state.gameTimer);
        renderGameTime();
        state.gameTimer = window.setInterval(() => {
            if (document.hidden || !state.gameSession) return;
            state.gameRemaining -= 1;
            renderGameTime();
            if (state.gameRemaining <= 0) finishGame();
        }, 1000);
    }

    function renderGameTime() {
        const minutes = Math.max(0, Math.floor(state.gameRemaining / 60));
        const seconds = Math.max(0, state.gameRemaining % 60);
        $("#gameTimer").textContent = `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
    }

    async function finishGame() {
        if (state.gameBusy || !state.gameSession) return;
        state.gameBusy = true;
        window.clearInterval(state.gameTimer);
        state.gameTimer = null;
        try {
            const result = await api(`/api/quest/game/sessions/${state.gameSession.session_id}/finish`, { method: "POST" });
            state.gameRuntime.complete();
            $("#gameCaseFile").classList.add("hidden");
            $("#gameFeedback").classList.add("hidden");
            $("#gameExit").classList.add("hidden");
            $("#gameShiftStatus").textContent = "班次完成";
            $("#gameResultScore").textContent = result.score;
            $("#gameResultSafety").textContent = `${result.safety_score}%`;
            $("#gameResultCorrect").textContent = `${result.correct_actions} / ${result.patients_served}`;
            $("#gameResultPoints").textContent = `+${result.points_awarded} VP`;
            $("#gameResultTitle").textContent = result.unsafe_actions === 0 ? "零差错交接" : "班次复盘完成";
            $("#gameResultPanel").classList.remove("hidden");
            handleUnlocks(result.unlocked);
            state.gameSession = null;
            await refreshQuestData();
            refreshIcons($("#gameResultPanel"));
        } catch (error) {
            notify("班次结算失败", error.message, "danger");
        } finally {
            state.gameBusy = false;
        }
    }

    async function refreshQuestData(force = false) {
        const [overview, articleData] = await Promise.all([
            api("/api/quest/overview"),
            api("/api/quest/articles"),
        ]);
        state.overview = overview;
        state.articles = articleData.items || [];
        state.categories = articleData.categories || [];
        renderOverview();
        if (force) renderArticleFilters();
        renderArticles();
        renderAtlas();
        drawKnowledgeMap();
    }

    function handleUnlocks(unlocked = []) {
        unlocked.forEach((achievement, index) => {
            window.setTimeout(() => {
                notify(`成就解锁 · ${achievement.name}`, achievement.description, "success");
            }, index * 450);
        });
    }

    function initials(value) {
        const clean = String(value || "MP").trim();
        return clean.length <= 2 ? clean.toUpperCase() : clean.slice(0, 2).toUpperCase();
    }

    function formatDateTime(value) {
        if (!value) return "--";
        const date = new Date(String(value).includes("T") ? value : String(value).replace(" ", "T"));
        if (Number.isNaN(date.getTime())) return value;
        return new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" }).format(date);
    }

    function formatDate(value) {
        if (!value) return "--";
        return String(value).slice(0, 10);
    }

    window.MedProQuest = { load, pause, switchView };
})();
