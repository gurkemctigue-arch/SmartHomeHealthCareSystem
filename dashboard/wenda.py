# -*- coding: utf-8 -*-
"""问答引擎 — Qwen2.5-1.5B-Instruct + LoRA + RAG（上下文检索）

提供给 Flask app 的接口:
  get_engine() -> QAEngine
    .ask(question, session_id)  → {"answer","session_id","contexts","backend"}
    .info()                     → {"ready","backend","rag","model"}
    .history(session_id)        → [{"role","content"},...]
    .clear(session_id)          → bool

特性:
  - LoRA 微调后本地 Qwen2.5-1.5B 推理
  - 上下文感知检索：利用对话历史增强检索查询，支持追问
  - 知识库双轨：ChromaDB+BGE 优先；BGE 缺失时自动降级为 TF-IDF
  - 首次运行自动从 train_encyclopedia.json 构建索引
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 路径配置 ──
ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "train_encyclopedia.json"
CHROMA_DIR = Path(os.environ.get("RAG_CHROMA_DIR", str(ROOT / "models" / "rag")))
BGE_MODEL_PATH = Path(os.environ.get("RAG_BGE_MODEL", str(ROOT / "models" / "bge-small-zh-v1.5")))
RAG_MAX_CHUNKS = int(os.environ.get("RAG_MAX_CHUNKS", "20000"))
RAG_SEED = int(os.environ.get("RAG_SEED", "43"))

# ── 系统提示词 ──
SYSTEM_NO_RAG = (
    "你是「多模态智能医疗家庭助手」中的健康科普模块。"
    "只做用药与护理常识说明，不做确诊、不开处方。"
    "信息不足时建议就医。回答简洁、中文。"
)

SYSTEM_RAG = (
    "你是「多模态智能医疗家庭助手」中的健康科普模块。"
    "请根据下列「参考资料」回答用户问题。"
    "如果参考资料与用户问题完全不相关，忽略参考资料，用你的知识直接回答。"
    "资料不足时明确说不知道，并建议就医。"
    "不要编造药品剂量。不做确诊、不开处方。"
)

SYSTEM_CHITCHAT = (
    "你是「多模态智能医疗家庭助手」中的健康科普模块。"
    "用户正在与你打招呼或闲聊。请友好简洁地回应（1-2句话），"
    "并引导用户提出健康相关的问题。中文。"
)

# ── 闲聊/问候关键词（不触发 RAG 检索）──
CHITCHAT_PATTERNS: set[str] = {
    "你好", "您好", "嗨", "hi", "hello", "hey",
    "早上好", "下午好", "晚上好", "早",
    "谢谢", "感谢", "多谢", "thanks", "thank you",
    "再见", "拜拜", "bye", "goodbye", "回见",
    "你是谁", "你叫什么", "你的名字", "你是什么",
    "在吗", "在不在", "在不",
}

# ── TF-IDF 相关性阈值：top-1 分数低于此值视为不相关，不走 RAG ──
TFIDF_RELEVANCE_THRESHOLD = 0.06

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ 免责声明：以上内容仅用于健康科普，不能替代专业医生诊断和治疗建议。"
    "如有不适请及时就医。"
)


# ═══════════════════════════════════════════════════════════════════
#  TF-IDF 检索器（BGE 不可用时的降级方案）
# ═══════════════════════════════════════════════════════════════════

class TFIDFRetriever:
    """基于 scikit-learn TfidfVectorizer 的关键词检索器"""

    def __init__(self, chunks: list[str]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._chunks = chunks
        # 中文文本使用字符级 n-gram（2~4字），比默认分词更适合中文检索
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), max_features=8000,
        )
        self._matrix = self._vectorizer.fit_transform(chunks)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix)[0]
        top = scores.argsort()[-top_k:][::-1]
        return [(self._chunks[i], float(scores[i])) for i in top if scores[i] > 0]


# ═══════════════════════════════════════════════════════════════════
#  问答引擎
# ═══════════════════════════════════════════════════════════════════

class QAEngine:
    """LoRA+RAG 问答引擎（单例）

    初始化在后台线程异步执行，避免阻塞 Flask 请求。
    """

    def __init__(self) -> None:
        self._model_bundle = None       # (model, tokenizer, device)
        self._retriever: TFIDFRetriever | None = None
        self._chroma_col = None
        self._chroma_ef = None
        self._retriever_backend = "loading..."
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._ready = False
        self._init_error: str | None = None
        self._init_done = threading.Event()

        # 后台线程异步加载模型和知识库
        t = threading.Thread(target=self._init_worker, daemon=True, name="qa-engine-init")
        t.start()

    def _init_worker(self) -> None:
        """后台初始化：加载模型 + 构建知识库索引"""
        try:
            logger.info("后台线程开始加载引擎...")
            self._init_model()
            self._init_retriever()
            self._ready = self._model_bundle is not None
            if self._ready:
                logger.info("引擎初始化完成")
            else:
                self._init_error = "模型加载失败，请检查模型路径配置"
        except Exception as e:
            self._init_error = str(e)
            logger.error("引擎初始化异常: %s", e)
        finally:
            self._init_done.set()

    # ── 模型加载 ────────────────────────────────────────────────

    def _init_model(self) -> None:
        """加载 Qwen2.5-1.5B 基座 + LoRA 适配器"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        # 从 config 统一读取路径
        try:
            from config import Config
            base_path = Path(Config.LLM_BASE_MODEL)
            lora_path = Path(Config.LLM_LORA_ADAPTER)
        except ImportError:
            base_path = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
            lora_path = ROOT / "models" / "lora_adapter"

        if not base_path.is_dir():
            logger.warning("Qwen 基座不存在: %s", base_path)
            return
        if not lora_path.is_dir():
            logger.warning("LoRA 适配器不存在: %s", lora_path)
            return

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("加载 Qwen+LoRA device=%s ...", device)

            tokenizer = AutoTokenizer.from_pretrained(
                str(base_path), trust_remote_code=True, local_files_only=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            base_model = AutoModelForCausalLM.from_pretrained(
                str(base_path),
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                trust_remote_code=True,
                local_files_only=True,
            )
            model = PeftModel.from_pretrained(base_model, str(lora_path))
            model.to(device)
            model.eval()

            self._model_bundle = (model, tokenizer, device)
            logger.info("Qwen+LoRA 加载成功")
        except Exception as e:
            logger.warning("模型加载失败: %s", e)

    # ── 知识库索引 ──────────────────────────────────────────────

    def _init_retriever(self) -> None:
        """初始化检索器：ChromaDB+BGE → TF-IDF 降级"""
        # ── 轨道 A: ChromaDB + BGE ──
        if CHROMA_DIR.is_dir() and BGE_MODEL_PATH.is_dir():
            try:
                import chromadb
                from chromadb.config import Settings
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )

                client = chromadb.PersistentClient(
                    path=str(CHROMA_DIR),
                    settings=Settings(anonymized_telemetry=False),
                )
                ef = SentenceTransformerEmbeddingFunction(
                    model_name=str(BGE_MODEL_PATH), device="cpu",
                )
                collections = client.list_collections()
                if collections:
                    self._chroma_col = collections[0]
                    self._chroma_ef = ef
                    self._retriever_backend = f"ChromaDB+BGE ({self._chroma_col.count()}条)"
                    logger.info("RAG: %s", self._retriever_backend)
                    return
            except Exception as e:
                logger.warning("ChromaDB 初始化失败: %s", e)

        # ── 轨道 B: TF-IDF 降级 ──
        self._build_tfidf_index()

    def _load_chunks(self) -> list[str]:
        """从 train_encyclopedia.json 加载语料块（含抽样）"""
        if not KB_PATH.is_file():
            logger.warning("知识库文件不存在: %s", KB_PATH)
            return []

        # 先用 reservoir sampling 读取，避免 36 万行全部进内存
        rng = random.Random(RAG_SEED)
        reservoir: list[str] = []
        idx = 0

        with open(KB_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", "")
                except json.JSONDecodeError:
                    continue
                if not text or len(text) < 20:
                    continue

                if RAG_MAX_CHUNKS > 0:
                    if len(reservoir) < RAG_MAX_CHUNKS:
                        reservoir.append(text)
                    else:
                        j = rng.randint(0, idx)
                        if j < RAG_MAX_CHUNKS:
                            reservoir[j] = text
                    idx += 1
                else:
                    reservoir.append(text)

        logger.info("知识库加载: %d 条（总数 %d）", len(reservoir), idx)
        return reservoir

    def _build_tfidf_index(self) -> None:
        """构建 TF-IDF 索引"""
        chunks = self._load_chunks()
        if not chunks:
            return

        try:
            self._retriever = TFIDFRetriever(chunks)
            self._retriever_backend = f"TF-IDF ({len(chunks)}条)"
            logger.info("TF-IDF 索引就绪: %d 条", len(chunks))
        except ImportError:
            logger.warning("scikit-learn 未安装，RAG 不可用")
        except Exception as e:
            logger.warning("TF-IDF 索引构建失败: %s", e)

    # ── 闲聊检测 ──────────────────────────────────────────────

    def _is_chitchat(self, question: str) -> bool:
        """判断用户输入是否为闲聊/问候（不触发 RAG 检索）"""
        q = question.strip().lower().rstrip("?？!！~～。.")
        # 极短输入视为闲聊
        if len(q) <= 2:
            return True
        # 关键词匹配
        if q in CHITCHAT_PATTERNS:
            return True
        return False

    # ── 上下文感知检索 ──────────────────────────────────────────

    def _build_context_query(self, question: str, session_id: str | None) -> str:
        """将对话历史中的用户问题与当前问题拼接，增强检索命中率

        例: 历史=["孩子发烧怎么办"], 当前="用什么药"
            → 检索查询 = "孩子发烧怎么办 用什么药"
        """
        if session_id and session_id in self._sessions:
            history = self._sessions[session_id]
            # 取最近 3 轮用户提问，与当前问题拼接
            user_qs = [h["content"] for h in history if h["role"] == "user"][-3:]
            if user_qs:
                return " ".join(user_qs + [question])
        return question

    def _search_kb(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        """从知识库检索相关文档，返回 [(text, score), ...]

        ChromaDB 返回固定分数 1.0（语义检索质量高），TF-IDF 返回余弦相似度。
        """
        # 轨道 A: ChromaDB
        if self._chroma_col is not None:
            try:
                results = self._chroma_col.query(
                    query_texts=[query],
                    n_results=min(top_k, self._chroma_col.count()),
                )
                docs = results.get("documents", [[]])[0]
                # ChromaDB 有内置距离，但此处统一返回 1.0 表示高质量
                return [(d, 1.0) for d in docs if d]
            except Exception as e:
                logger.warning("ChromaDB 检索失败: %s", e)

        # 轨道 B: TF-IDF
        if self._retriever is not None:
            return self._retriever.search(query, top_k=top_k)

        return []

    # ── 模型推理 ────────────────────────────────────────────────

    def _build_prompt(
        self,
        question: str,
        contexts: list[str],
        session_id: str | None,
        system: str | None = None,
    ) -> list[dict[str, str]]:
        """组装完整消息列表（system + 参考资料 + 历史 + 当前问题）

        Args:
            system: 覆盖默认 system prompt（用于闲聊等特殊场景）
        """
        if system is not None:
            system_content = system
        elif contexts:
            ctx_text = "\n\n---\n\n".join(contexts)
            system_content = SYSTEM_RAG + f"\n\n【参考资料】\n{ctx_text}"
        else:
            system_content = SYSTEM_NO_RAG

        messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]

        # 历史对话（最近 6 轮 = 12 条消息）
        if session_id and session_id in self._sessions:
            messages.extend(self._sessions[session_id][-12:])

        messages.append({"role": "user", "content": question})
        return messages

    def _generate(
        self,
        question: str,
        contexts: list[str],
        session_id: str | None,
        system: str | None = None,
        max_tokens: int = 150,
    ) -> str:
        """调用 Qwen+LoRA 生成回答"""
        if self._model_bundle is None:
            return "⏳ 模型仍在加载中，请稍候..."
        model, tokenizer, device = self._model_bundle
        messages = self._build_prompt(question, contexts, session_id, system=system)

        try:
            import torch

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            gen = out_ids[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(gen, skip_special_tokens=True).strip()
            answer = self._clean_tail(answer)
            return answer + DISCLAIMER
        except Exception as e:
            logger.warning("模型推理失败: %s", e)
            return f"抱歉，推理出错: {e}"

    # ═══════════════════════════════════════════════════════════════
    #  公开接口（Flask app 调用）
    # ═══════════════════════════════════════════════════════════════

    def ask(
        self, question: str, session_id: str | None = None,
    ) -> dict:
        """核心问答：闲聊检测 → 上下文检索 → 相关性过滤 → LoRA 推理

        Args:
            question: 用户问题
            session_id: 会话 ID（None 则使用 "default"）

        Returns:
            {"answer", "session_id", "contexts", "backend"}
        """
        sid = session_id or "default"

        if not self._ready:
            # 还在加载中 vs 加载失败
            if not self._init_done.is_set():
                return {
                    "answer": "⏳ 问答引擎正在加载中，大模型和知识库初始化需要 1~2 分钟，请稍后再试。",
                    "session_id": sid,
                    "contexts": [],
                    "backend": "initializing",
                }
            return {
                "answer": f"❌ 问答引擎未就绪: {self._init_error or '请检查模型路径配置'}",
                "session_id": sid,
                "contexts": [],
                "backend": "error",
            }

        # ── ① 闲聊/问候检测：跳过 RAG，直接友好回复 ──
        if self._is_chitchat(question):
            answer = self._generate(question, contexts=[], session_id=sid,
                                    system=SYSTEM_CHITCHAT, max_tokens=80)
            self._save_history(sid, question, answer)
            return {
                "answer": answer,
                "session_id": sid,
                "contexts": [],
                "backend": "chitchat (no RAG)",
            }

        # ── ② 上下文感知查询构造 ──
        ctx_query = self._build_context_query(question, sid)

        # ── ③ 知识库检索（带分数）──
        hits = self._search_kb(ctx_query, top_k=3)

        # ── ④ 相关性过滤：top-1 分数低于阈值则丢弃所有上下文 ──
        if hits and hits[0][1] >= TFIDF_RELEVANCE_THRESHOLD:
            contexts = [doc for doc, _ in hits]
        else:
            contexts = []
            if hits:
                logger.info("检索结果相关性不足, top1_score=%.4f < %.2f, 降级为无RAG模式",
                            hits[0][1], TFIDF_RELEVANCE_THRESHOLD)

        # ── ⑤ 模型推理 ──
        answer = self._generate(question, contexts=contexts, session_id=sid)

        # ── ⑥ 坏回答重试：去掉 RAG 再试一次 ──
        if self._is_bad_answer(answer) and contexts:
            logger.info("检测到退化回答，重试（无RAG模式）")
            answer = self._generate(question, contexts=[], session_id=sid)

        # ── ⑦ 保存会话历史（坏回答不入库）──
        self._save_history(sid, question, answer)

        rag_backend = self._retriever_backend if contexts else f"no-RAG ({self._retriever_backend})"
        return {
            "answer": answer,
            "session_id": sid,
            "contexts": [c[:200] + "..." if len(c) > 200 else c for c in contexts],
            "backend": rag_backend,
        }

    def _save_history(self, sid: str, question: str, answer: str) -> None:
        """保存本轮问答到会话历史（坏回答不入库）"""
        if self._is_bad_answer(answer):
            logger.warning("检测到退化回答，不存入历史: %s...", answer[:80])
            return
        if sid not in self._sessions:
            self._sessions[sid] = []
        self._sessions[sid].append({"role": "user", "content": question})
        self._sessions[sid].append({"role": "assistant", "content": answer})
        if len(self._sessions[sid]) > 20:
            self._sessions[sid] = self._sessions[sid][-20:]

    @staticmethod
    def _is_bad_answer(answer: str) -> bool:
        """检测回答是否退化（同一短语/词重复出现 ≥5 次）

        例如 "癌症学、癌症学、癌症学..." 中 "癌症学" 重复 N 次。
        """
        if len(answer) < 20:
            return False
        # 滑动窗口检测：取 2~5 字短语，检查是否连续重复多次
        import re
        # 方法：找所有连续汉字段，split 后统计频次最高的 token
        tokens = re.findall(r"[一-鿿]{2,5}", answer)
        if not tokens:
            return False
        from collections import Counter
        counts = Counter(tokens)
        most_common, freq = counts.most_common(1)[0]
        # 同一短语出现 ≥5 次且总长度占比 >30% 视为退化
        if freq >= 5:
            occupied = len(most_common) * freq
            if occupied / len(answer) > 0.3:
                return True
        return False

    @staticmethod
    def _clean_tail(answer: str) -> str:
        """清理模型输出尾部乱码（采样噪声导致的无意义字符）

        按句末标点分割成句子，从后往前找第一个"正常句子"：
        - 首字符是汉字
        - 汉字占比 ≥ 50%
        丢弃之后的所有乱码片段。
        """
        import re
        segments = re.split(r"(?<=[。！？])", answer)
        if len(segments) <= 1:
            return answer

        cut_at = len(answer)
        for seg in reversed(segments):
            if not seg.strip():
                cut_at -= len(seg)
                continue
            # 检查首字符是否为汉字
            first_char = seg.strip()[0]
            starts_cjk = "一" <= first_char <= "鿿"
            # 统计汉字占比
            chars = [c for c in seg if c.strip()]
            cjk = sum(1 for c in chars if "一" <= c <= "鿿")
            ratio = cjk / max(len(chars), 1)
            if starts_cjk and ratio >= 0.5:
                break  # 正常句子，停止
            cut_at -= len(seg)

        return answer[:cut_at]

    def info(self) -> dict:
        """引擎状态信息（字段名与前端的 chat.html 对齐）"""
        _kb_count = 0
        if self._retriever is not None:
            _kb_count = len(self._retriever._chunks)
        elif self._chroma_col is not None:
            _kb_count = self._chroma_col.count()

        return {
            "ready": self._ready,
            "lora_ok": self._ready and self._model_bundle is not None,
            "kb_sentences": _kb_count,
            "backend": "Qwen2.5-1.5B-Instruct + LoRA",
            "rag": self._retriever_backend,
            "model": "Qwen2.5-1.5B-Instruct (医疗LoRA微调)",
        }

    def history(self, session_id: str) -> list[dict[str, str]]:
        """获取会话历史"""
        return self._sessions.get(session_id, [])

    def clear(self, session_id: str) -> bool:
        """清除会话历史"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# ═══════════════════════════════════════════════════════════════════
#  全局单例
# ═══════════════════════════════════════════════════════════════════

_engine: QAEngine | None = None
_lock = threading.Lock()


def get_engine() -> QAEngine:
    """获取全局问答引擎单例（双重检查锁，线程安全）"""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = QAEngine()
    return _engine


# ═══════════════════════════════════════════════════════════════════
#  独立运行：终端交互问答
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print(" Qwen2.5-1.5B-Instruct + LoRA + RAG 问答对话")
    print(" 上下文检索 · 多轮对话 · 知识库增强")
    print(" 输入 quit / exit / q 退出")
    print("=" * 60)

    print("正在加载引擎（后台线程）...")
    engine = get_engine()
    # 等待后台初始化完成
    if not engine._init_done.is_set():
        print("等待模型和知识库加载...")
        engine._init_done.wait()

    info = engine.info()
    print(f"\n引擎状态: {'✅ 就绪' if info['ready'] else '❌ 未就绪'}")
    print(f"模型后端: {info['backend']}")
    print(f"LoRA: {'✅ 已加载' if info['lora_ok'] else '❌ 未加载'}")
    print(f"检索后端: {info['rag']} ({info['kb_sentences']}条)")

    if not info["ready"]:
        print(f"\n⚠️  引擎未就绪: {engine._init_error or '未知错误'}")
        raise SystemExit(1)

    sid = "interactive"
    while True:
        try:
            q = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            print("再见！")
            break

        result = engine.ask(q, session_id=sid)
        print(f"\n助手: {result['answer']}")
