"""本地大语言模型问答客户端 — Qwen+LoRA(优先) → Ollama → 错误提示

所有路径统一从 config.py 读取，支持环境变量覆盖。
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 从 config.py 统一读取配置 ──
# 确保 dashboard 目录在 path 中
_DASHBOARD_DIR = Path(__file__).resolve().parent.parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

try:
    from config import Config
except ImportError:
    # 降级：如果 config 不可用，用默认值
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    class Config:  # type: ignore[no-redef]
        LLM_BASE_MODEL = str(PROJECT_ROOT / "models" / "Qwen2.5-1.5B-Instruct")
        LLM_LORA_ADAPTER = str(PROJECT_ROOT / "models" / "lora")
        RAG_CHROMA_DIR = str(PROJECT_ROOT / "models" / "rag")
        RAG_BGE_MODEL = str(PROJECT_ROOT / "models" / "bge-small-zh-v1.5")
        OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# 模型路径（统一使用 Config 中的值）
LOCAL_BASE_MODEL = Path(Config.LLM_BASE_MODEL)
LORA_ADAPTER_DIR = Path(Config.LLM_LORA_ADAPTER)
BGE_MODEL_PATH = Path(Config.RAG_BGE_MODEL)
CHROMA_DIR = Path(Config.RAG_CHROMA_DIR)

# Ollama 配置
OLLAMA_URL = Config.OLLAMA_URL
OLLAMA_MODEL = Config.OLLAMA_MODEL

# ── 系统提示词 ──
MEDICAL_SYSTEM = (
    "你是「多模态智能医疗家庭助手」中的健康科普模块。"
    "只做用药与护理常识说明，不做确诊、不开处方。"
    "信息不足时建议就医。回答简洁、中文。"
)

RAG_SYSTEM = (
    "你是「多模态智能医疗家庭助手」中的健康科普模块。"
    "仅根据下列「参考资料」回答；资料不足时明确说不知道，并建议就医。"
    "不要编造药品剂量。不做确诊、不开处方。"
)

DISCLAIMER = "\n\n---\n⚠️ 免责声明：以上内容仅用于健康科普，不能替代专业医生诊断和治疗建议。如有不适请及时就医。"

# ── 全局缓存 ──
_LLM_CACHE = {}
_RAG_CACHE = {}


# ═══════════════════════════════════════════════════════════════
#  RAG 检索器
# ═══════════════════════════════════════════════════════════════

def _get_rag_retriever():
    """延迟加载 RAG 检索器（ChromaDB + BGE）"""
    if "retriever" in _RAG_CACHE:
        return _RAG_CACHE["retriever"]

    # 检查依赖
    try:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    except ImportError as e:
        logger.warning("RAG 缺少依赖: %s", e)
        _RAG_CACHE["retriever"] = None
        return None

    if not BGE_MODEL_PATH.is_dir():
        logger.warning("BGE 模型目录不存在: %s", BGE_MODEL_PATH)
        _RAG_CACHE["retriever"] = None
        return None

    if not CHROMA_DIR.is_dir():
        logger.warning("ChromaDB 目录不存在: %s", CHROMA_DIR)
        _RAG_CACHE["retriever"] = None
        return None

    try:
        import chromadb
        from chromadb.config import Settings

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )

        ef = SentenceTransformerEmbeddingFunction(
            model_name=str(BGE_MODEL_PATH),
            device="cpu",
        )

        collections = client.list_collections()
        if not collections:
            logger.warning("ChromaDB 中没有集合")
            _RAG_CACHE["retriever"] = None
            return None

        col = collections[0]
        _RAG_CACHE["retriever"] = (col, ef)
        logger.info("RAG 检索器就绪，%d 条医疗知识", col.count())
        return (col, ef)

    except Exception as e:
        logger.warning("RAG 初始化失败: %s", e)
        _RAG_CACHE["retriever"] = None
        return None


def _search_medical_knowledge(query, top_k=3):
    """从医疗知识库检索相关内容"""
    retriever = _get_rag_retriever()
    if retriever is None:
        return []

    col, ef = retriever
    try:
        results = col.query(query_texts=[query], n_results=min(top_k, col.count()))
        docs = results.get("documents", [[]])[0]
        return [d for d in docs if d]
    except Exception as e:
        logger.warning("RAG 检索失败: %s", e)
        return []


# ═══════════════════════════════════════════════════════════════
#  本地 Qwen + LoRA 模型
# ═══════════════════════════════════════════════════════════════

def _local_model_ready():
    """检查本地 Qwen + LoRA 是否可用"""
    if not LOCAL_BASE_MODEL.is_dir():
        return False
    if not (LOCAL_BASE_MODEL / "config.json").is_file():
        return False
    if not (LOCAL_BASE_MODEL / "model.safetensors").is_file():
        return False
    if not LORA_ADAPTER_DIR.is_dir():
        return False
    return True


def _load_local_model():
    """延迟加载 Qwen 基座 + LoRA 适配器"""
    if "bundle" in _LLM_CACHE:
        return _LLM_CACHE["bundle"]

    if not _local_model_ready():
        logger.info("本地模型不可用，将尝试 Ollama")
        _LLM_CACHE["bundle"] = None
        return None

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("加载本地模型 device=%s ...", device)

        tokenizer = AutoTokenizer.from_pretrained(
            str(LOCAL_BASE_MODEL), trust_remote_code=True, local_files_only=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_BASE_MODEL),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            trust_remote_code=True,
            local_files_only=True,
        )

        model = PeftModel.from_pretrained(base_model, str(LORA_ADAPTER_DIR))
        model.to(device)
        model.eval()

        bundle = (model, tokenizer, device)
        _LLM_CACHE["bundle"] = bundle
        logger.info("本地 Qwen+LoRA 模型加载成功")
        return bundle

    except ImportError as e:
        logger.warning("缺少依赖 (pip install transformers peft): %s", e)
        _LLM_CACHE["bundle"] = None
        return None
    except Exception as e:
        logger.warning("本地模型加载失败: %s", e)
        _LLM_CACHE["bundle"] = None
        return None


def _generate_local(question, rag_contexts=None):
    """使用本地 Qwen+LoRA 模型生成回答"""
    bundle = _load_local_model()
    if bundle is None:
        return None

    model, tokenizer, device = bundle

    # 构建消息
    if rag_contexts:
        ctx_text = "\n\n---\n\n".join(rag_contexts)
        system = RAG_SYSTEM + f"\n\n【参考资料】\n{ctx_text}"
    else:
        system = MEDICAL_SYSTEM

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        import torch
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen = out_ids[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(gen, skip_special_tokens=True).strip()

        return answer + DISCLAIMER

    except Exception as e:
        logger.warning("本地模型推理失败: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
#  Ollama 降级
# ═══════════════════════════════════════════════════════════════

def _ollama_list_models():
    """获取本地 Ollama 已拉取的模型列表"""
    try:
        import json
        import urllib.request
        base = OLLAMA_URL.replace("/api/generate", "")
        req = urllib.request.Request(f"{base}/api/tags")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def _ollama_pick_model():
    """自动选择最佳可用模型"""
    models = _ollama_list_models()
    if not models:
        return None

    # 优先级：指定的模型 > qwen家族 > 列表第一个
    preferred = OLLAMA_MODEL

    # 1. 精确匹配
    if preferred in models:
        return preferred

    # 2. qwen 前缀匹配（优先 7b > 0.5b）
    qwen_models = [m for m in models if "qwen" in m.lower()]
    for size in ("7b", "4b", "3b", "1.5b", "0.5b"):
        for m in qwen_models:
            if size in m.lower():
                return m

    # 3. 返回第一个可用模型
    return models[0] if models else None


def _ollama_available():
    """检查 Ollama 服务是否可用且有模型"""
    return _ollama_pick_model() is not None


def _generate_ollama(question):
    """使用 Ollama 生成回答（自动检测模型）"""
    import json
    import urllib.request

    model = _ollama_pick_model()
    if not model:
        return None

    logger.info("使用 Ollama 模型: %s", model)

    # ── 注入当前检测到的药品上下文 ──
    context_block = ""
    try:
        from models.fusion import get_medicine_context_for_chat
        ctx = get_medicine_context_for_chat()
        if ctx:
            context_block = f"\n{ctx}\n"
    except Exception:
        pass

    prompt = (
        "你是家庭医疗健康科普助手。请用通俗、谨慎、简洁的中文回答用户问题。\n"
        "要求：不要给出确定诊断，不要替代医生建议，涉及用药时提醒用户遵医嘱。\n"
        f"{context_block}"
        f"用户问题：{question}"
    )

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=120)
        data = json.loads(resp.read().decode("utf-8"))
        answer = data.get("response", "本地模型暂无返回结果。")
        return answer + DISCLAIMER
    except Exception as e:
        logger.warning("Ollama 调用失败: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
#  统一对外接口
# ═══════════════════════════════════════════════════════════════

def ask_local_llm(question: str) -> str:
    """调用本地大模型进行健康问答

    优先级：本地 Qwen+LoRA+RAG（医疗微调） > Ollama（通用） > 错误提示

    Args:
        question: 用户输入的问题

    Returns:
        str: 模型回答（包含免责声明）
    """
    # ── 尝试 RAG 检索 ──
    rag_contexts = _search_medical_knowledge(question)

    # ── 方式 1：本地 Qwen + LoRA（医疗微调，优先） ──
    if _local_model_ready():
        answer = _generate_local(question, rag_contexts)
        if answer:
            return answer
        logger.info("LoRA 模型未就绪，降级为 Ollama")

    # ── 方式 2：Ollama（降级方案） ──
    ollama_model = _ollama_pick_model()
    if ollama_model:
        logger.info("使用 Ollama: %s", ollama_model)
        answer = _generate_ollama(question)
        if answer:
            return answer

    # ── 方式 3：无可用模型 ──
    lines = [
        "⚠️ 当前无可用的本地大模型，请检查以下任一方式：\n",
    ]

    if not _local_model_ready():
        lines.append("❌ 本地 Qwen+LoRA 医疗微调模型未就绪")
        lines.append(f"   基座: {LOCAL_BASE_MODEL}")
        lines.append(f"   适配器: {LORA_ADAPTER_DIR}")
    else:
        lines.append("✅ 本地 Qwen+LoRA 模型存在，但加载失败（可能缺显存）")

    lines.append("")
    if not _ollama_pick_model():
        lines.append("❌ Ollama 降级方案也不可用")
        lines.append("   → 安装: ollama pull qwen2.5:0.5b")
    else:
        lines.append("✅ Ollama 可用，但调用失败")

    lines.append("\n💡 当前为离线模式，您可继续使用检测和统计功能。")
    return "\n".join(lines)


def is_llm_available():
    """检查是否有可用的 LLM（供前端状态显示）"""
    return _local_model_ready() or _ollama_available()


def get_llm_backend():
    """返回当前 LLM 后端名称（供前端显示）"""
    # 优先显示 LoRA（医疗微调模型）
    if _local_model_ready():
        return "本地 Qwen+LoRA（医疗微调）"
    # 降级显示 Ollama
    ollama_model = _ollama_pick_model()
    if ollama_model:
        return f"Ollama ({ollama_model})"
    return "未连接"


def get_rag_status():
    """返回 RAG 状态"""
    retriever = _get_rag_retriever()
    if retriever is not None:
        return f"医疗知识库 ({retriever[0].count()} 条)"
    return "未启用"
