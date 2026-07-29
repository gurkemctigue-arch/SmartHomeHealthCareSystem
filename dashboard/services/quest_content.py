"""Curated content and scenario seeds for the VITAL QUEST experience."""

from __future__ import annotations

import json


ARTICLES = [
    {
        "slug": "antibiotics-are-not-cold-medicine",
        "title": "抗生素不是感冒药",
        "dek": "从一次普通感冒开始，弄清抗生素什么时候无效、什么时候必须由医生判断。",
        "category": "安全用药",
        "source_name": "WHO · Antimicrobial resistance",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance",
        "reviewer": "MedPro 医学内容组",
        "reviewed_at": "2026-07-20",
        "reading_minutes": 5,
        "cover_asset": "/static/images/quest/article-medicine.jpg",
        "accent": "coral",
        "body": [
            {
                "heading": "先分清感染类型",
                "paragraphs": [
                    "普通感冒大多由病毒引起。抗生素针对细菌，对病毒性感冒没有治疗作用，也不能让感冒更快好转。",
                    "是否存在细菌感染，需要结合症状、病程、检查和个体情况判断。仅凭鼻涕颜色、咳嗽或发热，不能自行决定使用抗生素。",
                ],
            },
            {
                "heading": "错误使用会留下长期代价",
                "paragraphs": [
                    "不必要地使用、擅自减量或提前停药，都会增加耐药风险。耐药不是人体对药物产生抵抗，而是细菌发生变化，使原本有效的药物失去作用。",
                    "家中剩余的抗生素不应留给下次自行服用，也不要与他人分享。需要使用时，应由医生评估并按照处方完成疗程。",
                ],
            },
        ],
        "takeaways": ["普通感冒通常不需要抗生素", "不共享、不囤用剩余抗生素", "是否用药由医生结合病情判断"],
        "quiz": [
            {"question": "普通感冒多数由什么引起？", "options": ["病毒", "细菌", "真菌"], "answer": 0},
            {"question": "鼻涕变黄是否足以自行使用抗生素？", "options": ["是", "否", "只需减半剂量"], "answer": 1},
            {"question": "家中剩余抗生素应如何处理？", "options": ["留给家人", "下次感冒服用", "咨询药师并规范处置"], "answer": 2},
        ],
    },
    {
        "slug": "five-checks-before-taking-medicine",
        "title": "服药前的五次核对",
        "dek": "药名、对象、时间、用法和有效期，五个动作挡住家庭药箱里最常见的差错。",
        "category": "安全用药",
        "source_name": "国家药品监督管理局",
        "source_url": "https://www.nmpa.gov.cn/",
        "reviewer": "MedPro 药学内容组",
        "reviewed_at": "2026-07-22",
        "reading_minutes": 6,
        "cover_asset": "/static/images/quest/article-pharmacy.jpg",
        "accent": "gold",
        "body": [
            {
                "heading": "不要只认包装颜色",
                "paragraphs": [
                    "相似包装、相似名称和不同规格，是家庭用药差错的重要来源。取药后先读通用名称，再核对使用者和医嘱，不要仅凭药盒颜色或记忆判断。",
                    "儿童、孕妇、老年人以及肝肾功能异常者，适用范围可能不同。即使药名相同，也不能直接沿用他人的用法。",
                ],
            },
            {
                "heading": "有效期不是唯一的安全线",
                "paragraphs": [
                    "药品应按照说明书规定的温度、避光和防潮条件保存。已经开封、受潮、变色或包装破损的药品，即使仍在标示有效期内，也应先咨询药师。",
                    "服药前完成五项核对：药名、使用者、时间、用法以及有效期与包装状态。发现任何一项不一致，立即暂停。",
                ],
            },
        ],
        "takeaways": ["先读通用名称，不靠包装认药", "特殊人群需要额外核对", "包装异常时先暂停并咨询"],
        "quiz": [
            {"question": "两个药盒颜色相同，可以认为是同一种药吗？", "options": ["可以", "不可以", "价格相同就可以"], "answer": 1},
            {"question": "药品在有效期内但已经受潮，应该怎么做？", "options": ["继续服用", "加倍服用", "暂停并咨询药师"], "answer": 2},
            {"question": "服药核对中不包括哪一项？", "options": ["使用者", "广告知名度", "药名"], "answer": 1},
        ],
    },
    {
        "slug": "recognize-stroke-fast",
        "title": "用 FAST 识别卒中信号",
        "dek": "脸歪、手臂无力、言语不清，任何一个突然出现都值得立即行动。",
        "category": "急救识别",
        "source_name": "CDC · Signs and Symptoms of Stroke",
        "source_url": "https://www.cdc.gov/stroke/signs-symptoms/",
        "reviewer": "MedPro 急救内容组",
        "reviewed_at": "2026-07-19",
        "reading_minutes": 4,
        "cover_asset": "/static/images/quest/article-care.jpg",
        "accent": "red",
        "body": [
            {
                "heading": "FAST 是一套行动口令",
                "paragraphs": [
                    "F 代表观察面部是否突然不对称；A 代表让对方抬起双臂，观察一侧是否无力下垂；S 代表听言语是否含糊或表达困难；T 代表立即记录时间并呼叫急救。",
                    "症状即使短暂缓解，也不能因此等待观察。短暂性脑缺血发作同样需要紧急评估。",
                ],
            },
            {
                "heading": "等待急救时不要做什么",
                "paragraphs": [
                    "不要自行驾车长距离送医，不要给意识不清的人喂水、食物或药物，也不要因为想先量完所有指标而延误呼救。",
                    "保持环境安全，记录最后一次正常的时间、既往病史和正在使用的药物，等待专业急救人员接手。",
                ],
            },
        ],
        "takeaways": ["症状突然出现是关键信号", "立即呼叫急救并记录时间", "不向意识不清者喂水或药物"],
        "quiz": [
            {"question": "FAST 中的 T 代表什么？", "options": ["测体温", "立即行动并记录时间", "先休息"], "answer": 1},
            {"question": "症状几分钟后消失，可以不就医吗？", "options": ["可以", "不可以", "第二天再说"], "answer": 1},
            {"question": "等待急救时应做什么？", "options": ["记录最后正常时间", "强行喂水", "自行加药"], "answer": 0},
        ],
    },
    {
        "slug": "measure-blood-pressure-correctly",
        "title": "一次可信的家庭血压测量",
        "dek": "姿势、袖带和安静时间，会让同一台血压计给出完全不同的答案。",
        "category": "慢病管理",
        "source_name": "WHO · Hypertension",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/hypertension",
        "reviewer": "MedPro 慢病内容组",
        "reviewed_at": "2026-07-18",
        "reading_minutes": 5,
        "cover_asset": "/static/images/quest/article-emergency.jpg",
        "accent": "blue",
        "body": [
            {
                "heading": "测量前先让身体安静下来",
                "paragraphs": [
                    "测量前安静坐位休息至少五分钟。短时间内的运动、吸烟、咖啡因和情绪波动都可能影响结果。",
                    "坐姿时背部有支撑、双脚平放、不交叉双腿。手臂放在桌面，使袖带大致与心脏同高。",
                ],
            },
            {
                "heading": "看趋势，不追逐单个数字",
                "paragraphs": [
                    "选择尺寸合适的上臂式袖带，并按照设备说明操作。每次可间隔一分钟测量两次，记录日期、时间和读数。",
                    "家庭监测用于帮助医生理解长期趋势，不能根据一次偏高或偏低的读数自行停药、加药或减药。出现胸痛、呼吸困难、神经系统异常等症状时应及时就医。",
                ],
            },
        ],
        "takeaways": ["测量前安静休息至少五分钟", "袖带与心脏大致同高", "不要根据单次读数自行调整药物"],
        "quiz": [
            {"question": "测量前建议安静休息多久？", "options": ["至少 5 分钟", "无需休息", "30 秒"], "answer": 0},
            {"question": "测量时双脚应如何放置？", "options": ["交叉悬空", "平放地面", "踩在椅子上"], "answer": 1},
            {"question": "一次读数偏高后应该怎么做？", "options": ["自行加药", "规范复测并记录，必要时咨询医生", "立即停药"], "answer": 1},
        ],
    },
    {
        "slug": "heat-illness-warning-signs",
        "title": "高温下，什么时候必须停下来",
        "dek": "从大量出汗到意识异常，读懂热相关疾病升级前发出的身体信号。",
        "category": "环境健康",
        "source_name": "WHO · Climate change, heat and health",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health",
        "reviewer": "MedPro 环境健康组",
        "reviewed_at": "2026-07-23",
        "reading_minutes": 5,
        "cover_asset": "/static/images/quest/article-heat.jpg",
        "accent": "orange",
        "body": [
            {
                "heading": "热不适会逐步升级",
                "paragraphs": [
                    "头晕、乏力、恶心、头痛、大量出汗和肌肉痉挛，提示身体已经受到高温影响。此时应立即停止活动，转移到阴凉或空调环境并逐步降温。",
                    "老人、婴幼儿、户外劳动者以及患有心血管、呼吸或肾脏疾病的人，更容易受到高温影响。",
                ],
            },
            {
                "heading": "意识异常是紧急信号",
                "paragraphs": [
                    "意识模糊、抽搐、昏厥、无法正常饮水或皮肤明显灼热，都可能提示严重热相关疾病，应立即呼叫急救。",
                    "等待救援时持续帮助降温，但不要给意识不清的人强行喂水，也不要因为体温暂时下降而取消专业评估。",
                ],
            },
        ],
        "takeaways": ["出现不适立即停止高温活动", "意识异常时立即呼叫急救", "意识不清者不能强行喂水"],
        "quiz": [
            {"question": "高温中出现头晕乏力，第一步是什么？", "options": ["继续坚持", "停止活动并转移到凉爽处", "喝咖啡"], "answer": 1},
            {"question": "哪一项属于紧急信号？", "options": ["轻微口渴", "意识模糊", "想休息"], "answer": 1},
            {"question": "意识不清时能否强行喂水？", "options": ["能", "不能", "越多越好"], "answer": 1},
        ],
    },
    {
        "slug": "read-sodium-on-food-labels",
        "title": "食品标签里的隐形盐",
        "dek": "不只看咸不咸，学会用营养成分表比较每 100 克食品中的钠。",
        "category": "营养识读",
        "source_name": "WHO · Healthy diet",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/healthy-diet",
        "reviewer": "MedPro 营养内容组",
        "reviewed_at": "2026-07-21",
        "reading_minutes": 4,
        "cover_asset": "/static/images/quest/article-nutrition.jpg",
        "accent": "green",
        "body": [
            {
                "heading": "钠不只藏在盐罐里",
                "paragraphs": [
                    "加工肉制品、方便食品、调味酱、腌制食品和部分烘焙食品，即使尝起来不特别咸，也可能含有较多钠。",
                    "比较同类食品时，先确认营养成分表使用的是每 100 克、每份还是整包装，再比较钠含量，避免被不同份量误导。",
                ],
            },
            {
                "heading": "从一顿饭的整体做减法",
                "paragraphs": [
                    "减少高钠食品的频率和份量，同时增加新鲜食材比例，比只盯着某一种食品更容易坚持。烹调时可用香草、醋、柠檬和天然香辛料丰富味道。",
                    "高血压、肾脏疾病或正在使用特定药物的人，适合的钠摄入目标可能不同，应结合医生或营养师建议。",
                ],
            },
        ],
        "takeaways": ["比较前先统一营养标示单位", "酱料和加工食品可能含有较多钠", "基础疾病人群需要个体化建议"],
        "quiz": [
            {"question": "比较两款食品钠含量前，应先看什么？", "options": ["包装颜色", "标示单位", "广告语"], "answer": 1},
            {"question": "吃起来不咸是否代表钠一定低？", "options": ["是", "否", "甜食一定无钠"], "answer": 1},
            {"question": "肾脏疾病人群应如何确定摄入目标？", "options": ["照搬他人方案", "完全不吃盐", "结合专业建议"], "answer": 2},
        ],
    },
    {
        "slug": "asthma-and-air-quality",
        "title": "空气质量变化时保护呼吸",
        "dek": "空气污染升高并不意味着完全不活动，而是需要更聪明地调整时间和强度。",
        "category": "呼吸健康",
        "source_name": "WHO · Asthma",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/asthma",
        "reviewer": "MedPro 呼吸健康组",
        "reviewed_at": "2026-07-17",
        "reading_minutes": 5,
        "cover_asset": "/static/images/quest/article-air.jpg",
        "accent": "cyan",
        "body": [
            {
                "heading": "先看趋势，再安排活动",
                "paragraphs": [
                    "空气质量变差时，可缩短户外活动时间、降低强度，或选择空气质量更好的时段和室内场所。儿童、老人和呼吸系统疾病患者需要更谨慎。",
                    "道路附近、烟雾和强烈气味可能加重刺激。通风应结合室外空气质量，污染高峰期不宜长时间开窗。",
                ],
            },
            {
                "heading": "出现症状时使用既定行动计划",
                "paragraphs": [
                    "哮喘患者应随身携带医生开具的缓解药物，并按照个人哮喘行动计划处理，不要借用他人的吸入器。",
                    "如果呼吸困难快速加重、说话困难、嘴唇发紫或常用缓解药物效果不佳，应立即寻求紧急医疗帮助。",
                ],
            },
        ],
        "takeaways": ["污染升高时调整活动时间和强度", "遵循个人哮喘行动计划", "严重呼吸困难时立即求助"],
        "quiz": [
            {"question": "空气污染升高时适合怎么做？", "options": ["增加户外强度", "调整时间并降低强度", "焚烧香薰"], "answer": 1},
            {"question": "能否借用他人的吸入器？", "options": ["能", "不能", "只借儿童的"], "answer": 1},
            {"question": "哪种情况需要紧急求助？", "options": ["轻微打哈欠", "说话困难且呼吸加重", "短暂鼻痒"], "answer": 1},
        ],
    },
    {
        "slug": "medicine-list-for-every-visit",
        "title": "随身药物清单，关键时刻少一次猜测",
        "dek": "把处方药、非处方药、保健品和过敏史放进同一张可更新的清单。",
        "category": "照护沟通",
        "source_name": "WHO · Medication Without Harm",
        "source_url": "https://www.who.int/initiatives/medication-without-harm",
        "reviewer": "MedPro 患者安全组",
        "reviewed_at": "2026-07-24",
        "reading_minutes": 4,
        "cover_asset": "/static/images/quest/article-pharmacy.jpg",
        "accent": "violet",
        "body": [
            {
                "heading": "清单要包含所有正在使用的东西",
                "paragraphs": [
                    "清单不仅包括医院处方药，也要写入非处方药、中成药、维生素、保健品和外用药。记录通用名称、用途、使用方式以及开具机构。",
                    "同时记录明确的药物和食物过敏史，以及曾经出现过的严重不良反应。不要只写“消炎药过敏”这类含糊描述。",
                ],
            },
            {
                "heading": "每次变化后立刻更新",
                "paragraphs": [
                    "新增、停用或调整药物后及时更新，并在就诊、购药和紧急情况时主动出示。清单帮助医护人员发现重复用药和潜在相互作用。",
                    "清单只是沟通工具，不能替代处方。发现药物看起来重复或冲突时，应暂停猜测并向医生或药师核实。",
                ],
            },
        ],
        "takeaways": ["处方药与保健品都要记录", "过敏史尽量写明具体药名和反应", "药物变化后立即更新清单"],
        "quiz": [
            {"question": "药物清单是否需要记录保健品？", "options": ["需要", "不需要", "只记录进口产品"], "answer": 0},
            {"question": "过敏史怎样记录更有效？", "options": ["写“药物过敏”", "写明具体药名和反应", "不用记录"], "answer": 1},
            {"question": "发现两种药看起来重复时应怎么做？", "options": ["都服用", "随机选一种", "向医生或药师核实"], "answer": 2},
        ],
    },
]


GAME_ACTIONS = [
    {"key": "acetaminophen", "label": "对乙酰氨基酚", "short": "已核对医嘱", "icon": "pill", "tone": "coral"},
    {"key": "ibuprofen", "label": "布洛芬", "short": "已核对医嘱", "icon": "capsule", "tone": "blue"},
    {"key": "ors", "label": "口服补液盐", "short": "已核对医嘱", "icon": "cup-soda", "tone": "cyan"},
    {"key": "pharmacist", "label": "药师复核", "short": "暂停发药", "icon": "messages-square", "tone": "violet"},
    {"key": "emergency", "label": "急诊转运", "short": "立即升级处置", "icon": "ambulance", "tone": "red"},
    {"key": "quarantine", "label": "过期药隔离", "short": "禁止继续使用", "icon": "archive-x", "tone": "gold"},
]


GAME_CASES = [
    {
        "id": "rx-acetaminophen-01",
        "patient": "周女士",
        "age": "32 岁",
        "complaint": "门诊处方取药",
        "brief": "医嘱通用名：对乙酰氨基酚。身份、药名与过敏史均已核对，无异常提示。",
        "tag": "处方核对",
        "correct_action": "acetaminophen",
        "unsafe_actions": ["ibuprofen", "ors"],
        "explanation": "已有明确医嘱且核对一致，应按医嘱发放对乙酰氨基酚；游戏不展示或推断具体剂量。",
        "article_slug": "five-checks-before-taking-medicine",
    },
    {
        "id": "rx-ors-01",
        "patient": "林同学",
        "age": "16 岁",
        "complaint": "门诊处方取药",
        "brief": "医嘱通用名：口服补液盐。身份与药名一致，未记录相关过敏。",
        "tag": "处方核对",
        "correct_action": "ors",
        "unsafe_actions": ["acetaminophen", "ibuprofen"],
        "explanation": "药品通用名与医嘱一致时才能进入发放流程，不能用看起来相似的其他药品替代。",
        "article_slug": "five-checks-before-taking-medicine",
    },
    {
        "id": "rx-ibuprofen-01",
        "patient": "韩先生",
        "age": "27 岁",
        "complaint": "门诊处方取药",
        "brief": "医嘱通用名：布洛芬。药物清单完整，当前核对项未发现冲突。",
        "tag": "处方核对",
        "correct_action": "ibuprofen",
        "unsafe_actions": ["acetaminophen", "ors"],
        "explanation": "应依据已有医嘱匹配通用名，不凭包装、用途印象或他人经验换药。",
        "article_slug": "medicine-list-for-every-visit",
    },
    {
        "id": "allergy-amoxicillin-01",
        "patient": "赵奶奶",
        "age": "71 岁",
        "complaint": "处方存在过敏冲突",
        "brief": "处方中有阿莫西林，但患者药物清单记录青霉素类严重过敏史。",
        "tag": "红色核对",
        "correct_action": "pharmacist",
        "unsafe_actions": ["acetaminophen", "ibuprofen", "ors"],
        "explanation": "发现处方与严重过敏史冲突时必须暂停发药并交由药师或医生复核，不能自行替换。",
        "article_slug": "medicine-list-for-every-visit",
    },
    {
        "id": "duplicate-acetaminophen-01",
        "patient": "何先生",
        "age": "45 岁",
        "complaint": "可能重复用药",
        "brief": "处方含对乙酰氨基酚，患者同时在用一款成分表含对乙酰氨基酚的复方感冒药。",
        "tag": "成分核对",
        "correct_action": "pharmacist",
        "unsafe_actions": ["acetaminophen", "ibuprofen"],
        "explanation": "不同商品名可能含有相同成分。发现潜在重复时应暂停并请药师核对总用量和方案。",
        "article_slug": "five-checks-before-taking-medicine",
    },
    {
        "id": "cold-antibiotic-01",
        "patient": "吴同学",
        "age": "22 岁",
        "complaint": "自行要求购买抗生素",
        "brief": "流涕、咽部不适两天，无处方，要求使用上次剩下的抗生素。",
        "tag": "用药边界",
        "correct_action": "pharmacist",
        "unsafe_actions": ["acetaminophen", "ibuprofen", "ors"],
        "explanation": "普通感冒多数由病毒引起，不能在无评估和处方的情况下自行使用剩余抗生素。",
        "article_slug": "antibiotics-are-not-cold-medicine",
    },
    {
        "id": "expired-ibuprofen-01",
        "patient": "陈阿姨",
        "age": "58 岁",
        "complaint": "家庭药箱整理",
        "brief": "一盒布洛芬已经超过标示有效期，外包装还有明显受潮痕迹。",
        "tag": "药品质量",
        "correct_action": "quarantine",
        "unsafe_actions": ["ibuprofen", "acetaminophen", "ors"],
        "explanation": "过期或包装状态异常的药品应立即从可用药品中隔离，并咨询当地规范回收方式。",
        "article_slug": "five-checks-before-taking-medicine",
    },
    {
        "id": "stroke-fast-01",
        "patient": "孙爷爷",
        "age": "69 岁",
        "complaint": "突然言语不清",
        "brief": "十分钟前突然出现一侧手臂无力、嘴角歪斜和言语含糊。",
        "tag": "紧急分诊",
        "correct_action": "emergency",
        "unsafe_actions": ["acetaminophen", "ibuprofen", "ors"],
        "explanation": "这是典型的卒中危险信号。应立即呼叫急救、记录发作时间，不能先自行给药观察。",
        "article_slug": "recognize-stroke-fast",
    },
    {
        "id": "heat-confusion-01",
        "patient": "马师傅",
        "age": "52 岁",
        "complaint": "高温作业后意识异常",
        "brief": "高温环境工作后皮肤灼热、步态不稳，回答问题明显混乱。",
        "tag": "紧急分诊",
        "correct_action": "emergency",
        "unsafe_actions": ["acetaminophen", "ibuprofen", "ors"],
        "explanation": "意识异常提示严重热相关疾病，应立即急救转运并持续帮助降温，不能强行喂水。",
        "article_slug": "heat-illness-warning-signs",
    },
]


ACHIEVEMENTS = [
    {"key": "first_signal", "name": "第一束信号", "description": "完成第一篇知识文章", "icon": "radio-tower", "tone": "cyan", "bonus": 15},
    {"key": "perfect_recall", "name": "无损记忆", "description": "第一次获得文章测验满分", "icon": "brain-circuit", "tone": "violet", "bonus": 20},
    {"key": "thoughtful_voice", "name": "理性发声", "description": "发布 3 条通过审核的有效评论", "icon": "message-circle-heart", "tone": "gold", "bonus": 20},
    {"key": "first_shift", "name": "首班执勤", "description": "完成第一场急诊大作战", "icon": "stethoscope", "tone": "coral", "bonus": 20},
    {"key": "safety_first", "name": "零差错交接", "description": "以 100% 安全率完成一个班次", "icon": "shield-check", "tone": "green", "bonus": 35},
    {"key": "triage_operator", "name": "分诊行动官", "description": "累计正确处置 12 位患者", "icon": "siren", "tone": "red", "bonus": 40},
    {"key": "knowledge_path", "name": "知识路径点亮", "description": "完成 5 篇不同文章", "icon": "route", "tone": "blue", "bonus": 45},
    {"key": "vital_guardian", "name": "生命守望者", "description": "累计获得 500 Vital Points", "icon": "award", "tone": "platinum", "bonus": 0},
]


def seed_quest_content(connection):
    """Insert versioned editorial content without overwriting user progress."""
    for article in ARTICLES:
        connection.execute(
            "INSERT INTO quest_article (slug, title, dek, category, source_name, source_url, "
            "reviewer, reviewed_at, reading_minutes, cover_asset, accent, body, takeaways, quiz, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'published') "
            "ON CONFLICT(slug) DO UPDATE SET title=excluded.title, dek=excluded.dek, "
            "category=excluded.category, source_name=excluded.source_name, source_url=excluded.source_url, "
            "reviewer=excluded.reviewer, reviewed_at=excluded.reviewed_at, "
            "reading_minutes=excluded.reading_minutes, cover_asset=excluded.cover_asset, "
            "accent=excluded.accent, body=excluded.body, takeaways=excluded.takeaways, quiz=excluded.quiz",
            (
                article["slug"], article["title"], article["dek"], article["category"],
                article["source_name"], article["source_url"], article["reviewer"],
                article["reviewed_at"], article["reading_minutes"], article["cover_asset"],
                article["accent"], json.dumps(article["body"], ensure_ascii=False),
                json.dumps(article["takeaways"], ensure_ascii=False),
                json.dumps(article["quiz"], ensure_ascii=False),
            ),
        )

    for achievement in ACHIEVEMENTS:
        connection.execute(
            "INSERT INTO quest_achievement (achievement_key, name, description, icon, tone, bonus_points) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(achievement_key) DO UPDATE SET "
            "name=excluded.name, description=excluded.description, icon=excluded.icon, "
            "tone=excluded.tone, bonus_points=excluded.bonus_points",
            (
                achievement["key"], achievement["name"], achievement["description"],
                achievement["icon"], achievement["tone"], achievement["bonus"],
            ),
        )

    connection.execute(
        "INSERT OR IGNORE INTO quest_user (id, username, display_name, avatar_seed) "
        "VALUES (1, 'explorer', 'MedPro 探索者', 'aurora')"
    )
