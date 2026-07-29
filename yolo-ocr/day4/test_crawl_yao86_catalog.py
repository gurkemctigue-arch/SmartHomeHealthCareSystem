# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

from crawl_yao86_catalog import (
    ParsedMedicine,
    _listing_page_count,
    balanced_page_numbers,
    classify_medicine,
    parse_detail,
    parse_listing,
)


class Yao86ParserTests(unittest.TestCase):
    def test_western_detail_extracts_only_indication_section(self) -> None:
        html = """
        <h1>阿昔洛韦片</h1>
        <li class="item category"><span class="item-name">药理分类:</span>
          <span><a>抗微生物药</a></span>/<span><a>抗病毒药</a></span>
        </li>
        <li class="item category"><span class="item-name">ATC分类:</span>
          <span><a>系统用抗病毒药</a></span>
        </li>
        <dl class="item"><dt class="item-name"><h3>【适应症】</h3></dt>
          <dd class="item-text"><span class="text"><div>用于单纯疱疹病毒感染。<br>用于带状疱疹。</div></span></dd>
        </dl>
        <dl class="item"><dt class="item-name"><h3>【用法用量】</h3></dt>
          <dd class="item-text">一次一片，一日五次。</dd>
        </dl>
        """
        parsed = parse_detail(html, "western", "https://example.test/western")
        self.assertEqual(parsed.name, "阿昔洛韦片")
        self.assertEqual(parsed.efficacy, "用于单纯疱疹病毒感染。用于带状疱疹。")
        self.assertNotIn("一次一片", parsed.efficacy)
        self.assertEqual(parsed.pharmacology, ("抗微生物药", "抗病毒药"))
        self.assertEqual(parsed.atc, ("系统用抗病毒药",))
        self.assertEqual(classify_medicine(parsed).category_id, 2)

    def test_tcm_detail_extracts_function_and_preserves_category_metadata(self) -> None:
        html = """
        <h1>安坤胶囊</h1>
        <li class="item category"><span class="item-name">药理分类:</span>
          <span><a>补益剂</a></span>/<span><a>滋阴养血</a></span>
        </li>
        <li class="item category"><span class="item-name">ATC分类:</span>
          <span><a>生殖泌尿系统药物</a></span>/<span><a>妇科用药</a></span>
        </li>
        <dl class="item"><dt class="item-name"><h3>【功能主治】</h3></dt>
          <dd class="item-text"><span class="text">滋阴清热，健脾养血。用于月经不调。</span></dd>
        </dl>
        """
        parsed = parse_detail(html, "tcm", "https://example.test/tcm")
        classification = classify_medicine(parsed)
        self.assertEqual(parsed.efficacy, "滋阴清热，健脾养血。用于月经不调。")
        self.assertEqual(classification.category_id, 14)
        self.assertGreaterEqual(classification.confidence, 0.9)

    def test_tcm_falls_back_to_tcm_instead_of_other(self) -> None:
        parsed = ParsedMedicine(
            name="补益颗粒",
            kind="tcm",
            url="https://example.test/tcm",
            efficacy="补益气血，用于气血不足所致的乏力。",
            pharmacology=("补益剂",),
            atc=(),
        )
        self.assertEqual(classify_medicine(parsed).category_id, 13)

    def test_secondary_efficacy_does_not_override_primary_cough_category(self) -> None:
        parsed = ParsedMedicine(
            name="保赤一粒金丸",
            kind="tcm",
            url="https://example.test/tcm",
            efficacy="解热镇惊，止咳化痰，通便，助消化。",
            pharmacology=("祛痰剂", "清热化痰"),
            atc=("呼吸系统药物", "咳嗽和感冒用药"),
        )
        self.assertEqual(classify_medicine(parsed).category_id, 1)

    def test_antidiarrheal_is_not_mistaken_for_laxative_parent_category(self) -> None:
        parsed = ParsedMedicine(
            name="盐酸洛哌丁胺胶囊",
            kind="western",
            url="https://example.test/western",
            efficacy="用于各种病因引起的非感染性急、慢性腹泻。",
            pharmacology=("消化系统用药", "泻药及止泻药"),
            atc=("止泻药、肠道抗炎、抗感染药", "抗胃肠动力药"),
        )
        self.assertEqual(classify_medicine(parsed).category_id, 5)

    def test_laxative_metadata_wins_over_secondary_hemorrhoid_mention(self) -> None:
        parsed = ParsedMedicine(
            name="车前番泻颗粒",
            kind="tcm",
            url="https://example.test/tcm",
            efficacy="用于便秘，也可减轻痔疮患者排便困难。",
            pharmacology=("泻下剂",),
            atc=("轻泻药",),
        )
        self.assertEqual(classify_medicine(parsed).category_id, 6)

    def test_specific_laxative_atc_wins_over_mixed_parent_category(self) -> None:
        parsed = ParsedMedicine(
            name="聚卡波非钙片",
            kind="western",
            url="https://example.test/western",
            efficacy="用于缓解便秘患者的排便异常。",
            pharmacology=("消化系统用药", "泻药及止泻药"),
            atc=("轻泻药", "膨胀性缓泻剂"),
        )
        self.assertEqual(classify_medicine(parsed).category_id, 6)

    def test_vitamin_k_atc_parent_does_not_make_drug_a_vitamin(self) -> None:
        parsed = ParsedMedicine(
            name="示例血小板生成素片",
            kind="western",
            url="https://example.test/western",
            efficacy="用于慢性免疫性血小板减少症患者。",
            pharmacology=("血液系统用药", "其他血液系统用药"),
            atc=("抗出血药", "维生素K和其它止血药"),
        )
        self.assertIsNone(classify_medicine(parsed).category_id)

    def test_systemic_skin_medicine_is_not_treated_as_topical(self) -> None:
        parsed = ParsedMedicine(
            name="示例免疫调节片",
            kind="western",
            url="https://example.test/western",
            efficacy="用于需要系统治疗的中重度斑块状银屑病。",
            pharmacology=("皮肤科用药", "其他皮肤科药"),
            atc=("免疫抑制剂",),
        )
        self.assertIsNone(classify_medicine(parsed).category_id)

    def test_systemic_antihistamine_with_skin_indications_is_not_topical(self) -> None:
        parsed = ParsedMedicine(
            name="马来酸氯苯那敏滴丸",
            kind="western",
            url="https://example.test/western",
            efficacy="用于荨麻疹、湿疹、虫咬症和过敏性鼻炎。",
            pharmacology=("皮肤科用药", "其他皮肤科药"),
            atc=("系统用抗组胺药",),
        )
        self.assertEqual(classify_medicine(parsed).category_id, 8)

    def test_skin_atc_and_topical_form_override_broad_gynecology_parent(self) -> None:
        parsed = ParsedMedicine(
            name="克霉唑涂膜",
            kind="western",
            url="https://example.test/western",
            efficacy="用于体癣、股癣、手癣和足癣。",
            pharmacology=("妇产科用药", "其他妇产科用药"),
            atc=("皮肤病用抗真菌药", "局部用抗真菌药"),
        )
        self.assertEqual(classify_medicine(parsed).category_id, 9)

    def test_gynecology_metadata_wins_over_generic_local_use(self) -> None:
        parsed = ParsedMedicine(
            name="普罗雌烯乳膏",
            kind="western",
            url="https://example.test/western",
            efficacy="用于外阴、前庭部及阴道环部因雌激素不足引起的萎缩性病变。",
            pharmacology=("激素及影响内分泌药", "雌激素、孕激素及抗孕激素"),
            atc=("性激素和生殖系统调节药", "雌激素类"),
        )
        self.assertEqual(classify_medicine(parsed).category_id, 14)

    def test_reviewed_outlier_uses_other_category(self) -> None:
        parsed = ParsedMedicine(
            name="氨肽素片",
            kind="western",
            url="https://example.test/western",
            efficacy="用于原发性血小板减少性紫癜及银屑病。",
            pharmacology=("皮肤科用药",),
            atc=("抗贫血药",),
        )
        classification = classify_medicine(parsed)
        self.assertEqual(classification.category_id, 17)
        self.assertEqual(classification.confidence, 0.99)

    def test_western_bronchodilator_uses_reviewed_other_category(self) -> None:
        parsed = ParsedMedicine(
            name="奥达特罗吸入喷雾剂",
            kind="western",
            url="https://example.test/western",
            efficacy="用于慢性阻塞性肺疾病患者的维持治疗。",
            pharmacology=("呼吸系统用药", "平喘药"),
            atc=("阻塞性气管疾病用药",),
        )
        self.assertEqual(classify_medicine(parsed).category_id, 17)

    def test_listing_rejects_ingredients_and_formula_records(self) -> None:
        html = """
        <a href="https://www.yao86.com/%E8%A5%BF%E8%8D%AF/%E9%98%BF%E6%98%94%E6%B4%9B%E9%9F%A6">阿昔洛韦</a>
        <a href="https://www.yao86.com/%E8%A5%BF%E8%8D%AF/%E9%98%BF%E6%98%94%E6%B4%9B%E9%9F%A6%E7%89%87">阿昔洛韦片</a>
        <a href="https://www.yao86.com/%E4%B8%AD%E6%88%90%E8%8D%AF/%E5%AE%89%E5%9D%A4%E6%88%90%E6%96%B9">安坤成方</a>
        """
        parsed = parse_listing(html, "western", "西药", "https://example.test/list")
        self.assertEqual([item.name for item in parsed], ["阿昔洛韦片"])

    def test_detail_extracts_product_name_but_not_registered_trademark(self) -> None:
        html = """
        <h1>阿昔洛韦片</h1>
        <dl class="item"><dt class="item-name"><h3>【药品名称】</h3></dt>
          <dd class="item-text"><span>通用名称：阿昔洛韦片<br>
          商品名称：丽科平<br>英文名称：Aciclovir Tablets</span></dd>
        </dl>
        <dl class="item"><dt class="item-name"><h3>【注册商标】</h3></dt>
          <dd class="item-text">云丰</dd>
        </dl>
        <dl class="item"><dt class="item-name"><h3>【适应症】</h3></dt>
          <dd class="item-text">用于单纯疱疹病毒感染。</dd>
        </dl>
        """
        parsed = parse_detail(html, "western", "https://example.test/western")
        self.assertEqual(parsed.aliases, ("丽科平",))
        self.assertNotIn("云丰", parsed.aliases)

    def test_empty_product_name_does_not_capture_english_name(self) -> None:
        html = """
        <h1>他扎罗汀凝胶</h1>
        <dl class="item"><dt class="item-name"><h3>【药品名称】</h3></dt>
          <dd class="item-text">通用名称：他扎罗汀凝胶<br>
          商品名称：<br>英文名称：Tazarotene Gel<br>汉语拼音：Tazhating</dd>
        </dl>
        """
        parsed = parse_detail(html, "western", "https://example.test/western")
        self.assertEqual(parsed.aliases, ())

    def test_two_character_product_name_is_rejected_for_exact_ocr_matching(self) -> None:
        html = """
        <h1>酚麻美敏片</h1>
        <dl class="item"><dt class="item-name"><h3>【药品名称】</h3></dt>
          <dd class="item-text">通用名称：酚麻美敏片<br>商品名称：泰诺</dd>
        </dl>
        """
        parsed = parse_detail(html, "western", "https://example.test/western")
        self.assertEqual(parsed.aliases, ())

    def test_listing_excludes_clinical_formulations_by_default(self) -> None:
        html = """
        <a href="https://www.yao86.com/%E8%A5%BF%E8%8D%AF/%E9%98%BF%E6%98%94%E6%B4%9B%E9%9F%A6%E7%89%87">阿昔洛韦片</a>
        <a href="https://www.yao86.com/%E8%A5%BF%E8%8D%AF/%E9%98%BF%E6%98%94%E6%B4%9B%E9%9F%A6%E6%B3%A8%E5%B0%84%E6%B6%B2">阿昔洛韦注射液</a>
        """
        home = parse_listing(html, "western", "西药", "https://example.test/list")
        all_items = parse_listing(
            html,
            "western",
            "西药",
            "https://example.test/list",
            include_clinical=True,
        )
        self.assertEqual([item.name for item in home], ["阿昔洛韦片"])
        self.assertEqual(
            [item.name for item in all_items], ["阿昔洛韦片", "阿昔洛韦注射液"]
        )

    def test_balanced_page_order_spans_the_directory_early(self) -> None:
        pages = balanced_page_numbers(100)
        self.assertEqual(len(pages), 100)
        self.assertEqual(len(set(pages)), 100)
        self.assertEqual(pages[0], 1)
        self.assertLessEqual(min(pages[:8]), 13)
        self.assertGreaterEqual(max(pages[:8]), 87)

    def test_listing_page_count_handles_encoded_chinese_paths(self) -> None:
        html = """
        <a href="https://www.yao86.com/a/%E5%85%A8%E9%83%A8%E8%A5%BF%E8%8D%AF?page=455">455</a>
        <a href="https://www.yao86.com/a/%E5%85%A8%E9%83%A8%E8%A5%BF%E8%8D%AF?page=456">456</a>
        """
        self.assertEqual(
            _listing_page_count(html, "https://www.yao86.com/a/全部西药"), 456
        )


if __name__ == "__main__":
    unittest.main()
