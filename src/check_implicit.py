import json

# 1. 务必使用绝对路径，避免 FileNotFoundError
gold_file_path = r"E:\vv2\TMHG\data\dataset\jsons_zh\test.json"
# 请替换为你加了“互斥过滤”后跑出来的最新预测文件，这里用你之前的举例
pred_file_path = r"E:\vv\\DMIN\data\save\pred_zh_test_f1_35.5528ident_39.1960epo_39.json"

with open(gold_file_path, 'r', encoding='utf-8') as f:
    gold_data = {doc['doc_id']: doc for doc in json.load(f)}

with open(pred_file_path, 'r', encoding='utf-8') as f:
    pred_data = {doc['doc_id']: doc for doc in json.load(f)}


def get_quad_tuple(quad):
    # 统一极性格式
    pol = quad[6] if quad[6] in ['pos', 'neg'] else 'other'
    return tuple(quad[:6] + [pol])


def is_implicit(quad):
    # 判断是否为隐式：只要前6个坐标里有任何一个是 -1，就是隐式
    return any(pos == -1 for pos in quad[:6])


# 初始化统计字典
metrics = {
    'all': {'tp': 0, 'fp': 0, 'fn': 0},
    'exp': {'tp': 0, 'fp': 0, 'fn': 0},
    'imp': {'tp': 0, 'fp': 0, 'fn': 0}
}

for doc_id in gold_data:
    if doc_id not in pred_data:
        continue

    gold_quads = gold_data[doc_id].get('triplets', [])
    pred_quads = pred_data[doc_id].get('triplets', [])

    # 划分金标准 (Gold)
    gold_exp = set(get_quad_tuple(q) for q in gold_quads if not is_implicit(q))
    gold_imp = set(get_quad_tuple(q) for q in gold_quads if is_implicit(q))
    gold_all = gold_exp | gold_imp

    # 划分预测结果 (Pred)
    pred_exp = set(get_quad_tuple(q) for q in pred_quads if not is_implicit(q))
    pred_imp = set(get_quad_tuple(q) for q in pred_quads if is_implicit(q))
    pred_all = pred_exp | pred_imp

    # 统计显式 (Explicit)
    metrics['exp']['tp'] += len(pred_exp & gold_exp)
    metrics['exp']['fp'] += len(pred_exp - gold_exp)
    metrics['exp']['fn'] += len(gold_exp - pred_exp)

    # 统计隐式 (Implicit)
    metrics['imp']['tp'] += len(pred_imp & gold_imp)
    metrics['imp']['fp'] += len(pred_imp - gold_imp)
    metrics['imp']['fn'] += len(gold_imp - pred_imp)

    # 统计总体 (All = Explicit + Implicit)
    metrics['all']['tp'] += len(pred_all & gold_all)
    metrics['all']['fp'] += len(pred_all - gold_all)
    metrics['all']['fn'] += len(gold_all - pred_all)


def calc_scores(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p * 100, r * 100, f1 * 100


print("=" * 50)
print(" 🏆 全景透视评测报告 (测试集) 🏆")
print("=" * 50)

sections = [
    ('显式四元组 (Explicit) - 传统 Baseline 的舒适区', 'exp'),
    ('隐式四元组 (Implicit) - 本论文的核心突破区', 'imp'),
    ('总体四元组 (Overall)  - 真实业务场景大考', 'all')
]

for name, key in sections:
    tp, fp, fn = metrics[key]['tp'], metrics[key]['fp'], metrics[key]['fn']
    p, r, f1 = calc_scores(tp, fp, fn)
    total_gold = tp + fn
    total_pred = tp + fp

    print(f"\n▶ 【{name}】")
    print(f"  📌 金标准总数(分母): {total_gold} | 模型预测数: {total_pred}")
    print(f"  ✅ TP (全对): {tp} | ❌ FP (错猜): {fp} | 😭 FN (漏猜): {fn}")
    print(f"  🎯 Precision : {p:.2f}%")
    print(f"  🧲 Recall    : {r:.2f}%")
    print(f"  ⚖️ F1 Score  : {f1:.2f}%")

print("\n" + "=" * 50)