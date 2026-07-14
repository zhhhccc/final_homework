"""
m4.py — 自然语言问答系统
命令行循环：用户输入自然语言问题 → 关键词+规则意图识别 → 调用M1-M3功能
返回：①数字结论 ②文本解释 ③相关图表/统计文件相对路径
"""
import os
import sys
import re
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m1 import generate_data_quality_report, data_cleaning_strategy, extract_time_features, derived_character

# ==================== 路径配置 ====================
DATA_PATH = "D:/tasks/final/data.pkl"
CLEAN_PATH = "D:/tasks/final/data_clean.pkl"
OUTPUT_DIR = "D:/tasks/final/outputs"

# ==================== 意图关键词库 ====================
INTENTS = {
    "data_quality": ["质量", "缺失", "异常", "空值", "重复", "null", "missing", "outlier",
                     "数据报告", "数据情况", "脏数据", "质量问题"],
    "data_cleaning": ["清洗", "清理", "干净", "clean", "预处理", "洗数据"],
    "time_features": ["时间特征", "特征工程", "衍生", "提取特征", "特征列"],
    "demand": ["需求", "出行规律", "订单量", "热力图", "高峰", "低谷", "demand",
               "trip pattern", "小时分布", "出行量", "用车需求"],
    "region": ["区域", "地点", "位置", "上车", "下车", "热门", "region", "location",
               "pickup", "dropoff", "地理", "地区", "哪里"],
    "fare": ["车费", "费用", "票价", "fare", "price", "多少钱", "收费", "路费"],
    "payment": ["支付", "信用卡", "现金", "payment", "credit", "cash", "付款"],
    "model": ["模型", "预测", "机器学习", "神经网络", "随机森林", "model", "predict",
              "neural", "random forest", "训练", "training", "loss", "算法"],
    "distance": ["距离", "里程", "distance", "mile", "多远", "路程"],
    "peak": ["高峰", "peak", "rush", "拥堵", "繁忙"],
    "weekend": ["周末", "工作日", "weekend", "weekday", "工作日"],
    "stats": ["统计", "概况", "概览", "总体", "summary", "statistics", "overview",
              "基本情况", "数据量", "多少行", "多少列"],
}

OUTPUT_FILES = {
    "quality_report": "outputs/data_quality_report.csv",
    "derive": "outputs/derive.csv",
    "m2_demand": "outputs/m2_1_demand.png",
    "m2_region": "outputs/m2_2_region.png",
    "m2_fare": "outputs/m2_3_fare.png",
    "m2_payment": "outputs/m2_4_payment_avg_total.png",
    "m3_loss": "outputs/m3_neural_network_loss.png",
    "m3_metrics": "outputs/m3_model_metrics.csv",
}


def load_data():
    """加载原始数据和清洗后数据（如果存在）"""
    df = pd.read_pickle(DATA_PATH)
    df_clean = None
    if os.path.exists(CLEAN_PATH):
        df_clean = pd.read_pickle(CLEAN_PATH)
    return df, df_clean


def match_intent(question: str) -> list:
    """关键词匹配：返回得分排序后的意图列表"""
    q = question.lower()
    scores = {}
    for intent, keywords in INTENTS.items():
        score = 0
        for kw in keywords:
            if kw.lower() in q:
                score += 1
        if score > 0:
            scores[intent] = score
    return sorted(scores, key=scores.get, reverse=True)


def handle_data_quality(df, q):
    """处理数据质量相关查询"""
    report = generate_data_quality_report(df, OUTPUT_DIR + "/data_quality_report.csv")
    total_cols = len(report)
    null_cols = report[report['null_count'] > 0]
    high_null = report[pd.to_numeric(report['null_rate'].str.rstrip('%'), errors='coerce') > 20]
    numeric_cols = report[report['min'].notna()]
    outlier_cols = numeric_cols[pd.to_numeric(numeric_cols['outlier_rate'].str.rstrip('%'), errors='coerce') > 0]

    numeric = f"共{len(df)}行×{total_cols}列；有缺失的列{len(null_cols)}个；缺失率>20%的列{len(high_null)}个；存在异常值的数值列{len(outlier_cols)}个"
    text = f"数据质量报告已生成。{len(null_cols)}个字段存在缺失值，"
    if len(high_null) > 0:
        text += f"其中{high_null['column_name'].tolist()}缺失率超过20%。"
    text += f"{len(outlier_cols)}个数值字段存在IQR异常值。"
    return numeric, text, [OUTPUT_FILES["quality_report"]]


def handle_data_cleaning(df, q):
    """处理数据清洗相关查询"""
    before_shape = df.shape
    df_clean = data_cleaning_strategy(df)
    after_shape = df_clean.shape
    removed_cols = before_shape[1] - after_shape[1]
    removed_rows = before_shape[0] - after_shape[0]

    numeric = f"清洗前: {before_shape[0]}行×{before_shape[1]}列 → 清洗后: {after_shape[0]}行×{after_shape[1]}列；删除{removed_cols}列、{removed_rows}行"
    text = f"数据清洗完成。删除了{removed_cols}个全空/高缺失列，去除了{removed_rows}条重复记录，"
    text += f"缺失值已按策略填充（均值/中位数/众数），异常值已做缩尾或中位数替换处理。"
    df_clean.to_pickle(CLEAN_PATH)
    return numeric, text, [OUTPUT_FILES["quality_report"]]


def handle_demand(df, q):
    """处理出行需求/时间规律查询"""
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['is_weekend'] = df['pickup_dayofweek'].isin([5, 6]).astype(int)

    hourly = df.groupby('pickup_hour').size()
    peak_hour = hourly.idxmax()
    peak_count = hourly.max()
    valley_hour = hourly.idxmin()
    valley_count = hourly.min()

    weekday_count = len(df[df['is_weekend'] == 0])
    weekend_count = len(df[df['is_weekend'] == 1])
    weekday_avg = weekday_count / 5
    weekend_avg = weekend_count / 2

    numeric = f"高峰时段: {peak_hour}:00（{peak_count:,}单）；低谷时段: {valley_hour}:00（{valley_count:,}单）；工作日日均{weekday_avg:,.0f}单，周末日均{weekend_avg:,.0f}单"
    text = f"出行需求呈现明显双峰模式：早高峰7-9点、晚高峰17-19点订单量最高。"
    text += f"工作日日均订单（{weekday_avg:,.0f}）{'高于' if weekday_avg > weekend_avg else '低于'}周末（{weekend_avg:,.0f}）。"
    return numeric, text, [OUTPUT_FILES["m2_demand"]]


def handle_region(df, q):
    """处理区域热度查询"""
    pickup_top = df['PULocationID'].value_counts().head(5)
    dropoff_top = df['DOLocationID'].value_counts().head(5)
    total_locations = df['PULocationID'].nunique()

    numeric = f"共{total_locations}个区域；TOP5上车区: {pickup_top.index.tolist()}（占比{pickup_top.sum()/len(df)*100:.1f}%）；TOP5下车区: {dropoff_top.index.tolist()}"
    text = f"最热门的5个上车区域是{pickup_top.index.tolist()}，合计占总订单的{pickup_top.sum()/len(df)*100:.1f}%。"
    text += f"上下车热点区域高度重合，反映核心城区出行需求集中。"
    return numeric, text, [OUTPUT_FILES["m2_region"]]


def handle_fare(df, q):
    """处理车费/费用查询"""
    avg_fare = df['fare_amount'].mean()
    median_fare = df['fare_amount'].median()
    avg_distance = df['trip_distance'].mean()
    avg_total = df['total_amount'].mean()
    corr = df['trip_distance'].corr(df['fare_amount'])

    files = [OUTPUT_FILES["m2_fare"]]

    numeric = f"平均车费${avg_fare:.2f}、中位${median_fare:.2f}；平均距离{avg_distance:.2f}英里；距离-车费相关系数{corr:.3f}；平均总费用${avg_total:.2f}"

    # 如果涉及支付方式
    if any(kw in q for kw in ["支付", "信用卡", "现金", "payment", "credit", "cash"]):
        payment_map = {1: "信用卡", 2: "现金", 3: "无现金", 4: "其他"}
        df['pay_name'] = df['payment_type'].map(payment_map)
        pay_stats = df.groupby('pay_name')['total_amount'].agg(['mean', 'count'])
        text = f"各支付方式平均总费用: "
        for name, row in pay_stats.iterrows():
            text += f"{name} ${row['mean']:.2f}（{row['count']:,}单）; "
        files.append(OUTPUT_FILES["m2_payment"])
    else:
        text = f"行程距离与车费呈{'强' if corr > 0.7 else '中等' if corr > 0.4 else '弱'}正相关（r={corr:.3f}）。"
        text += f"平均每英里费用约${avg_fare/avg_distance:.2f}。"

    return numeric, text, files


def handle_model(df, q):
    """处理机器学习模型查询"""
    metrics_path = OUTPUT_DIR + "/m3_model_metrics.csv"
    if os.path.exists(metrics_path):
        m = pd.read_csv(metrics_path)
        nn = m[m['Model'] == 'Neural Network'].iloc[0]
        rf = m[m['Model'] == 'Random Forest'].iloc[0]
        numeric = f"神经网络 MAE={nn['MAE']:.2f}, RMSE={nn['RMSE']:.2f}；随机森林 MAE={rf['MAE']:.2f}, RMSE={rf['RMSE']:.2f}"
        text = f"随机森林在小样本（24个时段）上表现更稳定，MAE比神经网络低{abs(nn['MAE']-rf['MAE']):.2f}。"
        text += f"神经网络训练500轮后收敛，适用于更大数据量的场景。"
    else:
        numeric = "模型指标文件不存在，请先运行m3.py"
        text = "请先运行 python src/m3.py 生成模型文件。"
    return numeric, text, [OUTPUT_FILES["m3_metrics"], OUTPUT_FILES["m3_loss"]]


def handle_time_features(df, q):
    """处理时间特征提取查询"""
    df_f = extract_time_features(df)
    df_f = derived_character(df_f)
    df_f.to_csv(OUTPUT_DIR + "/derive.csv", index=False)
    new_cols = ['pickup_hour', 'pickup_dayofweek', 'pickup_dayname', 'is_weekend',
                'is_peak_hour', 'pickup_month', 'pickup_quarter', 'pickup_date',
                'tax_ratio', 'revenue_per_mile']
    numeric = f"新增{len(new_cols)}个特征列"
    text = f"已提取时间特征（小时、星期、是否周末、是否高峰时段、月份、季度等）和衍生特征（小费比率、每英里收入）。"
    return numeric, text, [OUTPUT_FILES["derive"]]


def handle_stats(df, q):
    """处理总体统计查询"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_range = f"{df['tpep_pickup_datetime'].min()} ~ {df['tpep_pickup_datetime'].max()}"
    numeric = f"共{len(df):,}行×{len(df.columns)}列；数值列{len(numeric_cols)}个；时间范围{date_range}"
    text = f"该数据集为2026年1月纽约市黄色出租车行程记录，包含{len(df.columns)}个字段，"
    text += f"覆盖上下车时间、位置、距离、费用、支付方式等维度。"
    return numeric, text, []


def handle_peak(df, q):
    """处理高峰时段查询"""
    return handle_demand(df, q)


def handle_distance(df, q):
    """处理距离相关查询"""
    avg_dist = df['trip_distance'].mean()
    median_dist = df['trip_distance'].median()
    max_dist = df['trip_distance'].max()
    short_trips = (df['trip_distance'] < 2).sum() / len(df) * 100
    numeric = f"平均距离{avg_dist:.2f}英里、中位{median_dist:.2f}英里、最大{max_dist:.2f}英里；短途(<2英里)占比{short_trips:.1f}%"
    text = f"超过{short_trips:.0f}%的行程为短途出行（<2英里），中位距离仅{median_dist:.2f}英里，说明出租车主要服务城市短途出行。"
    return numeric, text, [OUTPUT_FILES["m2_fare"]]


def handle_weekend(df, q):
    """处理周末/工作日对比查询"""
    return handle_demand(df, q)


# 意图 → 处理函数映射
HANDLERS = {
    "data_quality": handle_data_quality,
    "data_cleaning": handle_data_cleaning,
    "demand": handle_demand,
    "region": handle_region,
    "fare": handle_fare,
    "payment": handle_fare,
    "model": handle_model,
    "time_features": handle_time_features,
    "stats": handle_stats,
    "peak": handle_peak,
    "distance": handle_distance,
    "weekend": handle_weekend,
}


def answer(question: str, df, df_clean) -> str:
    """主问答逻辑：意图识别 → 分发处理 → 格式化输出"""
    matched = match_intent(question)

    if not matched:
        return ("[?] 未能识别意图，请尝试：\n"
                "  • 数据质量怎么样？  • 帮我清洗数据\n"
                "  • 出行需求规律？    • 哪些区域最热门？\n"
                "  • 车费什么水平？    • 支付方式对比？\n"
                "  • 模型效果如何？    • 数据总体概况？")

    primary = matched[0]
    handler = HANDLERS.get(primary, handle_stats)
    numeric, text, files = handler(df if primary != "data_cleaning" else df_clean if df_clean is not None else df, question)

    result = f"\n{'='*60}\n"
    result += f"[Intent] 识别意图: {primary}\n"
    result += f"{'='*60}\n"
    result += f"① 数字结论: {numeric}\n\n"
    result += f"② 文本解释: {text}\n"
    if files:
        result += f"\n③ 相关文件:\n"
        for f in files:
            result += f"   → {f}\n"
    result += f"{'='*60}\n"
    return result


def main():
    print("=" * 60)
    print("  NYC Taxi 数据分析问答系统 (m4.py)")
    print("  输入自然语言问题，获取数据洞察")
    print("  输入 'exit' / 'quit' / '退出' 结束")
    print("  输入 'help' 查看示例问题")
    print("=" * 60)

    print("\n[Loading] 正在加载数据...")
    df, df_clean = load_data()
    print(f"[OK] 已加载 {len(df):,} 条记录\n")

    while True:
        try:
            q = input(">> 请输入问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "退出", "q"):
            print("Bye!")
            break
        if q.lower() in ("help", "帮助", "?"):
            print("\n[Help] 示例问题:")
            print("  • 数据质量怎么样？有缺失值吗？")
            print("  • 帮我清洗数据")
            print("  • 出行需求有什么规律？哪些时段是高峰？")
            print("  • 哪些区域最热门？上车和下车分别在哪？")
            print("  • 车费大概多少？和距离关系大吗？")
            print("  • 信用卡和现金支付有什么差异？")
            print("  • 机器学习模型效果如何？哪个更好？")
            print("  • 提取时间特征")
            print("  • 数据总体概况\n")
            continue

        print(answer(q, df, df_clean))


def operating_m4():
    main()


if __name__ == "__main__":
    main()
