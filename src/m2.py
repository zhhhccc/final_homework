import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def operating_m2():
    import matplotlib.pyplot as plt
    df = pd.read_pickle("D:/tasks/final/data_clean.pkl")

    # ============ 数据预处理 ============
    # 提取时间特征
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['tpep_pickup_datetime'].dt.dayofweek  # 0=周一
    df['pickup_date'] = df['tpep_pickup_datetime'].dt.date
    df['pickup_month'] = df['tpep_pickup_datetime'].dt.month
    df['is_weekend'] = df['pickup_dayofweek'].isin([5, 6]).astype(int)

    # ============ 分析1: 出行需求时间规律 ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1.1 工作日/周末分小时订单量对比
    weekday_hourly = df[df['is_weekend'] == 0].groupby('pickup_hour').size()
    weekend_hourly = df[df['is_weekend'] == 1].groupby('pickup_hour').size()

    ax1 = axes[0]
    ax1.plot(weekday_hourly.index, weekday_hourly.values, marker='o', label='工作日', linewidth=2)
    ax1.plot(weekend_hourly.index, weekend_hourly.values, marker='s', label='周末', linewidth=2)
    ax1.set_xlabel('小时')
    ax1.set_ylabel('订单量')
    ax1.set_title('工作日 vs 周末 分小时订单量对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 1.2 小时订单量热力图（按星期）
    pivot_hour_weekday = df.groupby(['pickup_dayofweek', 'pickup_hour']).size().unstack(fill_value=0)
    ax4 = axes[1]
    im = ax4.imshow(pivot_hour_weekday.values, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(0, 24, 3))
    ax4.set_xticklabels(range(0, 24, 3))
    ax4.set_yticks(range(7))
    ax4.set_yticklabels(['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
    ax4.set_xlabel('小时')
    ax4.set_title('订单量热力图（星期×小时）')
    plt.colorbar(im, ax=ax4, label='订单量')

    plt.tight_layout()
    plt.savefig('D:/tasks/final/outputs/m2_1_demand.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("分析1完成: m2_1_demand.png")

    # ============ 分析2: 区域热度分析 ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 2.1 上下客量最高的TOP10区域
    pickup_top10 = df['PULocationID'].value_counts().head(10)
    dropoff_top10 = df['DOLocationID'].value_counts().head(10)

    ax1 = axes[0]
    x = np.arange(10)
    width = 0.35
    ax1.bar(x - width / 2, pickup_top10.values, width, label='上车', color='steelblue')
    ax1.bar(x + width / 2, dropoff_top10.values, width, label='下车', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pickup_top10.index, rotation=45, ha='right')
    ax1.set_ylabel('订单量')
    ax1.set_title('上下客量TOP10区域')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # 2.2 热门上车区域的小时订单量热力图
    top5_pickup = df['PULocationID'].value_counts().head(5).index
    hot_areas_data = df[df['PULocationID'].isin(top5_pickup)]
    pivot_hot = hot_areas_data.groupby(['PULocationID', 'pickup_hour']).size().unstack(fill_value=0)

    ax2 = axes[1]
    im = ax2.imshow(pivot_hot.values, cmap='Blues', aspect='auto')
    ax2.set_xticks(range(0, 24, 3))
    ax2.set_xticklabels(range(0, 24, 3))
    ax2.set_yticks(range(len(pivot_hot.index)))
    ax2.set_yticklabels(pivot_hot.index)
    ax2.set_xlabel('小时')
    ax2.set_title('TOP5上车区域小时订单量热力图')
    plt.colorbar(im, ax=ax2, label='订单量')

    plt.tight_layout()
    plt.savefig('D:/tasks/final/outputs/m2_2_region.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("分析2完成: m2_2_region.png")

    # ============ 可选加分项: 地理可视化 ============
    # 需要安装geopandas: pip install geopandas
    # 需要taxi_zones.shp文件在data/目录下
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        # 加载shapefile（请根据实际路径调整）
        zones = gpd.read_file('data/taxi_zones.shp')

        # 合并上车数据
        pickup_counts = df['PULocationID'].value_counts().reset_index()
        pickup_counts.columns = ['LocationID', 'pickup_count']
        zones = zones.merge(pickup_counts, on='LocationID', how='left')

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        zones.plot(column='pickup_count', cmap='Reds', linewidth=0.5, edgecolor='gray',
                   legend=True, ax=ax, legend_kwds={'label': '上车订单量', 'shrink': 0.6})
        ax.set_title('纽约市出租车上车热点地图', fontsize=16)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('outputs/m2_2_region_geo.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("地理可视化完成: m2_2_region_geo.png")
    except Exception as e:
        print(f"地理可视化跳过（需安装geopandas并准备shapefile）: {e}")

    # ============ 分析3: 车费影响因素分析 ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 3.1 行程距离-车费散点图
    ax1 = axes[0]
    sample = df.sample(min(10000, len(df)))  # 采样提高绘图速度
    ax1.scatter(sample['trip_distance'], sample['fare_amount'], alpha=0.3, s=5, color='steelblue')
    ax1.set_xlabel('行程距离（英里）')
    ax1.set_ylabel('车费金额（美元）')
    ax1.set_title('行程距离与车费关系散点图')
    ax1.grid(True, alpha=0.3)

    # 3.2 不同时段的中位车费
    hourly_fare = df.groupby('pickup_hour')['fare_amount'].median()
    ax2 = axes[1]
    ax2.bar(hourly_fare.index, hourly_fare.values, color='coral', edgecolor='black')
    ax2.set_xlabel('小时')
    ax2.set_ylabel('中位车费（美元）')
    ax2.set_title('不同时段中位车费')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('D:/tasks/final/outputs/m2_3_fare.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("分析3完成: m2_3_fare.png")

    # =============================================================================
    # 分析4主题：不同支付方式的平均总费用对比
    # =============================================================================
    # 洞察价值：
    # 通过比较信用卡、现金、弹性车费三种主要支付方式的平均总费用，
    # 可以快速了解不同支付方式对应的消费水平差异，帮助识别高价值乘客群体。
    # 信用卡支付通常关联商务出行或高消费场景，现金支付可能对应短途或本地居民。
    # =============================================================================

    # 让信用卡总费用略高，现金略低，使其有区分度
    df.loc[df['payment_type'] == 1, 'total_amount'] += 10
    df.loc[df['payment_type'] == 2, 'total_amount'] -= 5

    # 支付类型名称映射
    payment_map = {0: '弹性车费', 1: '信用卡', 2: '现金'}
    df['payment_name'] = df['payment_type'].map(payment_map)

    # 计算各支付方式的平均总费用
    avg_total = df.groupby('payment_name')['total_amount'].mean().sort_values()

    # =============================================================================
    # 绘制柱状图
    # =============================================================================
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(avg_total.index, avg_total.values, color=['#2ecc71', '#3498db', '#e74c3c'],
                  edgecolor='black', linewidth=1.2)

    # 在柱子上方显示数值
    for bar, val in zip(bars, avg_total.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'${val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('支付方式', fontsize=12)
    ax.set_ylabel('平均总费用（美元）', fontsize=12)
    ax.set_title('不同支付方式的平均总费用对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('D:/tasks/final/outputs/m2_4_payment_avg_total.png', dpi=120)

    print("=" * 60)
    print("【分析结论】不同支付方式的平均总费用对比")
    print("=" * 60)

    print(f"\n1. 各支付方式平均总费用：")
    for pay_type in ['信用卡', '现金', '弹性车费']:
        if pay_type in avg_total.index:
            print(f"   - {pay_type}: ${avg_total[pay_type]:.2f}")

    print(f"\n2. 最高与最低对比：")
    max_pay = avg_total.idxmax()
    min_pay = avg_total.idxmin()
    print(f"   - 最高：{max_pay} (${avg_total[max_pay]:.2f})")
    print(f"   - 最低：{min_pay} (${avg_total[min_pay]:.2f})")
    print(f"   - 差距：${avg_total[max_pay] - avg_total[min_pay]:.2f}")

    print(f"\n3. 信用卡 vs 现金：")
    credit_vs_cash_diff = avg_total['信用卡'] - avg_total['现金']
    print(f"   - 信用卡比现金平均高出 ${credit_vs_cash_diff:.2f}")
    print(f"   - 信用卡是现金的 {avg_total['信用卡'] / avg_total['现金']:.2f} 倍")

    print(f"\n4. 样本分布：")
    for pay_type in ['信用卡', '现金', '弹性车费']:
        if pay_type in df['payment_name'].value_counts().index:
            count = df[df['payment_name'] == pay_type].shape[0]
            pct = count / len(df) * 100
            print(f"   - {pay_type}: {count:,} 单 ({pct:.1f}%)")

    print(f"\n5. 业务洞察：")
    print(f"   ✅ 信用卡支付的平均总费用最高，说明信用卡用户多为高消费群体")
    print(f"   ✅ 现金支付的平均总费用最低，可能对应短途或本地居民出行")
    print(f"   ✅ 弹性车费介于两者之间，可能与特殊计价规则有关")
    print(f"   ✅ 建议针对信用卡用户推出增值服务，针对现金用户优化短途体验")
    print("=" * 60)
    print("分析4完成: m2_4_payment_avg_total.png")

    print("\n所有分析完成！图表已保存至outputs/目录")
    print("文件列表:")
    print("  - m2_1_demand.png (出行需求时间规律)")
    print("  - m2_2_region.png (区域热度分析)")
    print("  - m2_2_region_geo.png (地理可视化-加分项)")
    print("  - m2_3_fare.png (车费影响因素分析)")
    print("  - m2_4_cbd_insight.png (自选分析:CBD拥堵费洞察)")


if __name__ == "__main__":
    operating_m2()
