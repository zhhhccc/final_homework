import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os

df=pd.read_pickle("D:/tasks/final/data.pkl")

def generate_data_quality_report(df, output_path='D:/tasks/final/outputs/data_quality_report.csv'):
    """
    生成数据质量报告，包含缺失率、异常值统计等

    Parameters:
    -----------
    df : pandas.DataFrame
        输入数据框
    output_path : str
        报告保存路径
    """
    report_data = []

    for col in df.columns:
        col_info = {
            'column_name': col,
            'data_type': str(df[col].dtype),
            'total_count': len(df[col]),
            'non_null_count': df[col].count(),
            'null_count': df[col].isnull().sum(),
            'null_rate': f"{df[col].isnull().sum() / len(df[col]) * 100:.2f}%",
            'unique_count': df[col].nunique(),
            'distinct_rate': f"{df[col].nunique() / len(df[col]) * 100:.2f}%",
        }

        # 数值型字段的额外统计
        if pd.api.types.is_numeric_dtype(df[col]):
            # 异常值检测：使用IQR方法（四分位距法）
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 计算异常值数量
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outlier_mask.sum()

            # 补充统计信息
            col_info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'q1': Q1,
                'q3': Q3,
                'iqr': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': outlier_count,
                'outlier_rate': f"{outlier_count / len(df[col]) * 100:.2f}%",
                # 额外的Z-score异常值检测
                'zscore_outlier_count': ((np.abs(stats.zscore(df[col].dropna())) > 3).sum()),
            })
        else:
            # 非数值型字段
            col_info.update({
                'min': None,
                'max': None,
                'mean': None,
                'std': None,
                'median': None,
                'q1': None,
                'q3': None,
                'iqr': None,
                'lower_bound': None,
                'upper_bound': None,
                'outlier_count': None,
                'outlier_rate': None,
                'zscore_outlier_count': None,
            })

        report_data.append(col_info)

    # 生成报告DataFrame并保存
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"数据质量报告已保存至: {output_path}")

    return report_df

def data_cleaning_strategy(df):
    """
    数据清洗策略 - 每一步都包含详细注释说明理由

    Parameters:
    -----------
    df : pandas.DataFrame
        原始数据框

    Returns:
    --------
    pandas.DataFrame: 清洗后的数据框
    """
    # 创建副本，避免修改原始数据
    df_clean = df.copy()

    # ========== 步骤1: 删除全为空的列 ==========
    # 理由：全为空的列没有任何信息价值，保留只会增加数据维度，降低分析效率
    all_null_cols = df_clean.columns[df_clean.isnull().all()].tolist()
    if all_null_cols:
        df_clean = df_clean.drop(columns=all_null_cols)
        print(f"已删除全为空列: {all_null_cols}")

    # ========== 步骤2: 处理缺失值 ==========
    for col in df_clean.columns:
        null_rate = df_clean[col].isnull().sum() / len(df_clean)

        # 2.1 缺失率 > 60% 的列直接删除
        # 理由：缺失率过高意味着该字段数据收集存在严重问题，无法通过插补有效恢复，强行保留会引入大量噪声
        if null_rate > 0.6:
            df_clean = df_clean.drop(columns=[col])
            print(f"缺失率 {null_rate:.2%} > 60%，已删除列: {col}")
            continue

        # 2.2 数值型字段处理
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            # 2.2.1 缺失率 < 5% 且数据近似正态分布 -> 使用均值填充
            # 理由：低缺失率下，均值能较好地代表数据中心趋势，且不会对分布造成显著影响
            if null_rate < 0.05:
                # 检查数据是否近似正态（使用偏度判断）
                skewness = df_clean[col].skew()
                if abs(skewness) < 1:  # 偏度绝对值小于1认为近似正态
                    fill_value = df_clean[col].mean()
                    df_clean[col].fillna(fill_value, inplace=True)
                    print(f"列 {col}: 缺失率 {null_rate:.2%}，使用均值 {fill_value:.2f} 填充")
                else:
                    # 偏态分布使用中位数填充，更稳健
                    fill_value = df_clean[col].median()
                    df_clean[col].fillna(fill_value, inplace=True)
                    print(f"列 {col}: 缺失率 {null_rate:.2%}（偏态分布），使用中位数 {fill_value:.2f} 填充")

            # 2.2.2 缺失率 5%-60% -> 使用中位数填充
            # 理由：中位数对异常值不敏感，适合缺失率中等的情况，避免极端值影响填充结果
            elif 0.05 <= null_rate <= 0.6:
                fill_value = df_clean[col].median()
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"列 {col}: 缺失率 {null_rate:.2%}，使用中位数 {fill_value:.2f} 填充")

        # 2.3 非数值型（类别型）字段处理
        else:
            # 使用众数填充，保留最常出现的类别
            # 理由：类别型数据没有均值/中位数概念，众数是唯一合理的中心趋势度量
            if null_rate < 0.6:  # 缺失率低于60%才填充
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "UNKNOWN"
                df_clean[col].fillna(mode_value, inplace=True)
                print(f"列 {col}: 缺失率 {null_rate:.2%}，使用众数 '{mode_value}' 填充")

    # ========== 步骤3: 异常值处理（使用IQR方法） ==========
    # 理由：IQR方法基于四分位数，对数据分布没有正态假设要求，适用于各种分布形态的数据
    for col in df_clean.select_dtypes(include=[np.number]).columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 检测异常值
        outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            # 3.1 如果异常值比例 < 5%，使用中位数替换
            # 理由：低比例异常值可能是数据录入错误，使用中位数替换可以保留数据完整性且不影响整体分布
            if outlier_count / len(df_clean) < 0.05:
                median_val = df_clean[col].median()
                df_clean.loc[outlier_mask, col] = median_val
                print(
                    f"列 {col}: 发现 {outlier_count} 个异常值 ({outlier_count / len(df_clean):.2%})，已用中位数 {median_val:.2f} 替换")

            # 3.2 如果异常值比例 >= 5%，进行缩尾处理（Winsorize）
            # 理由：高比例异常值可能代表数据本身具有长尾分布，直接替换会丢失信息，缩尾处理能保留分布形状
            else:
                # 将异常值缩到边界值
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                print(
                    f"列 {col}: 发现 {outlier_count} 个异常值 ({outlier_count / len(df_clean):.2%})，已缩尾处理到 [{lower_bound:.2f}, {upper_bound:.2f}]")

    # ========== 步骤4: 删除重复行 ==========
    # 理由：重复记录会造成数据偏差，导致统计结果失真，且占用存储空间
    initial_len = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_len - len(df_clean)
    if duplicates_removed > 0:
        print(f"已删除 {duplicates_removed} 条重复记录")

    # ========== 步骤5: 数据格式标准化 ==========
    # 5.1 去除字符串字段的前后空格
    # 理由：避免因空格导致的分组、匹配等操作失败，保证数据一致性
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # 将空字符串转为NaN后再填充（处理特殊空白情况）
        df_clean[col] = df_clean[col].replace('', np.nan)
        # 再次填充（保留原清洗逻辑）
        if df_clean[col].isnull().any():
            mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "UNKNOWN"
            df_clean[col].fillna(mode_val, inplace=True)

    # 5.2 日期时间字段标准化
    # 理由：统一日期格式便于后续时间序列分析和特征工程
    date_keywords = ['date', 'time', 'dt', '日期', '时间']
    for col in df_clean.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            try:
                df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
                print(f"列 {col} 已转换为日期时间格式")
            except:
                pass  # 转换失败则保持原样

    return df_clean

def extract_time_features(df):
    """
    从行程时间中提取基础时间特征
    """
    df_features = df.copy()

    # 确保日期时间为datetime类型
    df_features['tpep_pickup_datetime'] = pd.to_datetime(df_features['tpep_pickup_datetime'])

    # 1. 提取小时 (0-23)
    df_features['pickup_hour'] = df_features['tpep_pickup_datetime'].dt.hour

    # 2. 提取星期 (0=Monday, 6=Sunday)
    df_features['pickup_dayofweek'] = df_features['tpep_pickup_datetime'].dt.dayofweek

    # 3. 提取星期名称
    df_features['pickup_dayname'] = df_features['tpep_pickup_datetime'].dt.day_name()

    # 4. 是否周末 (周六=5, 周日=6)
    df_features['is_weekend'] = df_features['pickup_dayofweek'].isin([5, 6]).astype(int)

    # 5. 是否高峰时段
    # 定义高峰时段：工作日早高峰 7:00-9:00，晚高峰 17:00-19:00
    # 周末高峰时段：10:00-12:00, 15:00-18:00

    def is_peak_hour(row):
        hour = row['pickup_hour']
        is_weekend = row['is_weekend']

        if is_weekend:
            # 周末高峰
            return 1 if (10 <= hour <= 12) or (15 <= hour <= 18) else 0
        else:
            # 工作日高峰
            return 1 if (7 <= hour <= 9) or (17 <= hour <= 19) else 0

    df_features['is_peak_hour'] = df_features.apply(is_peak_hour, axis=1)

    # 6. 提取月份
    df_features['pickup_month'] = df_features['tpep_pickup_datetime'].dt.month

    # 7. 提取季度
    df_features['pickup_quarter'] = df_features['tpep_pickup_datetime'].dt.quarter

    # 8. 提取日期（用于后续分组）
    df_features['pickup_date'] = df_features['tpep_pickup_datetime'].dt.date

    print("时间特征提取完成！新增特征列：")
    print(['pickup_hour', 'pickup_dayofweek', 'pickup_dayname', 'is_weekend',
           'is_peak_hour', 'pickup_month', 'pickup_quarter', 'pickup_date'])

    return df_features

def derived_character(df):
    #小费比率，可用于预测顾客是否会再叫车
    df['tax_ratio']=df['tip_amount']/df['total_amount']

    #每英里收入
    df['revenue_per_mile']=df['total_amount']/df['trip_distance']

    print("衍生特征提取完成！新增特征列：")
    print(['tax_ratio', 'revenue_per_mile'])

    return (df)

def operating_m1():
    # 1. 生成数据质量报告
    print("=" * 50)
    print("生成数据质量报告...")
    report = generate_data_quality_report(df)
    print(report.head())

    # 2. 数据清洗
    print("\n" + "=" * 50)
    print("开始数据清洗...")
    df_cleaned = data_cleaning_strategy(df)
    df_cleaned.to_pickle("D:/tasks/final/data_clean.pkl")

    # 清洗后验证
    print("\n" + "=" * 50)
    print("清洗完成！")
    print(f"原始数据形状: {df.shape}")
    print(f"清洗后数据形状: {df_cleaned.shape}")
    print(f"清洗后数据预览:\n{df_cleaned.head()}")

    #生成提取时间特征和衍生特征的csv文件
    derived_character(extract_time_features(df)).to_csv("D:/tasks/final/outputs/derive.csv")