import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import re

'''
计算混淆矩阵及相关指标（精确度、召回率、F1 分数）
input:  analyzed_paper.xlsx
output: final_paper.xlsx
'''


def calculate_confusion_matrix(input_path,llm):
    df = pd.read_excel(input_path)
    n = df['Overall Response'].notna().sum()
    # 如果指定了 n，则只取前 n 行数据
    df_n = df.head(n) if n is not None else df

    if df_n['Label'].isna().any():
        print("警告：Label 中存在 NaN 值！")

    missing = df[df['Label C'].isnull()]
    print(missing)
    # 提取实际标签和预测标签
    y_true = df_n['Label']  # 总体实际标签
    y_pred = df_n['Overall Response']  # 总体预测标签
    y_true_a = df_n['Label A']
    y_pred_a = df_n['Response A']
    y_true_b = df_n['Label B']
    y_pred_b = df_n['Response B']
    y_true_c = df_n['Label C']
    y_pred_c = df_n['Response C']

    result = []

    # 计算总体混淆矩阵\精确度、召回率和 F1 分数
    if y_true.dtype in ['int64', 'float64'] and y_pred.dtype in ['int64', 'float64']:
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        result.append(f"计算的文献数量: {n}篇\n")
        result.append(f"总体混淆矩阵: {cm}")
        result.append(f"精确度 (Precision): {precision:.3f}")
        result.append(f"召回率 (Recall): {recall:.3f}")
        result.append(f"F1分数: {f1:.3f}\n\n")
    else:
        result.append("总体指标无法计算，因为数据中没有有效的标签或预测值。\n\n")


    # 计算A部分的混淆矩阵和指标
    if y_true_a.dtype in ['int64', 'float64'] and y_pred_a.dtype in ['int64', 'float64']:
        cm_a = confusion_matrix(y_true_a, y_pred_a, labels=[1, 0])
        precision_a = precision_score(y_true_a, y_pred_a)
        recall_a = recall_score(y_true_a, y_pred_a)
        f1_a = f1_score(y_true_a, y_pred_a)

        result.append(f"A部分混淆矩阵: {cm_a}")
        result.append(f"精确度 (Precision): {precision_a:.3f}")
        result.append(f"召回率 (Recall): {recall_a:.3f}")
        result.append(f"F1分数: {f1_a:.3f}\n\n")
    else:
        result.append("A部分 指标无法计算，因为数据中没有有效的标签或预测值。\n\n")

    # 计算 B 部分的混淆矩阵和指标
    if y_true_b.dtype in ['int64', 'float64'] and y_pred_b.dtype in ['int64', 'float64']:

        cm_b = confusion_matrix(y_true_b, y_pred_b, labels=[1, 0])
        precision_b = precision_score(y_true_b, y_pred_b)
        recall_b = recall_score(y_true_b, y_pred_b)
        f1_b = f1_score(y_true_b, y_pred_b)

        result.append(f"B部分混淆矩阵: {cm_b}")
        result.append(f"精确度 (Precision): {precision_b:.3f}")
        result.append(f"召回率 (Recall): {recall_b:.3f}")
        result.append(f"F1分数: {f1_b:.3f}\n\n")
    else:
        result.append("B部分 指标无法计算，因为数据中没有有效的标签或预测值。\n\n")

    # 计算 C 部分的混淆矩阵和指标
    if y_true_c.dtype in ['int64', 'float64'] and y_pred_c.dtype in ['int64', 'float64']:
        cm_c = confusion_matrix(y_true_c, y_pred_c, labels=[1, 0])
        precision_c = precision_score(y_true_c, y_pred_c)
        recall_c = recall_score(y_true_c, y_pred_c)
        f1_c = f1_score(y_true_c, y_pred_c)

        result.append(f"C部分混淆矩阵: {cm_c}")
        result.append(f"精确度 (Precision): {precision_c:.3f}")
        result.append(f"召回率 (Recall): {recall_c:.3f}")
        result.append(f"F1分数: {f1_c:.3f}")
    else:
        result.append("C部分 指标无法计算，因为数据中没有有效的标签或预测值。\n\n")

    df.loc[df.index[1], 'Metrics'] = '\n'.join(result)  # 只在Metrics列添加结果

    prompt_version = re.search(r'promptv(\d+)', input_path).group(0)
    time = re.search(r'\d{6}-\d{2}-\d{2}', input_path).group(0)  # 匹配6位日期+2位时间
    out_path = f"E:\\desktop\\RO\\{llm}\\5_final_paper_{llm}_{prompt_version}_{time}.xlsx"
    df.to_excel(out_path, index=False)
    print(out_path)
    os.startfile(out_path)


def screen_different_paper(tech, input_path,llm):
    df = pd.read_excel(input_path)
    df_new = df[df['Label'] != df['Overall Response']]
    # df_new = df_new.drop(['Response A', 'Response B', 'Response C'], axis=1)

    # 让索引所有数值加 2，以匹配原始文献的行数、方便查找
    df_new.index = df_new.index + 2

    prompt_version = re.search(r'promptv(\d+)', input_path).group(0)
    output_path = f"E:\\Desktop\\{tech}\\{llm}\\6_error_paper_{llm}_{prompt_version}.xlsx"
    df_new.to_excel(output_path)
    os.startfile(output_path)


def screen_hard_paper(input_path, llm):
    df = pd.read_excel(input_path)

    if df['Label A'].dtype == 'float64':
        df_new = df[(df['Label A'] == df['Response A']) &
                    (df['Label B'] == df['Response B']) &
                    (df['Label C'] == df['Response C'])]
    else:
        df_new = df[(df['Label'] == df['Overall Response']) & (df['Note'].notna())]
        df_new = df_new.drop(['Response A', 'Response B', 'Response C'], axis=1)

    # 让索引所有数值加 2，以匹配原始文献的行数、方便查找
    df_new.index = df_new.index + 2

    prompt_version = re.search(r'promptv(\d+)', input_path, re.IGNORECASE).group(0)
    output_path = f"E:\\Desktop\\RO\\{llm}\\7_hard_paper_{llm}_{prompt_version}.xlsx"
    df_new.to_excel(output_path)
    # os.startfile(output_path)


# 查找遗漏的文献，即FN，pred是0 但label是1
def screen_missing_paper(input_path,llm):
    df = pd.read_excel(input_path)
    df_new = df[(df['Label'] == 1) &
                (df['Overall Response'] == 0)]

    # 让索引所有数值加 2，以匹配原始文献的行数、方便查找
    df_new.index = df_new.index + 2

    prompt_version = re.search(r'promptv(\d+)', input_path, re.IGNORECASE).group(0)
    output_path = f"E:\\Desktop\\RO\\{llm}\\8_missing_paper_{llm}_{prompt_version}.xlsx"
    df_new.to_excel(output_path)
    os.startfile(output_path)


# 查找错误的文献，即FP，pred是1 但label是0
def screen_error_paper(input_path, llm):
    df = pd.read_excel(input_path)
    df_new = df[(df['Label'] == 0) & (df['Overall Response'] == 1)]

    # 让索引所有数值加 2，以匹配原始文献的行数、方便查找
    df_new.index = df_new.index + 2

    prompt_version = re.search(r'promptv(\d+)', input_path, re.IGNORECASE).group(0)
    output_path = f"E:\\Desktop\\RO\\{llm}\\9_error_paper_{llm}_{prompt_version}.xlsx"
    df_new.to_excel(output_path)
    os.startfile(output_path)


def main():
    add = '分层抽样'
    llm = 'deepseek'
    tech = 'EO'
    time = '250801-17-24'
    prompt_version = 'promptv2'
    input_path = f"E:\\desktop\\{tech}\\{llm}\\5_final_paper_{add}_{llm}_{prompt_version}_{time}.xlsx"
    # screen_different_paper(tech, input_path, llm)
    # calculate_confusion_matrix(input_path, llm)
    # screen_missing_paper(input_path, llm)
    screen_error_paper(input_path, llm)


if __name__ == "__main__":
    main()
