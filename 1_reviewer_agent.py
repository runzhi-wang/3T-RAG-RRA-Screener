import pandas as pd
import json
from openai import OpenAI
import re
import os
import time
import concurrent.futures  # 用于并行数-时间处理
from datetime import datetime
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from RAG import simple_query_system
from RAG import Config
from RAG import load_corpus_from_excel
from RAG_keyword import keyword_query_system
'''
从wos新表格中，抽取题目和摘要投喂给大模型，进行分类，并结构化输出大模型的答案和理由
'''


# 从初始excel文件中提取文献的指定信息列
def extract_from_excel(tech, excel_path):
    if tech == 'EO':
        columns_to_read = ['Article Title', 'Source Title', 'Abstract', 'Publication Year', 'DOI', 'DOI Link',
                           'Label', 'Label A', 'Label B1', 'Label B2', 'Label C', 'Note', '摘要', '原始索引']
        df = pd.read_excel(excel_path, usecols=columns_to_read, nrows=None)

        # 手动定义列的顺序，将新增列插入到中间
        columns_order = ['Source Title', 'Article Title', 'Label', 'Label A', 'Label B1', 'Label B2', 'Label C',
                         '摘要', 'Abstract', 'Publication Year', 'DOI', 'DOI Link', 'Note', '原始索引']
        df = df[columns_order]  # 重新排列顺序
        return df
    else:
        columns_to_read = ['Article Title', 'Source Title', 'Abstract', 'Publication Year', 'DOI', 'DOI Link',
                           'Label', 'Label A', 'Label B', 'Label C', 'Note', '摘要']
        df = pd.read_excel(excel_path, usecols=columns_to_read, nrows=None)

        # 手动定义列的顺序，将新增列插入到中间
        columns_order = ['Source Title', 'Article Title', 'Label', 'Label A', 'Label B', 'Label C',
                         '摘要', 'Abstract', 'Publication Year', 'DOI', 'DOI Link', 'Note']
        df = df[columns_order]  # 重新排列顺序
        return df


# 从json文件中加载prompt
def load_prompt_from_json(json_path, prompt_version):
    with open(json_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
        prompt = data[prompt_version]
    return prompt


# 使用llm分析标题和摘要，判断是否属于特定的研究主题
def analyze_with_llm(text, prompt, llm, RAG, temperature):
    max_retries = 8  # 最大重试次数
    retry_delay = 1  # 初始重试延迟（秒）
    corpus = load_corpus_from_excel(Config.INPUT_EXCEL_PATH)
    if RAG == 'keyword':
        context = keyword_query_system(corpus, text)
    elif RAG == 'semantic':
        context = simple_query_system(corpus, text, vector_db_path=Config.VECTOR_DB_CSV_PATH)
    else:
        context = ''
    system_prompt = "You are an expert in paper screening and recommendation in the field of environmental science."
    user_prompt = f"{prompt}\n\n{text}\n\n{context}"
    full_prompt = f"{system_prompt}{user_prompt}"
    for attempt in range(max_retries):
        try:
            if llm == 'kimi':
                client = OpenAI(
                    api_key="your-api-key",
                    base_url="https://api.moonshot.cn/v1",
                )
                response = client.chat.completions.create(
                    model="moonshot-v1-8k",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content, full_prompt
            elif llm == 'deepseek':
                client = OpenAI(api_key="your-api-key", base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content":  user_prompt},
                    ],
                    stream=False,
                    temperature=temperature
                )
                return response.choices[0].message.content, full_prompt
            else:
                client = OpenAI(
                    api_key="your-api-key",
                    base_url="https://www.dmxapi.cn/v1",
                )

                response = client.chat.completions.create(
                    model=llm,
                    messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content, full_prompt
        except Exception as e:
            print(f"Error analyze abstract (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(retry_delay)
            retry_delay *= 2  # 每次重试延迟翻倍
    return None


# 结构化输出llm的回答
def format_llm_output(llm_output, full_prompt):
    # 用正则表达式提取llm回答的指定部分
    response = re.search(r'Overall Response.*?(Yes|No)', llm_output)
    response_a = re.search(r"Response A.*?:\s*(Yes|No)", llm_output, re.DOTALL)
    response_b = re.search(r"Response B.*?:\s*(Yes|No)", llm_output, re.DOTALL)
    response_c = re.search(r"Response C.*?:\s*(Yes|No)", llm_output, re.DOTALL)

    # Extract the matched text or set as empty string if not found
    response = response.group(1) if response else ''
    response_a = response_a.group(1) if response_a else ''
    response_b = response_b.group(1) if response_b else ''
    response_c = response_c.group(1) if response_c else ''

    # Encode response as 1 (Yes) or 0 (No)

    response_a_encoded = 1 if response_a in ['Yes', '是'] else 0 if response_a in ['No', '否'] else '-'
    response_b_encoded = 1 if response_b in ['Yes', '是'] else 0 if response_b in ['No', '否'] else '-'
    response_c_encoded = 1 if response_c in ['Yes', '是'] else 0 if response_c in ['No', '否'] else '-'

    if response_a_encoded == '-':
        overall_response_encoded = 1 if response in ['Yes', '是'] else 0 if response in ['No', '否'] else '-'
    elif response_a_encoded in {0, 1}:
        overall_response_encoded = 1 if (response_a_encoded == 1 and response_b_encoded == 1 and response_c_encoded == 1) else 0
    else:
        overall_response_encoded = 'error'

    return {
        'Overall Response': overall_response_encoded,
        'Response A': response_a_encoded,
        'Response B': response_b_encoded,
        'Response C': response_c_encoded,
        'Explanation from llm': llm_output,
        'Full Prompt': full_prompt,
    }


def format_llm_output_pro(llm_output, full_prompt):
    # 用正则表达式提取llm回答的指定部分
    response = re.search(r'Overall Response.*?(Yes|No)', llm_output)

    response_a = re.search(r"Response A.*?:\s*(Yes|No|是|否)", llm_output, re.DOTALL)
    response_b1 = re.search(r"Response B1.*?:\s*(Yes|No|是|否)", llm_output, re.DOTALL)
    response_b2 = re.search(r"Response B2.*?:\s*(Yes|No|是|否)", llm_output, re.DOTALL)
    response_c = re.search(r"Response C.*?:\s*(Yes|No|是|否)", llm_output, re.DOTALL)

    # Extract the matched text or set as empty string if not found
    response = response.group(1) if response else ''
    response_a = response_a.group(1) if response_a else ''
    response_b1 = response_b1.group(1) if response_b1 else ''
    response_b2 = response_b2.group(1) if response_b2 else ''
    response_c = response_c.group(1) if response_c else ''

    # Encode response as 1 (Yes) or 0 (No)

    response_a_encoded = 1 if response_a in ['Yes', '是'] else 0 if response_a in ['No', '否'] else '-'
    response_b1_encoded = 1 if response_b1 in ['Yes', '是'] else 0 if response_b1 in ['No', '否'] else '-'
    response_b2_encoded = 1 if response_b2 in ['Yes', '是'] else 0 if response_b2 in ['No', '否'] else '-'
    response_c_encoded = 1 if response_c in ['Yes', '是'] else 0 if response_c in ['No', '否'] else '-'

    if response_a_encoded == '-':
        overall_response_encoded = 1 if response in ['Yes', '是'] else 0 if response in ['No', '否'] else '-'
    elif response_a_encoded in {0, 1}:
        overall_response_encoded = 1 if (response_a_encoded == 1 and response_b1_encoded == 1
                                         and response_b2_encoded == 1 and response_c_encoded == 1) else 0
    else:
        overall_response_encoded = 'error'

    return {
        'Overall Response': overall_response_encoded,
        'Response A': response_a_encoded,
        'Response B1': response_b1_encoded,
        'Response B2': response_b2_encoded,
        'Response C': response_c_encoded,
        'Explanation from llm': llm_output,
        'Full Prompt': full_prompt,
    }


# 根据需求，选择前20篇或全部论文
def get_end_index(df, paper_number):
    if paper_number == 1:
        return df.shape[0]  # 测试所有论文（索引从0到len(df)-1）
    else:
        return min(paper_number, df.shape[0] - 1)  # 测试前20篇（索引0到19），防止数据不足


def calculate_confusion_matrix(tech, input_path,llm):
    df = pd.read_excel(input_path)
    n = df['Overall Response'].notna().sum()
    # 如果指定了 n，则只取前 n 行数据
    df_n = df.head(n) if n is not None else df
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
    if ':' in llm:
        llm = llm.replace(':', '')
    out_path = f"E:\\desktop\\{tech}\\{llm}\\5_final_paper{add}_{llm}_{prompt_version}_{time}.xlsx"
    out_path = f"E:\\desktop\\figure\\并行数-时间\\{tech}\\5_final_paper{add}_{llm}_{prompt_version}_{time}.xlsx"
    df.to_excel(out_path, index=False)
    print(out_path)
    # os.startfile(out_path)


def calculate_confusion_matrix_pro(tech, input_path,llm):
    df = pd.read_excel(input_path)
    n = df['Overall Response'].notna().sum()
    # 如果指定了 n，则只取前 n 行数据
    df_n = df.head(n) if n is not None else df
    # 提取实际标签和预测标签
    y_true = df_n['Label']  # 总体实际标签
    y_pred = df_n['Overall Response']  # 总体预测标签
    y_true_a = df_n['Label A']
    y_pred_a = df_n['Response A']
    y_true_b1 = df_n['Label B1']
    y_pred_b1 = df_n['Response B1']
    y_true_b2 = df_n['Label B2']
    y_pred_b2 = df_n['Response B2']
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

    # 计算 B1 部分的混淆矩阵和指标
    if y_true_b1.dtype in ['int64', 'float64'] and y_pred_b1.dtype in ['int64', 'float64']:

        cm_b1 = confusion_matrix(y_true_b1, y_pred_b1, labels=[1, 0])
        precision_b1 = precision_score(y_true_b1, y_pred_b1)
        recall_b1 = recall_score(y_true_b1, y_pred_b1)
        f1_b1 = f1_score(y_true_b1, y_pred_b1)

        result.append(f"B1部分混淆矩阵: {cm_b1}")
        result.append(f"精确度 (Precision): {precision_b1:.3f}")
        result.append(f"召回率 (Recall): {recall_b1:.3f}")
        result.append(f"F1分数: {f1_b1:.3f}\n\n")
    else:
        result.append("B1部分 指标无法计算，因为数据中没有有效的标签或预测值。\n\n")

    # 计算 B2 部分的混淆矩阵和指标
    if y_true_b2.dtype in ['int64', 'float64'] and y_pred_b2.dtype in ['int64', 'float64']:
        cm_b2 = confusion_matrix(y_true_b2, y_pred_b2, labels=[1, 0])
        precision_b2 = precision_score(y_true_b2, y_pred_b2)
        recall_b2 = recall_score(y_true_b2, y_pred_b2)
        f1_b2 = f1_score(y_true_b2, y_pred_b2)

        result.append(f"B2部分混淆矩阵: {cm_b2}")
        result.append(f"精确度 (Precision): {precision_b2:.3f}")
        result.append(f"召回率 (Recall): {recall_b2:.3f}")
        result.append(f"F1分数: {f1_b2:.3f}\n\n")
    else:
        result.append("B2部分 指标无法计算，因为数据中没有有效的标签或预测值。\n\n")

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
    if ':' in llm:
        llm = llm.replace(':', '')
    out_path = f"E:\\desktop\\{tech}\\{llm}\\5_final_paper{add}_{llm}_{prompt_version}_{time}.xlsx"
    out_path = f"E:\\desktop\\figure\\并行数-时间\\{tech}\\5_final_paper{add}_{llm}_{prompt_version}_{time}.xlsx"
    df.to_excel(out_path, index=False)
    print(out_path)
    # os.startfile(out_path)


def main(add, tech, llm, prompt_version, paper_number, max_workers, open_file, RAG, temperature):
    start_time = time.time()
    # 加载提示文本和论文信息
    prompt_path = f"E:\\Desktop\\{tech}\\prompt_{tech}.json"
    prompt = load_prompt_from_json(prompt_path, prompt_version)
    paper_path = f"E:\\Desktop\\{tech}\\3_labelled_paper{add}.xlsx"
    df = extract_from_excel(tech, paper_path)

    # 分析前N篇或所有论文
    start_index = 0  # 起始论文索引（从0开始计数，第10篇是索引9）
    end_index = get_end_index(df, paper_number)  # 设置处理的论文数量，或all为所有论文

    # 提取需要分析的文本(测试）
    texts_to_analyze = [
        f"Title: {row['Article Title']}\nAbstract：{row['Abstract']}"
        for index, row in df.iloc[start_index:end_index].iterrows()
    ]

    # 初始化结果列表，确保长度与 texts_to_analyze 一致
    results = [None] * len(texts_to_analyze)

    # 使用线程池并行数-时间分析
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        #  提交任务并记录每个任务的索引
        futures = {executor.submit(analyze_with_llm, text, prompt, llm, RAG, temperature): i for i, text in enumerate(texts_to_analyze)}
        # 按照任务完成的顺序处理结果
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]  # 获取任务的索引
            try:
                llm_output, full_prompt = future.result()
                if llm_output is not None:
                    if tech == 'EO':
                        format_output = format_llm_output_pro(llm_output, full_prompt)
                    else:
                        format_output = format_llm_output(llm_output, full_prompt)
                    results[index] = format_output  # 将结果存储到对应的位置
                    print(f"Paper {index + 1} analysis completed.")  # 打印当前论文的分析进度
                else:
                    print(f"Paper {index + 1} analysis failed after all retries.")
                    results[index] = {'Overall Response': '-',
                                      'Response A': '-',
                                      'Response B': '-',
                                      'Response C': '-',
                                      'Explanation from llm': 'Error',
                                      'Full Prompt': full_prompt}
            except Exception as e:
                print(f"Paper {index + 1} analysis failed with error: {e}")
                results.append({'Overall Response': '-', 'Response A': '-', 'Response B': '-', 'Response C': '-',
                                'Explanation from llm': 'Error','Full Prompt': full_prompt})

    # 将结果转换为 DataFrame
    df_results = pd.DataFrame(results)
    number_of_paper = df_results.shape[0]

    # 获取 df_results 的列名
    result_columns = df_results.columns.tolist()
    # 找到 df 中指定列的索引位置
    insert_position = df.columns.get_loc('Label C') + 1  # 在 'sa' 列后面插入
    # 逐列插入 df_results 的列到 df 中
    for col in result_columns:
        df.insert(insert_position, col, None)
        df.loc[start_index:start_index + len(df_results) - 1, col] = df_results[col].values  # 插入到df的指定行号位置
        insert_position += 1

    end_time = time.time()
    total_time = end_time - start_time

    time_result = (
        f"{llm}处理的论文数量：{number_of_paper} 篇\n"
        f"最大并行数-时间任务数：{max_workers} 个\n"
        f"程序总耗时：{total_time:.2f} 秒\n"
        f"每篇论文耗时：{total_time / number_of_paper:.2f} 秒\n"
        f"温度：{temperature}"
    )

    df.loc[:, 'Metrics'] = None  # 新增一列
    df.loc[df.index[0], 'Metrics'] = time_result  # 只在第一行添加结果

    # 保存结果到新的Excel文件并打开
    current_time = datetime.now()
    t = current_time.strftime("%y%m%d-%H-%M")
    if ':' in llm:
        llm = llm.replace(':', '')
    output_path = f"E:\\Desktop\\{tech}\\{llm}\\4_analyzed_paper{add}_{llm}_{prompt_version}_{t}.xlsx"
    output_path = f"E:\\Desktop\\figure\\并行数-时间\\{tech}\\4_analyzed_paper{add}_{llm}_{prompt_version}_{t}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Analysis results saved to {output_path}")
    print(time_result)
    return output_path


if __name__ == "__main__":
    add = '_分层抽样'
    techs = ['RO']
    llms = ['kimi', 'gpt-4.1', 'grok-3-mini', 'claude-3-5-haiku-20241022', 'deepseek']

    prompt_version = 'promptv1'  # 设置提示词的版本，v1是3T模板，v2是简单提示
    open_file = False  # 是否打开analyzed paper文件
    RAG = 'keyword'  # keyword是关键词检索，semantic是语义检索
    paper_number = 1  # 设置处理的论文数量，或1为所有论文
    numbers = [5]  # 设置并行数-时间的线程数
    temperature = 0
    for tech in techs:
        for llm in llms:
            for max_workers in numbers:
                for i in range(3):
                    input_path = main(add, tech, llm, prompt_version, paper_number, max_workers, open_file, RAG, temperature)
                    if tech == 'EO':
                        calculate_confusion_matrix_pro(tech, input_path, llm)
                    else:
                        calculate_confusion_matrix(tech, input_path, llm)
                    os.remove(input_path)
                    print(f'第{i+1}次已完成----------------------------------------------------------------------------')
                    time.sleep(40)
