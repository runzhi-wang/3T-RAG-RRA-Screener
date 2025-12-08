import pandas as pd
import json
from openai import OpenAI
import re
import os
import time
from datetime import datetime
import threading
from queue import Queue
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# 全局变量
result_queue = Queue()
lock = threading.Lock()


def analyze_with_llm(prompt, llm, temperature):
    """调用大模型API分析提示文本"""
    max_retries = 8
    retry_delay = 1

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
                        {"role": "system", "content": "hi"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                )
                return response.choices[0].message.content
            elif llm == 'deepseek':
                client = OpenAI(api_key="your-api-key",
                                base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": "hi"},
                        {"role": "user", "content": prompt},
                    ],
                    stream=False,
                    temperature=temperature
                )
                return response.choices[0].message.content
            else:
                client = OpenAI(
                    api_key="your-api-key",
                    base_url="https://www.dmxapi.cn/v1",
                )

                response = client.chat.completions.create(
                    model=llm,
                    messages=[
                            {"role": "system", "content": 'hi'},
                            {"role": "user", "content": prompt},
                    ],
                    temperature=temperature
                )
                return response.choices[0].message.content

        except Exception as e:
            print(f"分析提示时出错(尝试 {attempt + 1}/{max_retries}): {e}")
            time.sleep(retry_delay)
            retry_delay *= 2
    return None


def extract_response_data(llm_output):
    """从大模型响应中提取关键数据"""
    final_response = llm_output

    final_decision_match = re.search(r'Final.*?(Yes|No)', llm_output, re.IGNORECASE)
    final_decision = final_decision_match.group(1) if final_decision_match else None

    if final_decision:
        if final_decision.lower() == 'yes':
            final_decision = '1'
        elif final_decision.lower() == 'no':
            final_decision = '0'

    return {
        'Final Response': final_response,
        'Final Decision': final_decision
    }


def worker(prompts, llm, temperature):
    """工作线程函数"""
    for prompt_data in prompts:
        index, prompt = prompt_data
        if pd.isna(prompt) or not prompt.strip():
            with lock:
                result_queue.put((index, None, None))
            continue

        response = analyze_with_llm(prompt, llm, temperature)

        if response:
            extracted_data = extract_response_data(response)
            with lock:
                result_queue.put((index, extracted_data['Final Response'], extracted_data['Final Decision']))
        else:
            with lock:
                result_queue.put((index, "错误: 获取响应失败", "错误"))


def main():
    llms = ['deepseek','gpt-4.1','kimi','grok-3-mini','claude-3-5-haiku-20241022']
    temperature = 0.0
    num_threads = 100  # 线程数量
    tech = 'AD'
    for llm in llms:
        input_file = f"E:\\Desktop\\{tech}\\RRA\\diff-{tech}.xlsx"
        output_file = f"E:\\Desktop\\{tech}\\RRA\\arbiter_{llm}_{temperature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

        try:
            df = pd.read_excel(input_file)
        except FileNotFoundError:
            print(f"错误: 找不到文件 {input_file}")
            return

        if 'Prompt' not in df.columns:
            print("错误: 输入文件必须包含'Prompt'列")
            return

        df['Final Response'] = None
        df['Final Decision'] = None

        # 准备线程数据
        prompts = [(i, row['Prompt']) for i, row in df.iterrows()]
        chunk_size = len(prompts) // num_threads + 1
        chunks = [prompts[i:i + chunk_size] for i in range(0, len(prompts), chunk_size)]

        # 创建并启动线程
        threads = []
        for chunk in chunks:
            t = threading.Thread(target=worker, args=(chunk, llm, temperature))
            t.start()
            threads.append(t)

        # 显示进度
        processed = 0
        total = len(prompts)
        while processed < total:
            while not result_queue.empty():
                index, final_response, final_decision = result_queue.get()
                df.at[index, 'Final Response'] = final_response
                df.at[index, 'Final Decision'] = final_decision
                processed += 1
                print(f"已处理 {processed}/{total} ({processed / total * 100:.1f}%)")
            time.sleep(0.1)

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 保存结果
        df.to_excel(output_file, index=False)
        print(f"结果已保存到 {output_file}")
        os.startfile(output_file)


if __name__ == "__main__":
    main()
