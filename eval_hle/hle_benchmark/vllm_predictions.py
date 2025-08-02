# MIT License
# 
# Copyright (c) 2020 centerforaisafety
# Copyright (c) 2025 Kazuma Matsumoto
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from dataclasses import dataclass
import os
import json
import asyncio
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from hle_benchmark._configs import Config
import datetime # datetimeモジュールをインポート


SYSTEM_EXACT_ANSWER = "Your response should be in the following format:\nExplanation: {your explanation for your final answer}\nExact Answer: {your succinct, final answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"

SYSTEM_MC = "Your response should be in the following format:\nExplanation: {your explanation for your answer choice}\nAnswer: {your chosen answer}\nConfidence: {your confidence score between 0% and 100% for your answer}"


def format_message(args, question):
    answer_type = question['answer_type']
    system_prompt = SYSTEM_EXACT_ANSWER if answer_type == 'exact_match' else SYSTEM_MC
    question_text = question['question']

    text_content = dict(type="text", text=question_text)
    if question['image']: # "" if not multi-modal
        image_content = dict(type="image_url", image_url=dict(url=question['image']))
        content = [text_content, image_content]
    else:
        content = [text_content]

    system_role = "user" if "o1" in args.model else "system" # o1 no sys prompt
    messages = [
        {"role": system_role, "content": system_prompt}, 
        {"role": "user", "content": content}
    ]
    return messages


async def attempt_question(args, question):
    messages = format_message(args, question)
    try:
        response = await client.chat.completions.create(
            model=args.model,
            # temperature=args.temperature if "o1" not in args.model else None,
            max_completion_tokens=args.max_completion_tokens,
            messages=messages,
            stream=False,
        ) 
        content = response.choices[0].message.content
        if response.choices[0].finish_reason == "length":
            return None
        tokens = json.loads(response.usage.model_dump_json())
        if args.reasoning:
            tokens["reasoning_content"] = response.choices[0].message.reasoning_content
    except Exception as e:
        print("Error:", e)
        return None
    
    if content is None: # failed
        return None
    
    return question["id"], content, tokens


# 回答が10問ごとに保存
async def attempt_all(args, questions, predictions, output_filepath):
    async def bound_func(question):
        async with semaphore:
            content = await attempt_question(args, question)
            return content
            
    semaphore = asyncio.Semaphore(args.num_workers)
    async with semaphore:
        tasks = [bound_func(q) for q in questions]
        results = []
        completed_count = 0
        
        # Process results as they complete and save every 10 items
        for result in tqdm_asyncio.as_completed(tasks):
            result_data = await result
            results.append(result_data)
            
            if result_data is not None:
                unique_id, response, usage = result_data
                predictions[unique_id] = {
                    "model": args.model,
                    "response": response,
                    "usage": usage
                }
                completed_count += 1
                print(f"Processed question ID: {unique_id}. Total completed: {completed_count}")
                
                # Save every 10 completed items
                if completed_count % 10 == 0:
                    with open(output_filepath, "w") as f:
                        json.dump(predictions, f, indent=4)
                    print(f"Saved progress: {completed_count} items completed")
    
    return results

def main(args: Config):
    global client
    client = AsyncOpenAI(
        base_url=args.base_url,
        timeout=86400,
        max_retries=3,
        api_key="fakeapikey",
    )

    assert args.num_workers > 1, "num_workers must be 2 or greater"
    dataset = load_dataset(args.dataset, split="test")
    # 画像フィルタ
    dataset = dataset.filter(lambda item: item['image'] == "")

    # カテゴリをフィルタ
    if args.category_filter:
        category_filters_lower = [c.lower() for c in args.category_filter]
        dataset = dataset.filter(lambda item: item["category"].lower() in category_filters_lower)

    # dataset = dataset.select(range(3))
    dataset = dataset.to_dict()

    # convert to list of json for async parallelism
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    
    # 同じ問題は解かないためのフィルタリング機能
    # 設定したpredictionファイルについて被りがないように問題をフィルタ
    if args.duplicate_filter:
        # load only questions without responses
        if os.path.exists(args.duplicate_file):
            with open(args.duplicate_file, "r") as f:
                predictions = json.load(f)
            questions = [q for q in questions if q["id"] not in predictions]
        else:
            print("args.duplicate_fileの設定が間違っている")

    # If max_samples is set, limit the number of questions
    if args.max_samples:
        questions = questions[:args.max_samples]

    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ファイル名にモデル名とタイムスタンプを組み込む
    # 例: predictions/hle_qwen3-32b_20250731_023309.json
    # 複数の実行で衝突が起きないようにするため
    output_filepath = f"predictions/hle_{os.path.basename(args.model)}_{timestamp_str}.json"

    # もしディレクトリがない場合は作成
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    # load only questions without responses
    if os.path.exists(output_filepath):
        with open(output_filepath, "r") as f:
            predictions = json.load(f)
        questions = [q for q in questions if q["id"] not in predictions]
    else:
        predictions = {}

    results = asyncio.run(attempt_all(args, questions, predictions, output_filepath))

    # Final save to ensure all data is persisted
    with open(output_filepath, "w") as f:
        json.dump(predictions, f, indent=4)
    print(f"Final save completed: {len([r for r in results if r is not None])} total items processed")
