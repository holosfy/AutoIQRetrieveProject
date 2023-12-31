import json

from tqdm import tqdm

# res = llm_utils.llm_chat("chinese-llama-2-7b-hf", prompt % '我如何知道我的汽车的安全带是正常工作的?')
import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8888/v1"

model = "Qwen-14B-Chat-Int8"

def llm_chat(model_name, prompt):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content


def get_chat_result(prompt):
    res = llm_chat(model, prompt)
    #
    print(res)
    if res in res_set:
        res = llm_chat(model, prompt)
    return res

res_set = set()

def get_submit_data():
    res = []
    prompt_template = """
       现在你是一个吉利旗下的领克品牌的汽车专家，来负责解答下面与汽车驾驶操作相关的问题。
       你可以参考吉利汽车或其他汽车的操作手册。
       回答要简要概括，最好在100个字以内。遇到不知道的问题，诚实的回答不知道。下面是问题：
       %s
       """

    with open('../data/测试问题.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    for qa in tqdm(data):
        question = qa['question']
        print('当前的问题是question : '+question)
        prompt = prompt_template % question
        print("answer_1")
        qa['answer_1'] = get_chat_result(prompt)
        print("answer_2")
        qa['answer_2'] = get_chat_result(prompt)
        print("answer_3")
        qa['answer_3'] = get_chat_result(prompt)
        res.append(qa)

    filename = model + ".json"
    with open(filename, 'w', encoding='utf-8') as file_obj:
        json.dump(res, file_obj, ensure_ascii=False)




if __name__ == '__main__':
    get_submit_data()