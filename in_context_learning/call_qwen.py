from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from in_context_learning.prompt_templates import dynamic_prompt

# Model names: "Qwen/Qwen-7B", "Qwen/Qwen-14B"
model_dir = "../Qwen-7B"
model_dir = "../Qwen-7B-Chat"
#model_dir = "../ChatGLM2-6B/chatglm2-6b-32k/"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-7B", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use auto mode, automatically select precision based on the device.
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    trust_remote_code=True
).eval()

import json
with open("test.json", "r", encoding="utf-8") as f:
    l = json.load(f)


template = "你是吉利旗下的领克汽车机器人，现在需要你回答为提出的问题，你可以参考吉利汽车或其他汽车的操作手册，遇到不知道的问题，就诚实地回答不知道。下面是问题："


result = []

for item in l:
    query = template + item["question"]
    query = dynamic_prompt.format(sentence=item["question"])
    # qwen
    inputs = tokenizer(query, return_tensors="pt")
    inputs = inputs.to(model.device)
    pred = model.generate(**inputs)
    answer = tokenizer.decode(pred.cpu()[0][len(inputs["input_ids"][0]):], skip_special_tokens=True).replace("\n", "")

    # chatglm
    # answer, history = model.chat(tokenizer, query, history=[])

    print("query")
    print(query)
    print()
    print("answer")
    print(answer)
    print()


    item["answer_1"] = answer
    result.append(item)

with open("test_v1_qwen7bchat.json", "w", encoding="utf-8") as fo:
    json.dump(result, fo, ensure_ascii=False)
