#!/usr/bin/env python
# -*-coding=utf-8-*-
"""
@Author : liumiangang
@Date : 2023/10/25
"""
import json

with open("../data/test_v1_qwen7bchat.json", "r", encoding="utf-8") as f:
    examples = json.load(f)

    for item in examples:
        d = {"input": item["question"], "output": item["answer_1"]}
        print(f"{d},")