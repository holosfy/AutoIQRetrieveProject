

def check_prompt_content_len(prompt_content):
    max_len = 7500
    if len(prompt_content)<= max_len:
        pass
    else:
        d = int((len(prompt_content)-max_len) /2 )
        prompt_content = prompt_content[d:]
        prompt_content = prompt_content[:-d]

    return prompt_content


if __name__ == '__main__':
    print(check_prompt_content_len("1234567"))