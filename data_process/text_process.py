




def get_text_part():
    pass



if __name__ == '__main__':
    with open("../data/test_pdf2text.text", "r", encoding="utf-8") as f:
        text = f.readlines()

    text = map(lambda x: x.replace('\n', ''), text)
    #text = filter(lambda x: x!='', text)

    tmp_lst = []
    tmp = ''

    for t in text:
        if "......................" in t:
            continue
        tmp += t
        if tmp.endswith('。'):
            tmp_lst.append(tmp)
            tmp = ''

    part_lst = []
    tmp_part = ''
    part_len = 1000
    for line in tmp_lst:
        tmp_part += line
        if len(tmp_part)>1000:
            part_lst.append(tmp_part)
            tmp_part = ''
            continue

    for p in part_lst:
        print(str(len(p))+":" + p)

    with open("../data/test_part.text", "w", encoding="utf-8") as f:
        f.write('\n'.join(part_lst))