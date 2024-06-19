import pandas as pd

val = {}

with open(r'C:\code\g9-rs\data\noun_syn.txt', 'r', encoding="utf-8") as f:
    buffer = []
    for line in f.readlines():
        buffer.append(line.strip())
    val['eng_name'] = buffer

with open(r'C:\code\g9-rs\data\name_zh.txt', 'r', encoding="utf-8") as f:
    buffer = []
    for line in f.readlines():
        buffer.append(line.strip())
    val['zh_name'] = buffer

with open(r'C:\code\g9-rs\data\prices.txt', 'r', encoding="utf-8") as f:
    buffer = []
    for line in f.readlines():
        buffer.append(line.strip())
    val['ori_price'] = buffer

with open(r'C:\code\g9-rs\data\prices_recycle.txt', 'r', encoding="utf-8") as f:
    buffer = []
    for line in f.readlines():
        buffer.append(line.strip())
    val['recycle_price'] = buffer

df = pd.DataFrame(val)
df['recycle_price'] = df['recycle_price'].astype(float)
df['ori_price'] = df['ori_price'].astype(float)
df = df.drop_duplicates(subset='zh_name')
df = df.drop(df[df['ori_price'].eq(0.0)].index)
df = df.drop(df[df['recycle_price'].eq(0.0)].index)
df.to_csv(r'C:\code\g9-rs\data\data.csv', index=False, encoding="utf-8-sig")