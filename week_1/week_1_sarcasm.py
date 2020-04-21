import json

with open('sarcasm.json') as f:
    data_dict = json.load(f)

article_link = []
headline = []
is_sarcastic = []

for entry in data_dict:
    article_link.append(entry['article_link'])
    headline.append(entry['headline'])
    is_sarcastic.append(entry['is_sarcastic'])


if __name__ == '__main__':

        print(is_sarcastic)
