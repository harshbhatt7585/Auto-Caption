import json
import pandas as pd
import argparse

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--csv_path', default='metadata.csv', help='path of csv file')
    args.add_argument('--json_path', default='metadata.jsonl', help='json output path')
    args = args.parse_args()

    df = pd.read_csv(args.csv_path)
    arr = []
    for i in range(len(df)):
        file_name = df['file_name'][i]
        text = df['text'][i]
        arr.append({'file_name': file_name, 'text': text})

    with open(args.json_path, 'w') as f:
        for item in arr:
            f.write(json.dumps(item) + '\n')
