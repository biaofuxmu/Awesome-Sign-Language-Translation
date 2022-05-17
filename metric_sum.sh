import os
import argparse
import json
import sacrebleu
from statistics import mean
from natsort import natsorted
import tensorflow as tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent-folder', default="./", type=str,
                        help='Parent Folder')
    args = parser.parse_args()

    parent_folder = args.parent_folder


    result_json = []
    for currentpath, folders, files in tf.io.gfile.walk(parent_folder):
        files = natsorted(files)
        for file_name in files:
            if "instances.log" == file_name:
                file_name = os.path.join(currentpath, file_name)
                with tf.io.gfile.GFile(file_name, "r") as jfile:
                    for line in jfile:
                        result_json.append(json.loads(line))
    
    results = {}
    results["Quality"] = {}
    results["Latency"] = {}

    # caculate BLEU
    
    sacrebleu_tokenizer = "13a"
    refs = list(map(lambda x: x["reference"], result_json))
    hypos = list(map(lambda x: x["prediction"].replace("</s>", "").strip(), result_json))
    print("len(hypos)", len(hypos))

    bleu_score = sacrebleu.corpus_bleu(
                    hypos,
                    [refs],
                    tokenize=sacrebleu_tokenizer
                ).score
    results["Quality"]["BLEU"] = bleu_score

    for metric in ["AL", "AP", "DAL"]:
        score = list(map(lambda x: x["metric"]["latency"][metric], result_json))
        results["Latency"][metric] = mean(score)
        score = list(map(lambda x: x["metric"]["latency_ca"][metric], result_json))
        results["Latency"][metric+"_CA"] = mean(score)
    print(results)
    with tf.io.gfile.GFile(os.path.join(parent_folder, "scores"), 'w') as f:
        f.write(json.dumps(results, indent=4))