import os
import random
import argparse
import json



if __name__=="__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--key', type=str)
    # args = parser.parse_args()
    # image_name = args.key


    # with open("./dataset/eval_idx.json", "r") as st_json:
    #     group_dicts = json.load(st_json)
    with open("./dataset/dict_real.json", "r") as st_json:
        group_dicts = json.load(st_json)
    # group_dictionary = group_dicts
    all_keys  = group_dicts

    # all_keys_gt = ['112', '598', '770', '805', '1936', '2117', '2293', '2802', '2855', '3655', '4452', '6041', '7469', '9803']
    # all_keys_llm  = ['598', '820', '1805', '1937', '2196', '3957', '4835', '6230', '7727', '8210', '8942', '9018', '9039', '3957']


    all_keys = list(all_keys)
    # all_keys = all_keys[:len(all_keys)//2]

    for _ in range(20):
        for key in all_keys:
            if os.path.exists(f"./outputs/llm_total/{key}"):
                pass
            else:
                seed_num = random.randint(0, 10000000)
                os.system(f"python run_exp.py --key {key} --seed {seed_num} --h_num 2")