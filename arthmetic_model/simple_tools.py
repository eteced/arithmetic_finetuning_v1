#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This software may be used and distributed according to the terms of the GNU General Public License version 3.
"""Small Models for Arthmetic
"""
__author__ = "Yingdi Guo"
__license__ = "GPLv3"
__email__ = "eteced@gmail.com"

import sys
import os
import string
import random
import json

class ArthDataGenerator:
    def __init__(self) -> None:
        self.ops = ['+','-','*','/']
        self.max_decimal_len = 6
        self.max_integer_len = 6
        self.max_length = 120
        self.min_length = 30
        self.joined_char = ' '
        self.max_op_occur = 20
    def generate(self) -> list:
        list_ret=[]
        total_len_now = 0
        now_op = 0
        op_occur = 0
        length = random.randint(self.min_length, self.max_length)
        while total_len_now < length:
            if now_op == 0:
                int_part_size = random.randint(0, self.max_integer_len)
                int_part = int(random.random() * (10 ** int_part_size))
                dec_part_size = random.randint(0, self.max_decimal_len)
                dec_div = 10 ** dec_part_size
                dec_part = int(random.random() * dec_div) / dec_div
                num_final = int_part + dec_part
                if (num_final < 1e-6): # prevent div zero
                    num_final = 0.1
                list_ret.append(str(num_final))
                total_len_now += len(list_ret[-1]) + 1
                if total_len_now + self.max_decimal_len + self.max_integer_len + 1 + 2 * len(self.joined_char) > length: # 1 -> the op
                    break
                if op_occur >= self.max_op_occur:
                    break
                now_op = 1
            else:
                op_next = random.randint(0, len(self.ops) - 1)
                list_ret.append(self.ops[op_next])
                total_len_now += len(list_ret[-1]) + 1
                op_occur += 1
                now_op = 0
        return list_ret

def rev_polish_notation(xx: list) -> list:
    dict_op_level={'+': 0, '-': 0, '*': 1, '/': 1}
    op_od=[]
    final_res=[]
    for x in xx:
        if x not in dict_op_level:
            final_res.append(x)
        else:
            if len(op_od) == 0:
                op_od.append(x)
            elif (dict_op_level[op_od[-1]] < dict_op_level[x]):
                op_od.append(x)
            else:
                while ((len(op_od) > 0) and dict_op_level[op_od[-1]] >= dict_op_level[x]):
                    op_now = op_od.pop()
                    final_res.append(op_now)
                op_od.append(x)
    while (len(op_od) > 0):
        op_now = op_od.pop()
        final_res.append(op_now)
    return final_res

def merge_alpaca_and_generate_data(alpace_file_path, out_data_folder, percentage=0.4, onefile_record_nums=360):
    f1=open(alpace_file_path)
    all_data = json.load(f1)
    f1.close()
    arth_data_num = int(percentage * len(all_data))
    generator = ArthDataGenerator()
    for i in range(arth_data_num):
        arth_express = generator.generate()
        express_final = generator.joined_char.join(arth_express)
        output = str(eval(express_final))
        swift_express = generator.joined_char.join(rev_polish_notation(arth_express))
        dict_a={}
        dict_a['instruction'] = 'Please caculate this.'
        dict_a['input'] = express_final + ' = ?'
        dict_a['output'] = output
        dict_a['swift_express'] = swift_express
        all_data.append(dict_a)
    random.shuffle(all_data)
    dumped_record = 0
    file_nums = 0
    while dumped_record < len(all_data):
        next_cut = dumped_record + onefile_record_nums
        if next_cut > len(all_data):
            one_cut = all_data[dumped_record : ]
        else:
            one_cut = all_data[dumped_record : next_cut]
        f1=open(out_data_folder + '/data_'+str(file_nums)+'.json', "w")
        json.dump(one_cut, f1)
        f1.close()
        file_nums=file_nums+1
        dumped_record = next_cut

if __name__ == "__main__":
    merge_alpaca_and_generate_data('/home/eteced/dl_workspace/stanford_alpaca/alpaca_data.json', '/home/eteced/dl_self_workspace/arithmetic_finetuning_v1/data_split', 0.4, 50)