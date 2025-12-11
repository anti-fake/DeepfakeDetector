#!/bin/bash

# python3 validate_for_robustness.py --gen_model cyclegan
# python3 validate_for_robustness.py --gen_model stylegan
# python3 validate_for_robustness.py --gen_model stylegan2
python3 validate_for_robustness.py --dataset UFD --gen_model progan

# python3 validate_for_robustness.py --dataset GenImage --gen_model adm
# python3 validate_for_robustness.py --dataset GenImage --gen_model biggan
# python3 validate_for_robustness.py --dataset GenImage --gen_model glide
# python3 validate_for_robustness.py --dataset GenImage --gen_model midjourney
# python3 validate_for_robustness.py --dataset GenImage --gen_model sdv4
# python3 validate_for_robustness.py --dataset GenImage --gen_model vqdm
# python3 validate_for_robustness.py --dataset GenImage --gen_model wukong
