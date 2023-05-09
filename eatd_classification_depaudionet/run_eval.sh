#! /usr/bin/env bash


script=main_disent_fscore.py
python3 $script test --validate --prediction_metric=1 --threshold=total
