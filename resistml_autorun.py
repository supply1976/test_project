from __future__ import division

import os, sys
import itertools
import argparse
import subprocess
import pandas as pd
import numpy as np
import time, glob
#import Tkinter as tk


class CFG(object):
    def __init__(self):
        self.configs={"_LAYER_OP": ["XY:N"],
                      "CHANNELS": [4],
                      "MODEL_VERSION": ["0.1"],
                      "THRESH_ADJUST": [False],
                      "MAX_ITER": [1000],
                      "BATCH_SIZE": [256],
                      "SAVE_STEP": [100],
                      "SHOW_STEP": [20],
                      "GPU_INDICES": ["0,1"],
                      "INITIAL_LEARNING_RATE": [0.001],
                      "NUM_EPOCHS_PER_DECAY": [1000],
                      "LEARNING_RATE_DECAY_FACTOR": ["0.8"],
                      "REGULARIZATION_WEIGHT": [0.0001],
                      "_IS_XAVIER_INITIALIZER": [False],
                      }

    def create_split_table(self):
        configs = self.configs
        splits = list(itertools.product(*configs.values()))
        self.cfg_table = pd.DataFrame(splits, columns=configs.keys())
        zeropad = len(str(len(splits)))+1
        self.job_id = map(lambda x: x.zfill(zeropad), map(str, self.cfg_table.index+1))
        self.cfg_table["WORKING_DIR"]=map(lambda x: "./job_"+x, self.job_id)
        #print cfg_table
        self.cfg_table.to_csv("exp_cfg_tables.csv", sep="\t", index=False)
        #return cfg_table


    def gen_cfgs(self):
        job_id = self.job_id
        cfgTable = self.cfg_table
        
        for i, id in enumerate(job_id):
            fn = "train_"+id+".cfg"
            exp = cfgTable.iloc[i]
            exp.to_csv(fn, sep=" ")
        print "{} training config files are generated".format(len(cfgTable))
        time.sleep(1)


    def run_cfgs(self):
        cfg_files = glob.glob("./*.cfg")
        initT = time.time()
        print "running on {} configs in single host".format(len(cfg_files))
        print "start time={}".format(time.ctime())
        for count, cfg in enumerate(cfg_files):
            print "#{}, training cfg: {}".format(count, cfg)
            proc = subprocess.Popen(["resistmltrain_gpu", "--config", cfg])
            proc.wait()
            print "#"*50
        print "All Done, elaspe time={}".format(time.time()-initT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default="NONE")
    parser.add_argument('--valid_data', type=str, default="NONE")
    FLAGS, _ = parser.parse_known_args()

    cfg=CFG()

    cfg.configs["CHANNELS"]=[4,8,12,16]
    cfg.configs["_LAYER_OP"]=["XY:N", "XY:XY:N", "XY:XY:XY:N"]
    cfg.configs["MODEL_VERSION"]=["0.1", "0.2", "0.2.1", "0.3", "0.3.1"]
    cfg.configs["THRESH_ADJUST"]=[False, True]
    #cfg.configs["BATCH_SIZE"]=[64,128,256,512]
    cfg.configs["CDML_DATA"]=[FLAGS.train_data]
    cfg.configs["CDML_VALIDATION_DATA"]=[FLAGS.valid_data]

    cfg.create_split_table()
    cfg.gen_cfgs()
    cfg.run_cfgs()


if __name__=="__main__":
    main()
