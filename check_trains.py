#!/usr/bin/python
from __future__ import division

import os, sys
import argparse
import pandas as pd
import numpy as np
import time, glob
import matplotlib as mpl
mpl.rcParams['backend']='TkAgg'
import matplotlib.pyplot as plt
#import Tkinter as tk
#import itertools
#import subprocess

def find_record_files(workdir, filename):
    """Walk throgh the workdir and its sub-folders to find the file:filename,
       return the path list
    """
    filelist=[]
    walker = os.walk(workdir)
    for (root, dirs, files) in walker:
        if root==workdir:
            jobs_dir=dirs
            print "{} job folders".format(len(jobs_dir))
        if filename in files:
            filelist.append(os.path.join(root, filename))
        if root in jobs_dir and filename not in files:
            print "{} file not found in {} folder".format(root, filename)
    print "{} {} are found".format(len(filelist), filename)
    return filelist


class RecordPlots(object):
    def __init__(self, workdir):
        self.workdir=workdir
        self.fname_infer_record="inference_records.csv"
        self.fname_train_record="training_records.txt"
        self.fname_exp_table="exp_cfg_tables.csv"
        self.get_step=1000

    def infer_results_to_exps_table(self):
        exps_fn=os.path.join(self.workdir, self.fname_exp_table)
        df_exps=pd.read_csv(exps_fn, skipinitialspace=True, sep="\s+")
        #infer_files=find_record_files(self.workdir, self.fname_infer_record)

        df_list=[]
        for job_dir in df_exps.WORKING_DIR:
            _, job_id=os.path.basename(job_dir).split("_")
            job_id=int(job_id)
            job_dir=os.path.join(self.workdir, job_dir)
            infer_file=os.path.join(job_dir, self.fname_infer_record)
            df_temp=pd.read_csv(infer_file, skipinitialspace=True, sep="\s+", index_col=None, header=0)
            df_temp["job_id"]=job_id
            df_list.append(df_temp)
        data_all=pd.concat(df_list,ignore_index=True)
        data_at_step=data_all[data_all['step']==self.get_step]
        data_at_step.reset_index(inplace=True)
        df_exps["train_rms"]=data_at_step["training_RMS(all)"]
        df_exps["valid_rms"]=data_at_step["validation_RMS(all)"]
        df_exps["job_id"]=data_at_step["job_id"]
        return df_exps


    def record_plot_single(self, jobdir, show_plot=False):
        """Plot the results from inference_record.csv and training_record.txt
        """
        tvrms_fn=os.path.join(jobdir, self.fname_infer_record)
        cost_fn=os.path.join(jobdir, self.fname_train_record)
        df_tvrms=pd.read_csv(tvrms_fn, skipinitialspace=True, sep="\s+", skiprows=[1], header=0)
        df_cost=pd.read_csv(cost_fn, skipinitialspace=True, sep=",", skiprows=[0,1],header=None, skipfooter=1)
        dict_cost={}
        for col in df_cost:
            key, val = np.transpose(map(lambda x: x.split("="), df_cost[col].tolist()))
            try:
                val = val.astype(np.float32)
            except:
                pass
            dict_cost[key[0]]=val
        df_cost=pd.DataFrame(dict_cost)

        if show_plot: 
            fig, ax = plt.subplots()
            df_cost.plot(x='step', y='cost', secondary_y=True, style='-.', ax=ax)
            df_tvrms.plot(x='step', y=["training_RMS(all)", "validation_RMS(all)"], 
                          grid=True, title=os.path.basename(jobdir), style='-o', ax=ax)
        

    def inference_plots(self):
        data=self.infer_results_to_exps_table()
        data=data[data.train_rms<2.0]
        #data.boxplot(column=["train_rms", "valid_rms"], by="CHANNELS")
        #data.boxplot(column=["train_rms", "valid_rms"], by="_LAYER_OP")
        #data.boxplot(column=["train_rms"], by=["_LAYER_OP", "CHANNELS"])
        #data.boxplot(column=["train_rms", "valid_rms"], by="THRESH_ADJUST")
        #data.boxplot(column=["train_rms", "valid_rms"], by="MODEL_VERSION")
        data.boxplot(column=["valid_rms"], by=["MODEL_VERSION","THRESH_ADJUST"])
        data.plot(x="job_id", y=["train_rms", "valid_rms"], grid=True, kind='bar')
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default="./")
    FLAGS, _ = parser.parse_known_args()

    find_record_files(FLAGS.work_dir, "inference_records.csv")
    myplot=RecordPlots(FLAGS.work_dir)
    #myplot.record_plot_single(os.path.join(FLAGS.work_dir, "job_0001"))
    myplot.inference_plots()
    #myplot.inference_plot_all()


if __name__=="__main__":
    main()
