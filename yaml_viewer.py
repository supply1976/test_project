import os, sys
import yaml
import tkinter as tk
import ttk
# python 2
#import tkFileDialog as filedialog

# python 3
from tkinter import filedialog

_GRID_CFG1 = {'padx':20, 'ipadx': 10, 'ipady':5, 'sticky':'W'}



class TrainingUI(tk.Frame):
    def __init__(self, master=None):
        super(TrainingUI, self).__init__(master)
        self.root_name = "TRAINING"
        # system keys
        self.key_gpu_num ="|SYSTEM|GPU_NUM"
        self.key_gpu_pool="|SYSTEM|GPU_POOL"
        
        # flow keys
        self.key_output_dir = "|FLOW|OUTPUT_DIR"
        self.key_epo_report = "|FLOW|EPOCHS_TO_REPORT"
        self.key_ckpt_export= "|FLOW|EXPORT_INTERMEDIATE_CKPT"
        
        # model export keys
        self.key_mtp_path = "|AUTO_MODEL_EXPORT|MTP_PATH"
        
        # cnn keys
        self.key_cnn_inputs = "|CNN|INPUT_IMAGE_NAMES"
        self.key_cnn_init_algo= "CNN|INITIALIZATION|NETWORK_INITIALIZER"



class SectionUI(tk.Frame):
    def __init__(self, master, section_name, nestdict):
        super(SectionUI, self).__init__(master)
        self.keys=[]
        for i, (sub_k, sub_v) in enumerate(nestdict.items()):
            if type(sub_v) is not dict:
                self.keys.append("|".join([section_name, sub_k]))
                tk.Label(app, text=sub_k).grid(row=i, column=0, **_GRID_CFG1)
                tk.Label(app, text=sub_v).grid(row=i, column=1, **_GRID_CFG1)
                
            else:
                sub2_dict = sub1_dict[sub1_k]
                lb1 = tk.LabelFrame(app, text=sub1_k)
                lb1.grid(row=i, column=0, columnspan=2, **_GRID_CFG1)
                
            
        #self.lbfr1 = tk.LabelFrame(self, text="SECTION 1")
        #self.lbfr1.pack(padx=20, pady=10, fill='both')
        #tk.Label(self.lbfr1, text="foo").grid(
        #    row=0, column=0)
        #self.lbfr2 = tk.LabelFrame(self, text="SECTION 2")
        #tk.Label(self.lbfr2, text="foo2").grid(
        #    row=0, column=0)
        #self.lbfr2.pack()

        

class Tab2UI(tk.Frame):
    def __init__(self, master=None):
        super(Tab2UI, self).__init__(master)
        self.lbfr1 = tk.LabelFrame(self, text="PlaceHolder 1")
        self.lbfr1.pack()
        self.lbfr2 = tk.LabelFrame(self, text="PlaceHolder 2")
        self.lbfr2.pack()
        self.pack()


def main():
    root = tk.Tk()
    root.title("Yaml Viewer")
    root.geometry('1400x1000')
    root.configure(bg='#aaffff')
    #root.option_add("*Font", "courier 12")
    root.option_add("*Font", "NewTimes 10")
    
    notebook = ttk.Notebook(root)
    with open(sys.argv[1], 'r') as f:
        yamldict = yaml.full_load(f)
    
    apps = []
    for section_key in yamldict.keys():
        app = tk.Frame(master=notebook)
        app.pack()
        
        sub1_dict = yamldict[section_key]
        for i, (sub1_k, sub1_v) in enumerate(sub1_dict.items()):
            if type(sub1_v) is not dict:
                tk.Label(app, text=sub1_k).grid(row=i, column=0, **_GRID_CFG1)
                tk.Label(app, text=sub1_v).grid(row=i, column=1, **_GRID_CFG1)
            else:
                sub2_dict = sub1_dict[sub1_k]
                lb1 = tk.LabelFrame(app, text=sub1_k)
                lb1.grid(row=i, column=0, columnspan=2, **_GRID_CFG1)
                
                for j, (sub2_k, sub2_v) in enumerate(sub2_dict.items()):
                    if type(sub2_v) is not dict:
                        tk.Label(lb1, text=sub2_k).grid(row=j, column=0, **_GRID_CFG1)
                        tk.Label(lb1, text=sub2_v).grid(row=j, column=1, **_GRID_CFG1)
                    else:
                        lb2 = tk.LabelFrame(lb1, text=sub2_k)
                        lb2.grid(row=0, column=2+j, padx=5, rowspan=5, ipadx=5, sticky='N')
                        sub3_dict=sub2_dict[sub2_k]
                        for k, (sub3_k, sub3_v) in enumerate(sub3_dict.items()):
                            if type(sub3_v) is not dict:
                                tk.Label(lb2, text=sub3_k).grid(row=k, column=0, **_GRID_CFG1)
                                tk.Label(lb2, text=sub3_v).grid(row=k, column=1, **_GRID_CFG1)
                            else:
                                tk.Button(lb2, text=sub3_k).grid(row=k, column=0, **_GRID_CFG1)
        #app = SectionUI(master=notebook, keys=yamldict[k].keys())
        notebook.add(app, text=section_key)

    notebook.pack()
    
    root.mainloop()



main()
