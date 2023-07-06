import os, sys
import yaml
import tkinter as tk
import ttk
#import tkFileDialog as filedialog  ; # python 2
from tkinter import filedialog


class YamlEditor:
    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as f:
            self.yamldict = yaml.full_load(f)
        # DATABASE configs
        self.key_dbpath = "DATABASE|PATH"
        self.key_dbname = "DATRBASE|NAME"
        # FE configs
        self.key_fe_inputdata = "FEATURE_EXTRACTOR|INPUT_DATA"



class TrainingUI(tk.Frame):
    def __init__(self, master=None):
        super(TrainingUI, self).__init__(master)
        self.root_name = "TRAINING"
        # system keys
        self.key_gpu_num ="|SYSTEM|GPU_NUM"
        self.key_gpu_pool="|SYSTEM|GPU_POOL"
        # cnn keys
        self.key_cnn_inputs = "|CNN|INPUT_IMAGE_NAMES"
        self.key_cnn_init_algo= "CNN|INITIALIZATION|NETWORK_INITIALIZER"



class SectionUI(tk.Frame):
    def __init__(self, master=None, keys=None):
        super(SectionUI, self).__init__(master)
        for i, k in enumerate(keys):
            tk.Label(self, text=k).grid(
                row=i, column=0, padx=20, pady=10)
            
        #self.lbfr1 = tk.LabelFrame(self, text="SECTION 1")
        #self.lbfr1.pack(padx=20, pady=10, fill='both')
        #tk.Label(self.lbfr1, text="foo").grid(
        #    row=0, column=0)
        #self.lbfr2 = tk.LabelFrame(self, text="SECTION 2")
        #tk.Label(self.lbfr2, text="foo2").grid(
        #    row=0, column=0)
        #self.lbfr2.pack()
        self.pack()
        

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
    root.geometry('800x600')
    root.configure(bg='#aaffff')
    root.option_add("*Font", "courier 12")
    
    notebook = ttk.Notebook(root)
    
    yamldict = load_yaml_file(sys.argv[1])
    
    
    for k in yamldict.keys():
        app = SectionUI(master=notebook, keys=yamldict[k].keys())
        notebook.add(app, text=k)

    notebook.pack()
    
    root.mainloop()



main()
