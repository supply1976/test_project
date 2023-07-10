import os, sys
from functools import reduce
import yaml
import tkinter as tk
import ttk
# python 2
#import tkFileDialog as filedialog

# python 3
from tkinter import filedialog

_GRID_CFG1 = {'padx':20, 'ipadx': 10, 'ipady':5, 'sticky':'W'}
_GRID_CFG2 = {'padx':5, 'ipadx': 5, 'ipady':5, 'sticky':'W'}



def open_new(root, ndict):
    newWindow = tk.Toplevel(root)
    appX = SectionUI(newWindow, "test")
    appX.gen_view(ndict)
    appX.pack()


class SectionFrame(tk.Frame):
    def __init__(self, master, sec_name):
        super(SectionFrame, self).__init__(master, name=sec_name.lower())
        self.master = master
        self.sec_name = sec_name
        self.levels = 0
        self.keys=[]
        self.btn_widgets=[]
        self.ent_widgets=[]
        
    def show_next(self, parent, nestdict):
        self.levels +=1
        # 1st 
        for i, (k, v) in enumerate(nestdict.items()):
            if type(v) is not dict:
                _key = "|".join([self.sec_name, k])
                self.keys.append(_key)                
                lb = tk.Label(parent, text=k)
                en = tk.Entry(parent, name=k.lower())
                if v is not None: en.insert(0, v)
                #en.config(state='disabled')
                if self.levels >=3:
                    lb.grid(row=0, column=i, **_GRID_CFG1)
                    en.grid(row=1, column=i, **_GRID_CFG1)
                else:
                    lb.grid(row=i, column=0, **_GRID_CFG1)
                    en.grid(row=i, column=1, **_GRID_CFG1)

                
            else:
                assert type(v) is dict
                subdict = v
                lbfr = tk.LabelFrame(parent, text=k, name=k.lower())
                if k=='CNN':
                    lbfr.grid(row=0, column=4, rowspan=4)
                    tk.Button(lbfr, text="CNN SETTINGS").pack()
                else:
                    lbfr.grid(row=i, column=0, columnspan=4, **_GRID_CFG1)
                    self.show_next(lbfr, subdict)
                    
                """
                # 2nd
                for j, (sk, sv) in enumerate(subdict.items()):
                    _key = "|".join([self.sec_name, k, sk])
                    self.keys.append(_key)
                    
                    if type(sv) is not dict:
                        if sv is None: sv='none'
                        tk.Label(lb, text=sk).grid(row=j, column=0, **_GRID_CFG1)
                        en = tk.Entry(lb, name=sk.lower())
                        en.insert(0, sv)
                        #en.config(state='disabled')
                        en.grid(row=j, column=1, **_GRID_CFG1)
                    else:
                        assert type(sv) is dict
                        btn=tk.Button(lb, text=sk, name=sk.lower())
                        self.btn_widgets.append(btn)
                        if k=='CNN':
                            btn.grid(row=j, column=0, columnspan=2, **_GRID_CFG1)
                        else:
                            btn.grid(row=0, column=2+j, rowspan=4, **_GRID_CFG2)
                """
        # end of for loop
        return 0
        
    def print_keys(self):
        print(self.keys)
                
    def _foo(self):
        pass


def get_update(root, yamldict, key):
    yaml_val = reduce(lambda y, x: y[x], key.split("|"), yamldict)
    print(yaml_val)
    key = '!notebook.' + key.lower().replace("|", ".")
    en_w = root.nametowidget(key)
    print(en_w.get())
    


def main():
    root = tk.Tk()
    root.title("Yaml Viewer")
    root.geometry('1024x768')
    root.configure(bg='#aaffff')
    #root.option_add("*Font", "courier 12")
    #root.option_add("*Font", "NewTimes 10")
    
    notebook = ttk.Notebook(root)
    
    with open(sys.argv[1], 'r') as f:
        yamldict = yaml.full_load(f)
    
    for tab_name in yamldict.keys():
        app = SectionFrame(master=notebook, sec_name=tab_name)
        app.show_next(parent=app, nestdict=yamldict[tab_name])
        notebook.add(app, text=tab_name)
        if len(app.btn_widgets)!=0:
            for w in app.btn_widgets:
                print(w, w.winfo_name())
    
    notebook.pack()

    cfg_fe_proc_type = "FEATURE_EXTRACTOR|PROCESS_TYPE"
    get_update(root, yamldict, cfg_fe_proc_type)
    tk.Button(root, text="update", 
        command=lambda: get_update(root, yamldict, cfg_fe_proc_type)).pack()
    
    #btn_cnn_struct = root.nametowidget('!notebook.training.cnn.structure')
    
    #btn_cnn_struct.config(
    #    command=lambda: open_new(root, yamldict['TRAINING']['CNN']['STRUCTURE']))

    
    root.mainloop()



main()
