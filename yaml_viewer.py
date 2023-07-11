import os, sys
import argparse
import collections
from functools import reduce
import yaml
import tkinter as tk
import tkinter.messagebox as tkMB
from tkinter import ttk

# python 2
#import tkFileDialog as filedialog

# python 3
from tkinter import filedialog


_GRID_CFG1 = {'padx':10, 'ipadx': 5, 'ipady':5, 'sticky':'W'}


class MenuBar:
    def __init__(self, root):
        self.root = root
        self.yaml_file = tk.StringVar()
        menuBar = tk.Menu(root)
        fileMenuItems = tk.Menu(menuBar)
        fileMenuItems.add_command(label="Open Yaml File")
        fileMenuItems.add_command(label="Save as ...", command=self.save_yaml_file)
        fileMenuItems.add_command(label="Quit", command=self.root.quit)
        helpMenu = tk.Menu(menuBar, tearoff=0)
        helpMenu.add_command(label="About", command=self.about_this_app)
        menuBar.add_cascade(label="File", menu=fileMenuItems)
        menuBar.add_cascade(label="Help", menu=helpMenu)
        self.root.config(menu=menuBar)
    
    def about_this_app(self):
        text = "this is yaml viewer and editor"
        tkMB.showinfo("About This", text)

    def open_yaml_file(self):
        fd = filedialog.askopenfilename(
            title="choose yaml file", filetypes=[("YAML", '*.yaml')])
        self.yaml_file.set(fd)
        
    def save_yaml_file(self):
        pass
        

class SectionFrame(tk.Frame, object):
    """
    used to load n-level nest dict (from yaml) 
    and assign an Entry widget for every yaml config input
    """
    def __init__(self, master, sec_name):
        super(SectionFrame, self).__init__(master, name=sec_name.lower())
        self.master = master
        self.sec_name = sec_name
        self.levels = 0
        self.entry_widgets=[]
        
    def show_next(self, parent, nestdict):
        self.levels +=1
        for i, (k, v) in enumerate(nestdict.items()):
            if type(v) is not dict:
                lb = tk.Label(parent, text=k)
                en = ttk.Entry(parent, name=k.lower())
                self.entry_widgets.append(en)
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
                pname = lbfr.winfo_parent()
                if pname.endswith('.feature_settings') or pname.endswith('.gauge'):
                    lbfr.grid(row=0, column=i, **_GRID_CFG1)
                else:
                    lbfr.grid(row=i+2, column=0, columnspan=4, **_GRID_CFG1)
                self.show_next(lbfr, subdict)
        # end of for loop
        return 0
            
    def _foo(self):
        pass


def open_new(title, ndict):
    root = tk.Toplevel()
    root.title(title)
    newfr = SectionFrame(master=root, sec_name='dummy')
    newfr.show_next(root, ndict)
    tk.Button(root, text="Update", width=30).grid(columnspan=5)
    for en in newfr.entry_widgets:
        print(en.winfo_name())
    #print(newfr.entry_widgets)
    
    
def nestdict_update(d, u):
    """
    d -- full nested dict (load from yaml)
    u -- dict to be update
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = nestdict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
     

def yaml_update(yamldict, all_entry_widgets):
    for en in all_entry_widgets:
        pname = en.winfo_parent()
        if pname.endswith('server_settings'): continue
        val = en.get()
        try:
            val = int(val)
        except:
            try:
                val = float(val)
            except:
                val = val   
        yaml_key = [x.upper() for x in en.bindtags()[0].split('.')[2:]]
        yaml_key.append(val)
        udict = reduce(lambda x,y: {y:x}, reversed(yaml_key))
        nestdict_update(yamldict, udict)
    with open('test_out.yaml', 'w') as f:
        yaml.dump(yamldict, f, sort_keys=False)
        
        
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yamlfile', type=str, nargs='?', default=None)
    FLAGS, _ = parser.parse_known_args()
    
    # root window settings    
    root = tk.Tk()
    root.title("Yaml Viewer/Editor")
    #root.geometry('800x600')
    root.configure(bg='#aaffff')
    #root.option_add("*Font", "courier 12")
    #root.option_add("*Font", "NewTimes 10")
    menubar = MenuBar(root)
    notebook = ttk.Notebook(root)
    
    # get yaml file from argument or from file dialog
    fn = FLAGS.yamlfile
            
    while fn is None:
        menubar.open_yaml_file()
        fn = menubar.yaml_file.get()

    if len(fn)==0:
        print("no yaml file load")
        return 0
            
    # load the yaml file and convert it to nest dict format
    with open(fn, 'r') as fd:
        yamldict = yaml.full_load(fd)

    #test_key = ['DATABASE','PATH', 'foo_path']
    #dbpath_update_dict = reduce(lambda x,y: {y:x}, reversed(test_key))
    #print(dbpath_update_dict)
    
    # create yaml viewer
    all_entry_widgets = []
    for tab_name in yamldict.keys():
        if tab_name == "GS_FLOW": continue         
        app = SectionFrame(master=notebook, sec_name=tab_name)
        app.show_next(parent=app, nestdict=yamldict[tab_name])
        all_entry_widgets.extend(app.entry_widgets)
        for en in app.entry_widgets:
            if en.winfo_parent().endswith('server_settings'):
                en.config(state='disable')
        notebook.add(app, text=tab_name)

    #for en in all_entry_widgets:
    #    yaml_key = [x.upper() for x in en.bindtags()[0].split('.')[2:]]
    #    #print(yaml_key)

    tk.Button(root, text="yaml update save as", 
        command=lambda: yaml_update(yamldict, all_entry_widgets)).pack()
    
    
    # customize CNN GUI
    d_cnnimgtr = yamldict['TRAINING']['CNN']['IMAGE_TRAINING']
    d_cnnepetr = yamldict['TRAINING']['CNN']['EPE_TRAINING']

    cnn_lbfr = root.nametowidget('!notebook.training.cnn')
    cnnimgtr_lbfr = root.nametowidget('!notebook.training.cnn.image_training')
    cnnepetr_lbfr = root.nametowidget('!notebook.training.cnn.epe_training')
    cnnstruct_lbfr = root.nametowidget('!notebook.training.cnn.structure')
    cnnimgtr_lbfr.grid_forget()
    cnnepetr_lbfr.grid_forget()
    cnnstruct_lbfr.grid_forget()
    
    cnnstruct_btn = ttk.Button(cnn_lbfr, text="STRUCTURE")
    cnnstruct_btn.grid(row=1, column=4, **_GRID_CFG1)
    # IMG
    cnnimgtr_btn = ttk.Button(cnn_lbfr, text="IMAGE TRAINING")
    cnnimgtr_btn.config(command=lambda: open_new('IMAGE_TRAINING', d_cnnimgtr))
    cnnimgtr_btn.grid(row=1, column=5, **_GRID_CFG1)
    
    # EPE
    cnnepetr_btn = ttk.Button(cnn_lbfr, text="EPE TRAINING")
    cnnepetr_btn.config(command=lambda: open_new('EPE_TRAINING', d_cnnepetr))
    cnnepetr_btn.grid(row=1, column=6, **_GRID_CFG1)
    

    notebook.pack()    
    root.mainloop()



main()
