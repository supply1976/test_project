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
_GRID_CFG2 = {'padx':10, 'ipadx': 5, 'ipady':5}


class MenuBar:
    def __init__(self, root):
        self.root = root
        self.yaml_file = tk.StringVar()
        menuBar = tk.Menu(root)
        fileMenuItems = tk.Menu(menuBar)
        fileMenuItems.add_command(label="Open Yaml File", command=self.open_yaml_file)
        fileMenuItems.add_command(label="Save as ...", command=self.save_yaml_file)
        fileMenuItems.add_command(label="Clean", command=self._clean)
        fileMenuItems.add_command(label="Quit", command=self.root.quit)
        helpMenu = tk.Menu(menuBar, tearoff=0)
        helpMenu.add_command(label="Info", command=self._info)
        helpMenu.add_command(label="About", command=self.about_this_app)
        menuBar.add_cascade(label="File", menu=fileMenuItems)
        menuBar.add_cascade(label="Help", menu=helpMenu)
        self.root.config(menu=menuBar)
    
    def _clean(self):
        notebook = self.root.children['!notebook']
        notebook.destroy()
       
    def _info(self):
        print(self.root.children)
        notebook = self.root.children['!notebook']
        print(notebook.children)
        
    def about_this_app(self):
        text = "this is yaml viewer and editor"
        tkMB.showinfo("About This", text)

    def open_yaml_file(self):
        fd = filedialog.askopenfilename(
            title="choose yaml file", filetypes=[("YAML", '*.yaml')])
        self.yaml_file.set(fd)
        
    def save_yaml_file(self):
        pass
        

class AppFrame(tk.Frame, object):
    """
    used to load n-level nest-dict (from yaml.full_load())
    get yaml config keys, 
    assign Entry widgets for all yaml config input
    """
    def __init__(self, master, app_name, **kwargs):
        super(AppFrame, self).__init__(master, name=app_name.lower(), **kwargs)
        self.master = master
        self.app_name = app_name
        self.levels = 0
        self.entry_widgets=[]
        self.yamlkeys_to_tkvars=[]
        
    def show_next(self, parent, nestdict):
        self.levels +=1
        for i, (k, v) in enumerate(nestdict.items()):
            if type(v) is not dict:
                lb = tk.Label(parent, text=k, name=k.lower()+"_label")
                if type(v) is bool:
                    en = ttk.Combobox(parent, name=k.lower(), width=16)
                    en.config(value=[False, True])
                    en.current(int(v))
                else:
                    en = tk.Entry(parent, name=k.lower(), width=18, fg='blue')
                yamlkey = [x.upper() for x in en.bindtags()[0].split('.')[2:]]
                yamlkey = "|".join(yamlkey)
                if type(v) is bool:
                    var = tk.BooleanVar(name=yamlkey, value=v)
                elif type(v) is int:
                    var = tk.IntVar(name=yamlkey, value=v)
                elif type(v) is float:
                    var = tk.DoubleVar(name=yamlkey, value=v)
                elif v is None:
                    var = tk.StringVar(name=yamlkey, value=v)
                elif type(v) is list:
                    v = " ".join(map(str, v))
                    var = tk.StringVar(name=yamlkey, value=v)
                else:
                    var = tk.StringVar(name=yamlkey, value=v)
                en.config(textvariable=var)
                
                self.yamlkeys_to_tkvars.append((yamlkey, var))
                self.entry_widgets.append(en)
                
                if self.levels >=3:
                    lb.grid(row=0, column=i, **_GRID_CFG1)
                    en.grid(row=1, column=i, **_GRID_CFG1)
                else:
                    lb.grid(row=i+2, column=0, **_GRID_CFG1)
                    en.grid(row=i+2, column=1, **_GRID_CFG1)
            else:
                assert type(v) is dict
                subdict = v
                lbfr = tk.LabelFrame(parent, text=k, name=k.lower())
                pname = lbfr.winfo_parent()
                if pname.endswith('.feature_settings') or pname.endswith('.gauge'):
                    lbfr.grid(row=0, column=i, **_GRID_CFG1)
                elif pname.endswith('training.system'):
                    lbfr.grid(row=0, column=i, rowspan=2, **_GRID_CFG1)
                else:
                    lbfr.grid(row=i+2, column=0, columnspan=4, **_GRID_CFG1)
                self.show_next(lbfr, subdict)
        # end of for loop
        return 0
            
    def _foo(self):
        pass



class CNNStructGUI(tk.Frame, object):
    def __init__(self, master, cnndict, name=None):
        super(CNNStructGUI, self).__init__(master)
        self.master = master
        self.cnndict = cnndict
        self.name = name
        self.layer_id = -1
        self.widgets = []
        self.other_nets = ['simple', 'unet626', 'forP76', 'resnet50', 'TBD']
        self.yamlkey_ksize = "TRAINING|CNN|STRUCTURE|KERNEL_SIZE"
        self.yamlkey_chans = "TRAINING|CNN|STRUCTURE|CHANNELS"
        self.yamlkey_actfs = "TRAINING|CNN|STRUCTURE|ACTIVATION_FUNCTIONS"
        self.yamlkey_symms = "TRAINING|CNN|STRUCTURE|SYMMETRY"
        self.yamlkey_skips = "TRAINING|CNN|_SKIP_INPUTS"
        _kern_dila = "TRAINING|CNN|_KERNEL_DILATION"
        self.yamlkey_kdila_en = _kern_dila+"|ENABLE"
        self.yamlkey_kdila_ra = _kern_dila+"|DILATION_RATE"

        # tk variables lists
        self.layerIDs = []
        self.skip_vars = []
        self.kern_vars = []
        self.chan_vars = []
        self.actf_vars = []
        self.symm_vars = []
        
    def _start(self, parent):
        lbfr_0 = tk.LabelFrame(parent, text=" MICS ")
        lbfr_0.pack()
        # update btn
        self.update_btn = tk.Button(lbfr_0, text="update structure", 
            command=self.update_cnn_tkvar)
        self.update_btn.grid(row=0, column=0, rowspan=2, sticky='NS', **_GRID_CFG2)
        # oldBKM2 btn
        self.oldBKM2_btn = tk.Button(lbfr_0, text="oldBKM2")
        self.oldBKM2_btn.grid(row=0, column=1, rowspan=2, sticky='NS', **_GRID_CFG2)
        # 
        self.unet66_btn = tk.Button(lbfr_0, text="unet66")
        self.unet66_btn.grid(row=0, column=2, rowspan=2, sticky='NS', **_GRID_CFG2)
        #
        self.resnet13_btn = tk.Button(lbfr_0, text="resnet13 (yan)")
        self.resnet13_btn.grid(row=0, column=3, rowspan=2, sticky='NS', **_GRID_CFG2)
        # 
        tk.Label(lbfr_0, text="other nets").grid(row=0, column=4, **_GRID_CFG2)
        self.others_cb = ttk.Combobox(lbfr_0, value=self.other_nets)
        self.others_cb.current(0)
        self.others_cb.grid(row=1, column=4, **_GRID_CFG2)

        self.lbfr = tk.LabelFrame(parent, text=" Net Structure ")
        self.lbfr.pack()
        tk.Label(self.lbfr, text="layerID").grid(row=0, column=0, **_GRID_CFG1)    
        tk.Label(self.lbfr, text="skip"   ).grid(row=0, column=1, **_GRID_CFG1)
        tk.Label(self.lbfr, text="ksize"  ).grid(row=0, column=2, **_GRID_CFG1)
        tk.Label(self.lbfr, text="channel").grid(row=0, column=3, **_GRID_CFG1)
        tk.Label(self.lbfr, text="actf"   ).grid(row=0, column=4, **_GRID_CFG1)
        tk.Label(self.lbfr, text="symm"   ).grid(row=0, column=5, **_GRID_CFG1)
        #tk.Label(self.lbfr, text="dilation").grid(row=0, column=6, **_GRID_CFG1)
        #
        self.add_layer_btn = tk.Button(self.lbfr, text="add layer")
        self.add_layer_btn.grid(row=0, column=7, **_GRID_CFG1)
        self.del_layer_btn = tk.Button(self.lbfr, text="del layer")
        self.del_layer_btn.grid(row=0, column=8, **_GRID_CFG1)
        
        # load CNN structure from yaml
        struct_d = self.cnndict['STRUCTURE']
        ksize = struct_d['KERNEL_SIZE']
        chans = struct_d['CHANNELS']
        actfs = struct_d['ACTIVATION_FUNCTIONS']
        symm = struct_d['SYMMETRY']
        _skips = self.cnndict['_SKIP_INPUTS']
        
        for i, (sk, k, c, a, s) in enumerate(zip(_skips, ksize, chans, actfs, symm)):
            #print(i, sk, k, c, a, s)
            self.add_layer(sk=sk, k=k, c=c, a=a, s=s)
        self.add_layer_btn.config(command=lambda: self.add_layer())
        self.del_layer_btn.config(command=lambda: self.del_layer())
            
    def del_layer(self):
        if self.layer_id <=-1:
            return 0
        [w.destroy() for w in self.widgets[-1]]
        self.widgets.pop(-1)
        self.skips.pop(-1)
        self.kerns.pop(-1)
        self.chans.pop(-1)
        self.actfs.pop(-1)
        self.symms.pop(-1)
        self.layer_id = self.layer_id -1
          
    def add_layer(self, sk=-1, k=1, c=1, a='linear', s='off'):
        self.layer_id = self.layer_id + 1
        tkvar_skip = tk.IntVar()
        tkvar_kern = tk.IntVar()
        tkvar_chan = tk.IntVar()
        #tkvar_skip = tk.StringVar()
        #tkvar_kern = tk.StringVar()
        #tkvar_chan = tk.StringVar()
        tkvar_actf = tk.StringVar()
        tkvar_symm = tk.StringVar()

        self.skip_vars.append(tkvar_skip)
        self.kern_vars.append(tkvar_kern)
        self.chan_vars.append(tkvar_chan)
        self.actf_vars.append(tkvar_actf)
        self.symm_vars.append(tkvar_symm)
        
        tk.Label(self.lbfr, text=str(self.layer_id)).grid(
            row=self.layer_id+1, column=0, **_GRID_CFG1)
        wid_skip = ttk.Combobox(self.lbfr, 
            values=list(range(-1, self.layer_id+1)), textvariable=tkvar_skip)
        wid_skip.set(sk)
        wid_skip.grid(row=self.layer_id+1, column=1, **_GRID_CFG1)
        wid_kern = ttk.Combobox(self.lbfr, 
            values=list(range(3, 43, 2)), textvariable=tkvar_kern)
        wid_kern.grid(row=self.layer_id+1, column=2, **_GRID_CFG1)
        wid_kern.set(k)
        wid_chan = ttk.Combobox(self.lbfr, 
            values=list(range(1, 17)), textvariable=tkvar_chan)
        wid_chan.grid(row=self.layer_id+1, column=3, **_GRID_CFG1)
        wid_chan.set(c)
        wid_actf = ttk.Combobox(self.lbfr, 
            values=['linear', 'sigmoid', 'relu'], textvariable=tkvar_actf)
        wid_actf.grid(row=self.layer_id+1, column=4, **_GRID_CFG1)
        wid_actf.set(a)
        wid_symm = ttk.Combobox(self.lbfr, 
            values=['all', 'xy_mirror', 'off'], textvariable=tkvar_symm)
        wid_symm.grid(row=self.layer_id+1, column=5, **_GRID_CFG1)
        wid_symm.set(s)
        self.widgets.append([wid_skip, wid_kern, wid_chan, wid_actf, wid_symm])
    
    #TODO
    def chans_splits_update(self):
        for var in self.chan_vars:
            val = var.get()
            try:
                val = int(val)
            except
                val = val.replace(" ", "").split(",")
                val = list(map(int, val))
        
    def update_cnn_tkvar(self):
        _skip_inputs = [_.get() for _ in self.skips]
        kernel_size = [_.get() for _ in self.kerns]
        channels = [_.get() for _ in self.chans]
        activation_functions = [_.get() for _ in self.actfs]
        symmetry = [_.get() for _ in self.symms]
        #print(_skip_inputs)
        #print(kernel_size)
        #print(channels)
        #print(activation_functions)
        #print(symmetry)
        tk.StringVar(name=self.yamlkey_ksize).set(kernel_size)
        tk.StringVar(name=self.yamlkey_chans).set(channels)
        tk.StringVar(name=self.yamlkey_actfs).set(activation_functions)
        tk.StringVar(name=self.yamlkey_symms).set(symmetry)
        tk.StringVar(name=self.yamlkey_skips).set(_skip_inputs)       


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
        
        

class UpdateUtil(object):
    def __init__(self, yamldict, all_entry_widgets, all_yamlkeys_to_tkvars):
        self.yamldict = yamldict
        self.all_entry_widgets = all_entry_widgets
        self.all_yamlkeys_to_tkvars = all_yamlkeys_to_tkvars
        self.saveto_fn = tk.StringVar()
        self.saveto_fn.set('test_out.yaml')
        
    def yaml_update(self):
        #yamlkey = None
        list_input_str_keys = [
        "TRAINING|CNN|INPUT_IMAGE_NAMES",
        "TRAINING|CNN|IMAGE_TRAINING|TARGET_IMAGE|NAME",
        "TRAINING|CNN|STRUCTURE|ACTIVATION_FUNCTIONS",
        "TRAINING|CNN|STRUCTURE|SYMMETRY"]
        list_input_int_keys = [
        "TRAINING|CNN|STRUCTURE|KERNEL_SIZE",
        "TRAINING|CNN|STRUCTURE|CHANNELS",
        "TRAINING|CNN|_SKIP_INPUTS"]
        list_input_float_keys = [
        "TRAINING|CNN|IMAGE_TRAINING|TARGET_IMAGE|COEFFICIENT"]
    
        for (yamlkey, tkvar) in self.all_yamlkeys_to_tkvars.items():
            val = tkvar.get()
        
            if yamlkey in list_input_str_keys:
                val = val.split(" ")
            elif yamlkey in list_input_int_keys:
                val = val.split(" ")
                val = list(map(int, val))
            elif yamlkey in list_input_float_keys:
                val = val.split(" ")
                val = list(map(float, val))
            else:
                pass  
            yamlkey_list = yamlkey.split("|")
            yamlkey_list.append(val)
            udict = reduce(lambda x,y: {y:x}, reversed(yamlkey_list))
            nestdict_update(self.yamldict, udict)
        #print(self.yamldict['DATABASE']['PATH'])
        #print(self.yamldict['TRAINING']['CNN']['STRUCTURE']['KERNEL_SIZE'])
        
        with open(self.saveto_fn.get(), 'w') as f:
            yaml.dump(self.yamldict, f, sort_keys=False)
        

    def open_cnn_gui(self):
        root = tk.Toplevel()
        root.title("network structure")
        self.yaml_update()
        cnndict = self.yamldict['TRAINING']['CNN']
        cnn_gui = CNNStructGUI(root, cnndict=cnndict, name=None)
        cnn_gui._start(root)


    def imgtr_hp_update(self):
        new_kv = [("TRAINING|CNN|IMAGE_TRAINING|"+k, v) for (k, v) in self.imgkv]
        for (k, v) in new_kv:
            self.all_yamlkeys_to_tkvars[k].set(v.get())
   
    def epetr_hp_update(self):
        new_kv = [("TRAINING|CNN|EPE_TRAINING|"+k, v) for (k, v) in self.epekv]
        for (k, v) in new_kv:
            self.all_yamlkeys_to_tkvars[k].set(v.get())

    def open_imgtr_gui(self):
        root = tk.Toplevel()
        root.title("IMG Training")
        self.yaml_update()
        imgtr_dict = self.yamldict['TRAINING']['CNN']['IMAGE_TRAINING']
        newfr = AppFrame(master=root, app_name='imgtr')
        newfr.show_next(root, imgtr_dict)
        self.imgkv = newfr.yamlkeys_to_tkvars
        #self.imgtr_hp_update()
        tk.Button(root, text="Update Setting", 
            command=self.imgtr_hp_update).grid(columnspan=5)

    def open_epetr_gui(self):
        root = tk.Toplevel()
        root.title("EPE Training")
        self.yaml_update()
        epetr_dict = self.yamldict['TRAINING']['CNN']['EPE_TRAINING']
        newfr = AppFrame(master=root, app_name='epetr')
        newfr.show_next(root, epetr_dict)
        self.epekv = newfr.yamlkeys_to_tkvars
        #self.epetr_hp_update()
        tk.Button(root, text="Update Setting", 
            command=self.epetr_hp_update).grid(columnspan=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yamlfile', type=str, nargs='?', default=None)
    FLAGS, _ = parser.parse_known_args()
    
    # root window settings    
    root = tk.Tk()
    root.title("Yaml Viewer/Editor")
    root.geometry('1400x900')
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
    
    ksize = yamldict['TRAINING']['CNN']['STRUCTURE']['KERNEL_SIZE']
    cnndict = yamldict['TRAINING']['CNN']
    if "_SKIP_INPUTS" not in cnndict.keys():
        yamldict['TRAINING']['CNN']['_SKIP_INPUTS']=[-1]*len(ksize)
    
    #test_key = ['FEATURE_EXTRACTOR', 'INPUT_DATA', 'FOV_TILING']
    #d = reduce(lambda x,y: x[y], test_key, yamldict)
    #print(d)
    #test_key = ['DATABASE','PATH', 'foo_path']
    #dbpath_update_dict = reduce(lambda x,y: {y:x}, reversed(test_key))
    #print(dbpath_update_dict)
    
    # create yaml viewer
    all_entry_widgets = []
    all_yamlkeys_to_tkvars = []
    for tab_name in yamldict.keys():
        # not show "GS_FLOW" section
        if tab_name == "GS_FLOW": continue        
        app = AppFrame(master=notebook, app_name=tab_name)
        app.show_next(parent=app, nestdict=yamldict[tab_name])
        all_yamlkeys_to_tkvars.extend(app.yamlkeys_to_tkvars)
        all_entry_widgets.extend(app.entry_widgets)
        for en in app.entry_widgets:
            if en.winfo_parent().endswith('server_settings'):
                en.config(state='disable')
        notebook.add(app, text=tab_name)
    
    all_yamlkeys_to_tkvars = dict(all_yamlkeys_to_tkvars)
    update_util=UpdateUtil(yamldict, all_entry_widgets, all_yamlkeys_to_tkvars)    
    
    froot = tk.Frame(root)
    froot.pack()
    top_btn = tk.Button(froot, text="yaml update save to",
        command=lambda: update_util.yaml_update())
    top_btn.grid(row=0, column=0, **_GRID_CFG1)
    top_en = tk.Entry(froot, textvariable=update_util.saveto_fn)
    top_en.grid(row=0, column=1, **_GRID_CFG1) 
    
    # customize CNN GUI
    wid_cnn_lbfr = root.nametowidget('!notebook.training.cnn')
    wid_cnnstruc_lbfr = root.nametowidget('!notebook.training.cnn.structure')
    wid_cnnimgtr_lbfr = root.nametowidget('!notebook.training.cnn.image_training')
    wid_cnnepetr_lbfr = root.nametowidget('!notebook.training.cnn.epe_training')
    wid_cnnimgtr_lbfr.grid_forget()
    wid_cnnepetr_lbfr.grid_forget()
    wid_cnnstruc_lbfr.grid_forget()
    root.nametowidget('!notebook.training.cnn._skip_inputs_label').grid_forget()    
    root.nametowidget('!notebook.training.cnn._skip_inputs').grid_forget()
    
    cnnstruc_btn = ttk.Button(wid_cnn_lbfr, text="STRUCTURE")
    cnnstruc_btn.config(command=lambda: update_util.open_cnn_gui())
    cnnstruc_btn.grid(row=1, column=1, **_GRID_CFG1)

    # IMG
    cnnimgtr_btn = ttk.Button(wid_cnn_lbfr, text="IMAGE TRAINING")
    cnnimgtr_btn.config(command=lambda: update_util.open_imgtr_gui())
    cnnimgtr_btn.grid(row=1, column=2, **_GRID_CFG1)
    
    # EPE
    cnnepetr_btn = ttk.Button(wid_cnn_lbfr, text="EPE TRAINING")
    cnnepetr_btn.config(command=lambda: update_util.open_epetr_gui())
    cnnepetr_btn.grid(row=1, column=3, **_GRID_CFG1)
    
    notebook.pack()
    root.mainloop()



main()
