#
import os, sys
import numpy as np
import argparse
import collections
import itertools
from functools import reduce
import yaml
import tkinter as tk
import tkinter.messagebox as tkMB
from tkinter import ttk
from tkinter import filedialog


_GRID_CFG1 = {'padx':10, 'ipadx': 5, 'ipady':5, 'sticky':'W'}
_GRID_CFG2 = {'padx':10, 'ipadx': 5, 'ipady':5}


class MenuBar:
    def __init__(self, root):
        self.root = root
        self.yaml_file = tk.StringVar()
        menuBar = tk.Menu(root)
        fileMenuItems = tk.Menu(menuBar)
        #fileMenuItems.add_command(label="Open Yaml File", command=self.open_yaml_file)
        #fileMenuItems.add_command(label="Save as ...", command=self.save_yaml_file)
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
        text = "this is yaml viewer, editor, generator"
        tkMB.showinfo("About This", text)

    def open_yaml_file(self):
        fd = filedialog.askopenfilename(
            title="choose yaml file", filetypes=[("YAML", '*.yaml')])
        self.yaml_file.set(fd)
        

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
        for i, (k, v) in enumerate(nestdict.items()):
            if type(v) is not dict:
                lb = tk.Label(parent, text=k, name=k.lower()+"_label")
                if type(v) is bool:
                    en = ttk.Combobox(parent, name=k.lower(), width=12)
                    en.config(value=[False, True])
                    en.current(int(v))
                else:
                    en = None
                    if k=="NETWORK_INITIALIZER":
                        en = ttk.Combobox(parent, name=k.lower(), width=12)
                        en.config(value=[
                            'glorot_uniform', 'lecun_uniform', 'checkpoint_capsule'])
                    else:
                        en = tk.Entry(parent, name=k.lower(), width=14, fg='blue')
                #        
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
                    v = " ".join(list(map(str, v)))
                    var = tk.StringVar(name=yamlkey, value=v)
                else:
                    var = tk.StringVar(name=yamlkey, value=v)
                en.config(textvariable=var)
                
                self.yamlkeys_to_tkvars.append((yamlkey, var))
                self.entry_widgets.append(en)
                
                if self.levels >1:
                    lb.grid(row=0, column=i, **_GRID_CFG1)
                    en.grid(row=1, column=i, **_GRID_CFG1)
                else:
                    lb.grid(row=i, column=0, **_GRID_CFG1)
                    en.grid(row=i, column=1, **_GRID_CFG1)
            else:
                self.levels += 1
                assert type(v) is dict
                subdict = v
                lbfr = tk.LabelFrame(parent, text=k, name=k.lower())
                pname = lbfr.winfo_parent()
                if pname.endswith('.feature_settings') or pname.endswith('.gauge'):
                    lbfr.grid(row=0, column=i, **_GRID_CFG1)
                elif pname.endswith('training.system'):
                    lbfr.grid(row=0, column=i, rowspan=2, **_GRID_CFG1)
                else:
                    lbfr.grid(row=i+2, column=0, columnspan=7, **_GRID_CFG1)
                self.show_next(lbfr, subdict)
        # end of for loop
        return 0
            
    def _foo(self):
        pass


class CNNStructGUI(tk.Frame, object):
    def __init__(self, master, update_util, cnndict, name=None):
        super(CNNStructGUI, self).__init__(master)
        self.master = master
        self.update_util = update_util
        self.cnndict = cnndict
        self.name = name
        self.layer_id = -1
        self.en_widgets = []
        self.net_names = ['unet636', 'resnet50', 'TBD']
        self.yamlkey_kerns = "TRAINING|CNN|STRUCTURE|KERNEL_SIZE"
        self.yamlkey_chans = "TRAINING|CNN|STRUCTURE|CHANNELS"
        self.yamlkey_actfs = "TRAINING|CNN|STRUCTURE|ACTIVATION_FUNCTIONS"
        self.yamlkey_symms = "TRAINING|CNN|STRUCTURE|SYMMETRY"
        self.yamlkey_skips = "TRAINING|CNN|_SKIP_INPUTS"
        #_kern_dila = "TRAINING|CNN|_KERNEL_DILATION"
        #self.yamlkey_kdila_en = _kern_dila+"|ENABLE"
        #self.yamlkey_kdila_ra = _kern_dila+"|DILATION_RATE"
        
        # tk variables containers
        self.skip_vars = []
        self.kern_vars = []
        self.chan_vars = []
        self.actf_vars = []
        self.symm_vars = []
    
    def _foo(self):
        tkMB.showinfo("Orz", "Not Yet Implemented ...")
    
    def gui_start(self, parent):
        lbfr_0 = tk.LabelFrame(parent, text=" MICS ")
        lbfr_0.pack()
        # update struct btn
        self.update_struct_btn = tk.Button(lbfr_0, text="update structure")
        self.update_struct_btn.grid(
            row=0, column=0, rowspan=2, sticky='NS', **_GRID_CFG2)
        # 
        self.simple_net_btn = tk.Button(lbfr_0, text="simple")
        self.simple_net_btn.grid(row=0, column=1, rowspan=2, sticky='NS', **_GRID_CFG2)
        # 
        self.oldBKM2_btn = tk.Button(lbfr_0, text="oldBKM2")
        self.oldBKM2_btn.grid(row=0, column=2, rowspan=2, sticky='NS', **_GRID_CFG2)
        # 
        self.unet66_btn = tk.Button(lbfr_0, text="unet66")
        self.unet66_btn.grid(row=0, column=3, rowspan=2, sticky='NS', **_GRID_CFG2)
        # 
        self.resnet13_btn = tk.Button(lbfr_0, text="resnet13 (yan)")
        self.resnet13_btn.grid(row=0, column=4, rowspan=2, sticky='NS', **_GRID_CFG2)
        # 
        tk.Label(lbfr_0, text="other nets").grid(row=0, column=5, **_GRID_CFG2)
        self.others_cb = ttk.Combobox(lbfr_0, value=self.net_names)
        self.others_cb.current(0)
        self.others_cb.grid(row=1, column=5, **_GRID_CFG2)
       
        self.lbfr = tk.LabelFrame(parent, text=" Net Structure ")
        self.lbfr.pack()
        tk.Label(self.lbfr, text="layerID").grid(row=0, column=0, **_GRID_CFG1)    
        tk.Label(self.lbfr, text="skip"   ).grid(row=0, column=1, **_GRID_CFG1)
        tk.Label(self.lbfr, text="kern"   ).grid(row=0, column=2, **_GRID_CFG1)
        tk.Label(self.lbfr, text="chan"   ).grid(row=0, column=3, **_GRID_CFG1)
        tk.Label(self.lbfr, text="actf"   ).grid(row=0, column=4, **_GRID_CFG1)
        tk.Label(self.lbfr, text="symm"   ).grid(row=0, column=5, **_GRID_CFG1)
        # placeholder for dilation, TODO later
        tk.Label(self.lbfr, text="dilation").grid(row=0, column=6, **_GRID_CFG1)
        #
        self.add_layer_btn = tk.Button(self.lbfr, text="add layer")
        self.add_layer_btn.grid(row=0, column=7, **_GRID_CFG1)
        self.del_layer_btn = tk.Button(self.lbfr, text="del layer")
        self.del_layer_btn.grid(row=0, column=8, **_GRID_CFG1)
        
        # load CNN structure configs from original yaml dict
        cnn_struct_d = self.cnndict['STRUCTURE']
        kerns = cnn_struct_d['KERNEL_SIZE']
        chans = cnn_struct_d['CHANNELS']
        actfs = cnn_struct_d['ACTIVATION_FUNCTIONS']
        symms = cnn_struct_d['SYMMETRY']
        skips = self.cnndict['_SKIP_INPUTS']
        
        for i, (sk, k, c, a, s) in enumerate(zip(skips, kerns, chans, actfs, symms)):
            self.add_layer(sk=sk, k=k, c=c, a=a, s=s)
        self.update_struct_btn.config(command=self.update_btn_event_handle)
        self.add_layer_btn.config(command=lambda: self.add_layer())
        self.del_layer_btn.config(command=lambda: self.del_layer())
        self.simple_net_btn.config(command=self.load_simple_net)
        self.oldBKM2_btn.config(command=self.load_bkm2_struct)
        self.unet66_btn.config(command=self.load_unet66_struct)
        self.resnet13_btn.config(command=self.load_resnet13_struct)
    
    def load_simple_net(self):
        num_layers = 1+self.layer_id
        for _ in range(num_layers):
            self.del_layer()
        assert self.layer_id == -1
        kerns = [21,41,31,51]
        chans = [ 2, 2, 2, 1]
        skips = [-1, 0,-1, 2]
        actfs = ['sigmoid']*3+['linear']
        symms = ['all']*4
        for (sk, k, c, a, s) in zip(skips, kerns, chans, actfs, symms):
            self.add_layer(sk=sk, k=k, c=c, a=a, s=s)
    
    def load_bkm2_struct(self):
        num_layers = 1+self.layer_id
        for _ in range(num_layers):
            self.del_layer()
        assert self.layer_id == -1
        kerns = [21, 41, 31, 51, 61, 1]
        chans = [ 4,  4,  4,  4,  4, 1]
        skips = [-1,  0, -1,  2, -1, 4]
        actfs = ['sigmoid']*5 + ['linear']
        symms = ['xy_mirror']*5 + ['off']
        for (sk, k, c, a, s) in zip(skips, kerns, chans, actfs, symms):
            self.add_layer(sk=sk, k=k, c=c, a=a, s=s)

    def load_unet66_struct(self):
        num_layers = 1+self.layer_id
        for _ in range(num_layers):
            self.del_layer()
        assert self.layer_id == -1
        kerns = [19,41,23,37,29,31, 1, 1, 1, 1, 1, 1]
        chans = [ 2, 2, 3, 3, 4, 4, 4, 4, 3, 3, 2, 1]
        skips = [-1, 0,-1, 2,-1, 4, 5, 4, 3, 2, 1,-1]
        actfs = ['sigmoid']*6 + ['linear']*6
        symms = ['xy_mirror']*6 + ['off']*6
        for (sk, k, c, a, s) in zip(skips, kerns, chans, actfs, symms):
            self.add_layer(sk=sk, k=k, c=c, a=a, s=s)

    def load_resnet13_struct(self):
        num_layers = 1+self.layer_id
        for _ in range(num_layers):
            self.del_layer()
        assert self.layer_id == -1
        kerns = [ 7, 5, 3, 7, 5, 3, 7, 5, 3, 7, 5, 3, 3]
        chans = [16,16,16,16,16,16, 8, 8, 8, 8, 8, 8, 1]
        skips = [-1,-1, 0,-1,-1, 3,-1,-1, 6,-1,-1, 9, 0]
        actfs = ['relu']*12 + ['linear']
        symms = ['xy_mirror']*13
        for (sk, k, c, a, s) in zip(skips, kerns, chans, actfs, symms):
            self.add_layer(sk=sk, k=k, c=c, a=a, s=s)
                
    def del_layer(self):
        if self.layer_id <=-1:
            return 0
        [w.destroy() for w in self.en_widgets[-1]]
        self.en_widgets.pop(-1)
        self.skip_vars.pop(-1)
        self.kern_vars.pop(-1)
        self.chan_vars.pop(-1)
        self.actf_vars.pop(-1)
        self.symm_vars.pop(-1)
        self.layer_id = self.layer_id -1
          
    def add_layer(self, sk=-1, k=1, c=1, a='linear', s='off'):
        # single layer setting
        self.layer_id = self.layer_id + 1
        tkvar_skip = tk.IntVar()
        #tkvar_kern = tk.IntVar() 
        #tkvar_chan = tk.IntVar()
        tkvar_kern = tk.StringVar()   # allow split 
        tkvar_chan = tk.StringVar()   # allow split
        tkvar_actf = tk.StringVar()
        tkvar_symm = tk.StringVar()
        tkvar_skip.set(sk)
        tkvar_kern.set(k)
        tkvar_chan.set(c)
        tkvar_actf.set(a)
        tkvar_symm.set(s)
        self.skip_vars.append(tkvar_skip)
        self.kern_vars.append(tkvar_kern)
        self.chan_vars.append(tkvar_chan)
        self.actf_vars.append(tkvar_actf)
        self.symm_vars.append(tkvar_symm)
        
        tk.Label(self.lbfr, text=str(self.layer_id)).grid(
            row=self.layer_id+1, column=0, **_GRID_CFG1)
        wid_skip = ttk.Combobox(self.lbfr, width=14,
            values=list(range(-1, self.layer_id+1)), textvariable=tkvar_skip)
        wid_skip.grid(row=self.layer_id+1, column=1, **_GRID_CFG1)
        
        wid_kern = ttk.Combobox(self.lbfr, width=14,
            values=list(range(3, 43, 2)), textvariable=tkvar_kern)
        wid_kern.grid(row=self.layer_id+1, column=2, **_GRID_CFG1)
        
        wid_chan = ttk.Combobox(self.lbfr, width=14,
            values=list(range(1, 17)), textvariable=tkvar_chan)
        wid_chan.grid(row=self.layer_id+1, column=3, **_GRID_CFG1)
        
        wid_actf = ttk.Combobox(self.lbfr, width=14,
            values=['linear', 'sigmoid', 'relu'], textvariable=tkvar_actf)
        wid_actf.grid(row=self.layer_id+1, column=4, **_GRID_CFG1)
        
        wid_symm = ttk.Combobox(self.lbfr, width=14,
            values=['all', 'xy_mirror', 'off'], textvariable=tkvar_symm)
        wid_symm.grid(row=self.layer_id+1, column=5, **_GRID_CFG1)
        
        self.en_widgets.append([wid_skip, wid_kern, wid_chan, wid_actf, wid_symm])

    def update_btn_event_handle(self):
        self.get_cnn_struct_values()
        
    def get_cnn_struct_values(self):
        self.struct_values_combo = []
        self.skips_value = [_.get() for _ in self.skip_vars]
        self.kerns_value = [_.get() for _ in self.kern_vars]
        self.chans_value = [_.get() for _ in self.chan_vars]
        self.actfs_value = [_.get() for _ in self.actf_vars]
        self.symms_value = [_.get() for _ in self.symm_vars]
        

        for j, k in enumerate(self.kerns_value):
            if not hasattr(k, '__iter__'):
                self.kerns_value[j] = [int(k)]
            else:
                self.kerns_value[j] = list(map(int, k.split(",")))
        kerns_combo = list(itertools.product(*self.kerns_value))

        for i, c in enumerate(self.chans_value):
            if not hasattr(c, '__iter__'):
                self.chans_value[i] = [int(c)]
            else:
                self.chans_value[i] = list(map(int, c.split(",")))
        chans_combo = list(itertools.product(*self.chans_value))

        if len(kerns_combo)==1 and len(chans_combo)==1:
            tk.StringVar(name=self.yamlkey_skips).set(self.skips_value)
            tk.StringVar(name=self.yamlkey_kerns).set(kerns_combo[0])
            tk.StringVar(name=self.yamlkey_chans).set(chans_combo[0])
            tk.StringVar(name=self.yamlkey_actfs).set(self.actfs_value)
            tk.StringVar(name=self.yamlkey_symms).set(self.symms_value)
            self.update_util.nets_combo = None
        else:
            all_nets_combo = list(itertools.product(
                [self.skips_value],
                kerns_combo,
                chans_combo,
                [self.actfs_value],
                [self.symms_value]))
        
            self.update_util.nets_combo = np.array(all_nets_combo, dtype=object)


class UpdateUtil(object):
    def __init__(self, yamldict, all_entry_widgets, all_yamlkeys_to_tkvars):
        self.yamldict = yamldict
        self.all_entry_widgets = all_entry_widgets
        self.all_yamlkeys_to_tkvars = all_yamlkeys_to_tkvars
        self.saveto_fn = tk.StringVar()
        self.saveto_fn.set('test_out.yaml')
        self.nets_combo = None
        self.yamlkey_kerns = "TRAINING|CNN|STRUCTURE|KERNEL_SIZE"
        self.yamlkey_chans = "TRAINING|CNN|STRUCTURE|CHANNELS"
        self.yamlkey_actfs = "TRAINING|CNN|STRUCTURE|ACTIVATION_FUNCTIONS"
        self.yamlkey_symms = "TRAINING|CNN|STRUCTURE|SYMMETRY"
        self.yamlkey_skips = "TRAINING|CNN|_SKIP_INPUTS"
            
    def yaml_update(self):
        list_input_str_keys = [
        "TRAINING|CNN|INPUT_IMAGE_NAMES",
        "TRAINING|CNN|IMAGE_TRAINING|TARGET_IMAGE|NAME",
        "TRAINING|CNN|STRUCTURE|ACTIVATION_FUNCTIONS",
        "TRAINING|CNN|STRUCTURE|SYMMETRY",
        "TRAINING|CNN|EPE_TRAINING|AR_ENHANCEMENT|STRUCTURE|ACTIVATION_FUNCTIONS"]
        list_input_int_keys = [
        "TRAINING|CNN|STRUCTURE|KERNEL_SIZE",
        "TRAINING|CNN|STRUCTURE|CHANNELS",
        "TRAINING|CNN|_SKIP_INPUTS",
        "TRAINING|CNN|EPE_TRAINING|AR_ENHANCEMENT|STRUCTURE|NEURON"]
        list_input_float_keys = [
        "TRAINING|CNN|IMAGE_TRAINING|TARGET_IMAGE|COEFFICIENT"]
    
        for (yamlkey, tkvar) in self.all_yamlkeys_to_tkvars.items():
            val = tkvar.get()
            if type(val) is str:
                if len(val)==0:
                    val = None
        
            if yamlkey in list_input_str_keys and val is not None:
                val = val.split(" ")
            elif yamlkey in list_input_int_keys and val is not None:
                val = val.split(" ")
                val = list(map(int, val))
            elif yamlkey in list_input_float_keys and val is not None:
                val = val.split(" ")
                val = list(map(float, val))
            else:
                pass  
            yamlkey_list = yamlkey.split("|")
            yamlkey_list.append(val)
            udict = reduce(lambda x,y: {y:x}, reversed(yamlkey_list))
            nestdict_update(self.yamldict, udict)

        
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


def single_yaml_saveto(update_util):
    update_util.yaml_update()
    with open(update_util.saveto_fn.get(), 'w') as f:
        yaml.dump(update_util.yamldict, f, sort_keys=False)
        
        
def nets_combo_export(update_util):
    scripts = []
    assert update_util.nets_combo is not None
    update_util.yaml_update()
    fn, ext = os.path.splitext(update_util.saveto_fn.get())
    print(update_util.nets_combo.shape)
    for i, t in enumerate(update_util.nets_combo):
        tid = str(i+1).zfill(4)
        new_fn = fn+"-"+tid+ext
        sk = " ".join(list(map(str, t[0])))
        k = " ".join(list(map(str, t[1])))
        c = " ".join(list(map(str, t[2])))
        a = " ".join(t[3])
        s = " ".join(t[4])        
  
        tk.StringVar(name=update_util.yamlkey_skips).set(sk)
        tk.StringVar(name=update_util.yamlkey_kerns).set(k)
        tk.StringVar(name=update_util.yamlkey_chans).set(c)
        tk.StringVar(name=update_util.yamlkey_actfs).set(a)
        tk.StringVar(name=update_util.yamlkey_symms).set(s)
        update_util.yaml_update()
        with open(new_fn, 'w') as f:
            yaml.dump(update_util.yamldict, f, sort_keys=False)
        scripts.append(" ".join(['adv_modeling', '-tr', new_fn, '--gpu', '\n']))
        scripts.append("sleep 30 ; rm -rf .proj ; sleep 10 \n")
    with open('run_all_yamls.sh', 'w') as f:
        f.writelines(scripts)
    os.chmod('./run_all_yamls.sh', 0o0777)
    

def open_cnn_gui(update_util):
    root = tk.Toplevel()
    root.title("network structure")
    update_util.yaml_update()
    cnndict = update_util.yamldict['TRAINING']['CNN']
    cnn_gui = CNNStructGUI(root, update_util, cnndict=cnndict, name=None)
    cnn_gui.gui_start(root)

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
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yamlfile', type=str, nargs='?', default=None)
    FLAGS, _ = parser.parse_known_args()
    # root window settings    
    root = tk.Tk()
    root.title("Yaml Viewer/Editor/Generator")
    #root.geometry('1400x900')
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
    
    if 'TRAINING' in yamldict.keys():
        ksize = yamldict['TRAINING']['CNN']['STRUCTURE']['KERNEL_SIZE']
        cnndict = yamldict['TRAINING']['CNN']
        if "_SKIP_INPUTS" not in cnndict.keys():
            yamldict['TRAINING']['CNN']['_SKIP_INPUTS']=[-1]*len(ksize)
    
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
    
    app_assess = tk.Frame(notebook)
    tk.Label(app_assess, text="under construction...").pack()
    notebook.add(app_assess, text="Assessment")
    
    all_yamlkeys_to_tkvars = dict(all_yamlkeys_to_tkvars)
    update_util=UpdateUtil(yamldict, all_entry_widgets, all_yamlkeys_to_tkvars)    
    
    topfr = tk.Frame(root)
    topfr.pack()
    top_btn = tk.Button(topfr, text="yaml update save to",
        command=lambda: single_yaml_saveto(update_util))
    top_btn.grid(row=0, column=0, **_GRID_CFG1)
    top_en = tk.Entry(topfr, textvariable=update_util.saveto_fn)
    top_en.grid(row=0, column=1, **_GRID_CFG1)
    tk.Button(topfr, text="splits combo export all",
        command=lambda: nets_combo_export(update_util)).grid(
            row=0, column=2, **_GRID_CFG1) 
    
    nbname=notebook._name
    # customize DATABASE tab GUI
    app_database = root.nametowidget(nbname+'.database')
    tk.Button(app_database, text="Browse").grid(row=0, column=2, **_GRID_CFG1)
    en_dbpath = root.nametowidget(nbname+'.database.path')
    en_dbpath.config(width=60)
    tk.Button(app_database, text="get available names").grid(row=1, column=2, **_GRID_CFG2)
   
    #customize FE tab GUI
    fe_inp_data_name = nbname+'.feature_extractor.input_data'
    en_feModelFile = root.nametowidget(fe_inp_data_name+'.model.file')
    en_feLayoutFile = root.nametowidget(fe_inp_data_name+'.layout.file')
    en_feGaugeFileCD = root.nametowidget(fe_inp_data_name+'.gauge.file.cd_asd')
    en_feGaugeFileEP = root.nametowidget(fe_inp_data_name+'.gauge.file.ep_asd')

    en_feModelFile.config(width=60)
    en_feLayoutFile.config(width=60)
    en_feGaugeFileCD.config(width=60)
    en_feGaugeFileEP.config(width=60)
    lbfr_fe_asdcolnamePW = root.nametowidget(
        fe_inp_data_name+'.gauge.asd_column_name.process_window')
    lbfr_fe_asdcolnamePW.grid_forget()
    lbfr_fe_asdcolnamePW.grid(row=0, column=1, rowspan=2)
    
    # customize CNN GUI
    if 'TRAINING' in yamldict.keys():
        lbfr_cnn_sampout = root.nametowidget(nbname+'.training.sampling.output')
        lbfr_cnn = root.nametowidget(nbname+'.training.cnn')
        lbfr_cnnstruc = root.nametowidget(nbname+'.training.cnn.structure')
        lbfr_cnnimgtr = root.nametowidget(nbname+'.training.cnn.image_training')
        lbfr_cnnepetr = root.nametowidget(nbname+'.training.cnn.epe_training')
        lbfr_cnninit = root.nametowidget(nbname+'.training.cnn.initialization')
        lbfr_cnnimgtr.grid_forget()
        lbfr_cnnepetr.grid_forget()
        lbfr_cnnstruc.grid_forget()
        lbfr_cnninit.grid_forget()
        lbfr_cnninit.grid(row=0, column=2, rowspan=2)
        lbfr_cnn_sampout.grid_forget()
        lbfr_cnn_sampout.grid(row=0, column=6, rowspan=2)
        root.nametowidget(nbname+'.training.cnn._skip_inputs_label').grid_forget()    
        root.nametowidget(nbname+'.training.cnn._skip_inputs').grid_forget()
    
        cnnstruc_btn = ttk.Button(lbfr_cnn, text="STRUCTURE")
        cnnstruc_btn.config(command=lambda: open_cnn_gui(update_util))
        cnnstruc_btn.grid(row=1, column=3, **_GRID_CFG1)
        
        # IMG
        cnnimgtr_btn = ttk.Button(lbfr_cnn, text="IMAGE TRAINING")
        cnnimgtr_btn.config(command=lambda: update_util.open_imgtr_gui())
        cnnimgtr_btn.grid(row=1, column=4, **_GRID_CFG1)
        
        # EPE
        cnnepetr_btn = ttk.Button(lbfr_cnn, text="EPE TRAINING")
        cnnepetr_btn.config(command=lambda: update_util.open_epetr_gui())
        cnnepetr_btn.grid(row=1, column=5, **_GRID_CFG1)
    
    notebook.pack()
    root.mainloop()

main()
