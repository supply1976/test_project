#
import os, sys
import numpy as np
import pandas as pd
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


def _not_imp():
    tkMB.showinfo("Orz", "Not Yet Implemented ...")
        
        
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
    use to load n-level nest-dict (from yaml.full_load())
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
                if yamlkey.endswith("|BATCH_SIZE"):
                    lb.config(bg='orange')
                if yamlkey.endswith("LEARNING_RATE|BASE_LR"):
                    lb.config(bg='orange')
                if yamlkey.endswith("HYPER_PARAMETERS|REG_L2"):
                    lb.config(bg='orange')
                if yamlkey.endswith("HYPER_PARAMETERS|REG_POWER"):
                    lb.config(bg='orange')
                if yamlkey.endswith("HYPER_PARAMETERS|REG_CURV_POWER"):
                    lb.config(bg='orange')
                if yamlkey.endswith("HYPER_PARAMETERS|SLOPE_ENHANCEMENT"):
                    lb.config(bg='orange')
                    
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
                self.entry_widgets.append((yamlkey, en))
                
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
        self.yamlkey_kerns = "TRAINING|CNN|STRUCTURE|KERNEL_SIZE"
        self.yamlkey_chans = "TRAINING|CNN|STRUCTURE|CHANNELS"
        self.yamlkey_actfs = "TRAINING|CNN|STRUCTURE|ACTIVATION_FUNCTIONS"
        self.yamlkey_symms = "TRAINING|CNN|STRUCTURE|SYMMETRY"
        self.yamlkey_skips = "TRAINING|CNN|_SKIP_INPUTS"
        _kern_dilat = "TRAINING|CNN|_KERNEL_DILATION"
        self.yamlkey_kdilat_en = _kern_dilat+"|ENABLE"
        self.yamlkey_kdilat_ra = _kern_dilat+"|DILATION_RATE"
        
        # tk variables containers
        self.skip_vars = []
        self.kern_vars = []
        self.chan_vars = []
        self.actf_vars = []
        self.symm_vars = []
        # tk variables for Checkbutton 
        self.tkvar_kernCrossAllCheck = tk.BooleanVar()
        self.tkvar_kernCrossAllCheck.set(False)
        self.tkvar_chanCrossAllCheck = tk.BooleanVar()
        self.tkvar_chanCrossAllCheck.set(False)
        self.tkvar_actfCrossAllCheck = tk.BooleanVar()
        self.tkvar_actfCrossAllCheck.set(False)
        self.tkvar_symmCrossAllCheck = tk.BooleanVar()
        self.tkvar_symmCrossAllCheck.set(False)


    def gui_start(self, parent):
        lbfr_0 = tk.LabelFrame(parent, text=" MICS ")
        lbfr_0.pack()
        # update struct btn
        self.update_struct_btn = tk.Button(lbfr_0, text="update structure")
        self.update_struct_btn.grid(
            row=0, column=0, rowspan=2, sticky='NS', **_GRID_CFG2)
        # 
        self.simple_net_btn = tk.Button(lbfr_0, text="simple")
        self.simple_net_btn.grid(row=0, column=1, sticky='NS', **_GRID_CFG2)
        # 
        self.oldBKM2_btn = tk.Button(lbfr_0, text="oldBKM2")
        self.oldBKM2_btn.grid(row=0, column=2, sticky='NS', **_GRID_CFG2)
        # 
        self.unet66_btn = tk.Button(lbfr_0, text="unet66")
        self.unet66_btn.grid(row=0, column=3, sticky='NS', **_GRID_CFG2)
        # 
        self.resnet13_btn = tk.Button(lbfr_0, text="resnet13 (yan)")
        self.resnet13_btn.grid(row=0, column=4, sticky='NS', **_GRID_CFG2)
        #
        self.p7x_btn = tk.Button(lbfr_0, text="P7X (TBD)")
        self.p7x_btn.grid(row=1, column=1, sticky='NS', **_GRID_CFG2)
        #
        self.etch_btn = tk.Button(lbfr_0, text="etch (TBD)")
        self.etch_btn.grid(row=1, column=2, sticky='NS', **_GRID_CFG2)
        #
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

        self.kern_cb = tk.Checkbutton(self.lbfr, text="cross all", 
            variable=self.tkvar_kernCrossAllCheck, onvalue=True, offvalue=False)
        self.chan_cb = tk.Checkbutton(self.lbfr, text="cross all", 
            variable=self.tkvar_chanCrossAllCheck, onvalue=True, offvalue=False)
        self.actf_cb = tk.Checkbutton(self.lbfr, text="cross all", 
            variable=self.tkvar_actfCrossAllCheck, onvalue=True, offvalue=False)
        self.symm_cb = tk.Checkbutton(self.lbfr, text="cross all", 
            variable=self.tkvar_symmCrossAllCheck, onvalue=True, offvalue=False)

        self.kern_cb.grid(row=1, column=2, **_GRID_CFG1)
        self.chan_cb.grid(row=1, column=3, **_GRID_CFG1)
        self.actf_cb.grid(row=1, column=4, **_GRID_CFG1)
        self.symm_cb.grid(row=1, column=5, **_GRID_CFG1)
  
        # load CNN structure configs from original yaml dict
        cnn_struct_d = self.cnndict['STRUCTURE']
        kerns = cnn_struct_d['KERNEL_SIZE']
        chans = cnn_struct_d['CHANNELS']
        actfs = cnn_struct_d['ACTIVATION_FUNCTIONS']
        symms = cnn_struct_d['SYMMETRY']
        skips = self.cnndict['_SKIP_INPUTS']
        
        for i, (sk, k, c, a, s) in enumerate(zip(skips, kerns, chans, actfs, symms)):
            self.add_layer(sk=sk, k=k, c=c, a=a, s=s)
        # button events
        self.update_struct_btn.config(command=self.update_btn_event_handle)
        self.add_layer_btn.config(command=lambda: self.add_layer())
        self.del_layer_btn.config(command=lambda: self.del_layer())
        self.simple_net_btn.config(command=self.load_simple_net)
        self.oldBKM2_btn.config(command=self.load_bkm2_struct)
        self.unet66_btn.config(command=self.load_unet66_struct)
        self.resnet13_btn.config(command=self.load_resnet13_struct)
        #self.draw_cnn = cnngraphv.MyDraw(parent, height=200, width=800)

   
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
        chans = [ 2, 2, 2, 3, 3, 4, 4, 4, 3, 3, 2, 1]
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
        tkvar_skip = tk.IntVar()      # not allow split
        #tkvar_kern = tk.IntVar() 
        #tkvar_chan = tk.IntVar()
        tkvar_kern = tk.StringVar()   # allow split 
        tkvar_chan = tk.StringVar()   # allow split
        tkvar_actf = tk.StringVar()   # allow split
        tkvar_symm = tk.StringVar()   # allow split
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
            row=self.layer_id+2, column=0, **_GRID_CFG1)
        wid_skip = ttk.Combobox(self.lbfr, width=14,
            values=list(range(-1, self.layer_id+1)), textvariable=tkvar_skip)
        wid_skip.grid(row=self.layer_id+2, column=1, **_GRID_CFG1)
        
        wid_kern = ttk.Combobox(self.lbfr, width=14,
            values=list(range(3, 43, 2)), textvariable=tkvar_kern)
        wid_kern.grid(row=self.layer_id+2, column=2, **_GRID_CFG1)
        
        wid_chan = ttk.Combobox(self.lbfr, width=14,
            values=list(range(1, 17)), textvariable=tkvar_chan)
        wid_chan.grid(row=self.layer_id+2, column=3, **_GRID_CFG1)
        
        wid_actf = ttk.Combobox(self.lbfr, width=14,
            values=['linear', 'sigmoid', 'relu'], textvariable=tkvar_actf)
        wid_actf.grid(row=self.layer_id+2, column=4, **_GRID_CFG1)
        
        wid_symm = ttk.Combobox(self.lbfr, width=14,
            values=['all', 'xy_mirror', 'off'], textvariable=tkvar_symm)
        wid_symm.grid(row=self.layer_id+2, column=5, **_GRID_CFG1)
        self.en_widgets.append([wid_skip, wid_kern, wid_chan, wid_actf, wid_symm])


    def update_btn_event_handle(self):
        self.get_cnn_struct_values()
    
    def struct_splits(self, input_list, combo_all=False):
        _combos = None
        output = [[_.strip() for _ in x.split(",") if len(_.strip())>0] for x in input_list]
        if combo_all:
            _combos = list(itertools.product(*output))
        else:
            max_splits = max([len(x) for x in output])
            output = [l*max_splits for l in output]
            output = [l[0:max_splits] for l in output[:]]
            _combos = list(zip(*output))
        return _combos


    def get_cnn_struct_values(self):
        self.struct_values_combo = []
        self.skips_value = [_.get() for _ in self.skip_vars]
        self.kerns_value = [_.get() for _ in self.kern_vars]
        self.chans_value = [_.get() for _ in self.chan_vars]
        self.actfs_value = [_.get() for _ in self.actf_vars]
        self.symms_value = [_.get() for _ in self.symm_vars]

        # handle on splits
        kern_combo_status = self.tkvar_kernCrossAllCheck.get()
        chan_combo_status = self.tkvar_chanCrossAllCheck.get()
        actf_combo_status = self.tkvar_actfCrossAllCheck.get()
        symm_combo_status = self.tkvar_symmCrossAllCheck.get()
        
        kerns_combo = self.struct_splits(self.kerns_value, combo_all=kern_combo_status)
        chans_combo = self.struct_splits(self.chans_value, combo_all=chan_combo_status)
        actfs_combo = self.struct_splits(self.actfs_value, combo_all=actf_combo_status)
        symms_combo = self.struct_splits(self.symms_value, combo_all=symm_combo_status)
        kerns_combo = [list(map(int , x)) for x in kerns_combo]
        chans_combo = [list(map(int , x)) for x in chans_combo]  

        all_nets_combo = list(itertools.product(
            [self.skips_value],
            kerns_combo,
            chans_combo,
            actfs_combo,
            symms_combo))
        #print(len(all_nets_combo))
        
        if len(all_nets_combo)==1:
            tk.StringVar(name=self.yamlkey_skips).set(self.skips_value)
            tk.StringVar(name=self.yamlkey_kerns).set(kerns_combo[0])
            tk.StringVar(name=self.yamlkey_chans).set(chans_combo[0])
            tk.StringVar(name=self.yamlkey_actfs).set(actfs_combo[0])
            tk.StringVar(name=self.yamlkey_symms).set(symms_combo[0])
            self.update_util.nets_combo = None
        else:
            self.update_util.nets_combo = np.array(all_nets_combo, dtype=object)
            df = self.update_util.print_nets_df(self.update_util.nets_combo)
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(df)


class UpdateUtil(object):
    def __init__(self, yamldict, all_entry_widgets, all_yamlkeys_to_tkvars):
        self.yamldict = yamldict
        self.all_entry_widgets = all_entry_widgets
        self.all_yamlkeys_to_tkvars = all_yamlkeys_to_tkvars
        self.saveto_fn = tk.StringVar()
        self.saveto_fn.set('test_out.yaml')
        self.nets_combo = None
        self.epetrhps_combos = None
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

    def print_nets_df(self, nets_combo):
        data=[]
        for i, t in enumerate(nets_combo):
            sid = str(i+1).zfill(4)
            df = pd.DataFrame(t, index=["SKIP", "KERN", "CHAN", "ACTF", "SYMM"])
            df = pd.concat([df], names=["ID"], keys=[sid])
            data.append(df)
        dfall = pd.concat(data)
        return dfall
    
    def print_epehps_df(self, epetrhps_combo):
        cols = ['BATCH_SIZE', 'BASE_LR', 'REG_L2', 
            'REG_POWER', 'REG_CURV_POWER', 'SLOPE_ENHANCEMENT']
        df = pd.DataFrame(epetrhps_combo, columns=cols)
        df['ID'] = [str(_).zfill(4) for _ in df.index.values+1]
        df.set_index("ID", inplace=True)
        return df
        
    def imgtr_hp_update(self):
        new_kv = [("TRAINING|CNN|IMAGE_TRAINING|"+k, v) for (k, v) in self.imgkv]
        for (k, v) in new_kv:
            self.all_yamlkeys_to_tkvars[k].set(v.get())
   
    def epetr_hps_update(self):
        d1 = self.newfr_epe_key2var 
        d2 = self.newfr_epe_key2ens
        key_bt = 'HYPER_PARAMETERS|BATCH_SIZE'
        key_l2 = 'HYPER_PARAMETERS|REG_L2'
        key_blr = 'HYPER_PARAMETERS|LEARNING_RATE|BASE_LR'
        key_pow = 'HYPER_PARAMETERS|REG_POWER'
        key_curv = 'HYPER_PARAMETERS|REG_CURV_POWER'
        key_slp = 'HYPER_PARAMETERS|SLOPE_ENHANCEMENT'
        orange_list = [key_bt, key_l2, key_pow, key_curv, key_slp, key_blr]
        for (k, v) in d1.items():
            name = "TRAINING|CNN|EPE_TRAINING|" + k        
            if k in orange_list:
                try:
                    v.get()
                    self.all_yamlkeys_to_tkvars[name].set(v.get())
                except:
                    continue
            else:
                self.all_yamlkeys_to_tkvars[name].set(v.get())
        # batch size
        _bt = d2[key_bt].get()
        _bt = list(map(int, _bt.split(",")))
        # reg L2
        _l2 = d2[key_l2].get()
        _l2 = list(map(float, _l2.split(",")))
        # base lr
        _blr = d2[key_blr].get()
        _blr = list(map(float, _blr.split(",")))
        # reg power
        _pow = d2[key_pow].get()
        _pow = list(map(float, _pow.split(",")))
        # reg curv power
        _curv = d2[key_curv].get()
        _curv = list(map(float, _curv.split(",")))
        # slope
        _slp = d2[key_slp].get()
        _slp = list(map(float, _slp.split(",")))
        _split_list = [_bt, _blr, _l2, _pow, _curv, _slp]
        self.epetrhps_combos = list(itertools.product(*_split_list))
        df = self.print_epehps_df(self.epetrhps_combos)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df)

             

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
        self.newfr_epe_key2var = dict(newfr.yamlkeys_to_tkvars)
        self.newfr_epe_key2ens = dict(newfr.entry_widgets)
        tk.Button(root, text="Update Setting", 
            command=self.epetr_hps_update).grid(columnspan=5)


def single_yaml_saveto(update_util):
    update_util.yaml_update()
    with open(update_util.saveto_fn.get(), 'w') as f:
        yaml.dump(update_util.yamldict, f, sort_keys=False)
        
        
def nets_combo_export(update_util):
    orig_stdout = sys.stdout
    save_dir = filedialog.askdirectory(initialdir='~/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        save_dir = os.path.abspath(save_dir)
    print("all yamls are saved in {}".format(save_dir))
    scripts = []
    assert update_util.nets_combo is not None
    update_util.yaml_update()
    fn, ext = os.path.splitext(update_util.saveto_fn.get())
    print(update_util.nets_combo.shape)
    df = update_util.print_nets_df(update_util.nets_combo)
    print("network_splits_table.txt is saved to {}".format(save_dir))
    with open(os.path.join(save_dir, 'network_splits_table.txt'), 'w') as f:
        sys.stdout = f
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df)
        sys.stdout = orig_stdout
        
    for i, t in enumerate(update_util.nets_combo):
        tid = str(i+1).zfill(4)
        new_fn = fn+"-struct-"+tid+ext
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
        with open(os.path.join(save_dir, new_fn), 'w') as f:
            yaml.dump(update_util.yamldict, f, sort_keys=False)
        scripts.append(" ".join(['adv_modeling', '-tr', new_fn, '--gpu', '\n']))
        scripts.append("sleep 30 ; rm -rf .proj ; sleep 10 \n")
    with open(os.path.join(save_dir, 'run_all_cnnNets_yamls.sh'), 'w') as f:
        f.writelines(scripts)
    os.chmod(os.path.join(save_dir, './run_all_cnnNets_yamls.sh'), 0o0777)
    

def epetrhps_combos_export(update_util):
    orig_stdout = sys.stdout
    save_dir = filedialog.askdirectory(initialdir='~/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        save_dir = os.path.abspath(save_dir)
    print("all EPE_TRAINING hyper-parameters split yamls are saved in {}".format(save_dir))
    scripts = []
    assert update_util.epetrhps_combos is not None
    update_util.yaml_update()
    fn, ext = os.path.splitext(update_util.saveto_fn.get())

    df = update_util.print_epehps_df(update_util.epetrhps_combos)
    print("epe_train_HPs_splits_table.txt is saved to {}".format(save_dir))
    with open(os.path.join(save_dir, 'epe_train_HPs_splits_table.txt'), 'w') as f:
        sys.stdout = f
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df)
        sys.stdout = orig_stdout
        
    pname = "TRAINING|CNN|EPE_TRAINING|HYPER_PARAMETERS|"
    for i, t in enumerate(update_util.epetrhps_combos):
        tid = str(i+1).zfill(4)
        new_fn = fn+"-epe_train_HPs-"+tid+ext
        
        tk.StringVar(name=pname+"BATCH_SIZE").set(t[0])
        tk.StringVar(name=pname+"|LEARINING_RATE|BASE_LR").set(t[1])
        tk.StringVar(name=pname+"REG_L2").set(t[2])
        tk.StringVar(name=pname+"REG_POWER").set(t[3])
        tk.StringVar(name=pname+"REG_CURV_POWER").set(t[4])
        tk.StringVar(name=pname+"SLOPE_ENHANCEMENT").set(t[5])
        update_util.yaml_update()
        with open(os.path.join(save_dir, new_fn), 'w') as f:
            yaml.dump(update_util.yamldict, f, sort_keys=False)
        scripts.append(" ".join(['adv_modeling', '-tr', new_fn, '--gpu', '\n']))
        scripts.append("sleep 30 ; rm -rf .proj ; sleep 10 \n")
    with open(os.path.join(save_dir, 'run_all_epetr_splits_yamls.sh'), 'w') as f:
        f.writelines(scripts)
    os.chmod(os.path.join(save_dir, './run_all_epetr_splits_yamls.sh'), 0o0777)

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

def ask_open_path(yamlkey):
    pathdir = filedialog.askdirectory(initialdir='~/')
    tk.StringVar(name=yamlkey).set(pathdir)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('yamlfile', type=str, nargs='?', default=None)
    FLAGS, _ = parser.parse_known_args()
    # root window settings    
    root = tk.Tk()
    root.title("Yaml Viewer/Editor/Generator")
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
        for (i, en) in app.entry_widgets:
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
    # net splits export
    tk.Button(topfr, text="network splits export all",
        command=lambda: nets_combo_export(update_util)).grid(
            row=0, column=2, **_GRID_CFG1) 
    # epe train HPs splits export
    tk.Button(topfr, text="EPE train HPs splits export all",
        command=lambda: epetrhps_combos_export(update_util)).grid(
            row=0, column=3, **_GRID_CFG1) 
    
    
    nbname=notebook._name
    # customize DATABASE tab GUI
    app_database = root.nametowidget(nbname+'.database')
    en_dbpath = root.nametowidget(nbname+'.database.path')
    en_dbpath.config(width=60)
    tk.Button(app_database, text="Browse", 
        command=lambda: ask_open_path("DATABASE|PATH")).grid(row=0, column=2, **_GRID_CFG1)
    
    tk.Button(app_database, text="get available names", 
        command=_not_imp).grid(row=1, column=2, **_GRID_CFG2)
   
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
