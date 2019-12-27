#!/usr/bin/env python3
import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont


def openfile():
    print("File Open")


def menu_bar(root):
    menuBar = tk.Menu(root)
    fileMenuItems = tk.Menu(menuBar)
    fileMenuItems.add_command(label="Open", command=openfile)
    fileMenuItems.add_command(label="Save", command=openfile)
    fileMenuItems.add_command(label="Save As", command=openfile)
    fileMenuItems.add_command(label="Close", command=openfile)
    fileMenuItems.add_command(label="Quit", command=root.quit)    
    editMenu = tk.Menu(menuBar, tearoff=0)
    editMenu.add_command(label="Cut")
    editMenu.add_command(label="Copy")
    editMenu.add_command(label="Paste")
    helpMenu = tk.Menu(menuBar, tearoff=0)
    helpMenu.add_command(label="About")
    debugMenu = tk.Menu(menuBar)
    debugMenu.add_command(label="Debug Mode")
    menuBar.add_cascade(label="File", menu=fileMenuItems)
    menuBar.add_cascade(label="Edit", menu=editMenu)
    menuBar.add_cascade(label="Help", menu=helpMenu)
    menuBar.add_cascade(label="Debug", menu=debugMenu)
    root.config(menu=menuBar)


def _quick_start_tab(parent):
    font = ('14')
    tk.Button(parent, 
        text="Welcome to QUANTUM! \n(Qualified Ultra Accurate Network TUnable Model)", 
        font=font).pack(pady=50)
    # label frame for usecase control
    var_usecase = tk.IntVar()
    var_usecase.set(1)
    lf0a = tk.LabelFrame(parent, text="use case", font=font)
    lf0a.pack(padx=20, pady=20)
    text="I want to start from optical or simple resist model. Quickly get an"
    text+="\nResistML model with good accuracy, stability, and simulation TAT."
    tk.Radiobutton(lf0a, text=text, variable=var_usecase, value=1, font=font).pack(pady=10)
    
    text="I want to further improve my existing resist model using ResistML."
    text+="\nModel accuracy is first priority."
    text+="\n(more steps and hyper-parameters to tune)"
    tk.Radiobutton(lf0a, text=text, variable=var_usecase, value=2, font=font).pack(pady=10)

    tk.Radiobutton(lf0a, 
        text="I am a super user, give all parameters control to me.",
        variable=var_usecase, value=3, font=font).pack(pady=10)
    msgTest="interactive message here to explain more details for each use case..."
    tk.Message(parent, text=msgTest, width=300, font=font).pack(pady=10)
    tk.Button(parent, text="Next", font=font).pack(pady=10)


def _CD_data_option():
    font = tkfont.Font(underline=1)
    top = tk.Toplevel()
    lf1b = tk.LabelFrame(top, text="CD Data Options")
    lf1b.pack(padx=20, pady=5, fill='both')
    tk.Label(lf1b, text="ASD File", font=font).grid(row=0, column=0, sticky='w')
    tk.Entry(lf1b).grid(row=0, column=1, padx=5)
    tk.Button(lf1b, text="Browse").grid(row=0, column=2, padx=10, sticky='w')

    tk.Label(lf1b, text="Split Ratio", font=font).grid(row=1, column=0, sticky='w')
    tk.Entry(lf1b).grid(row=1, column=1, padx=5)
    tk.Label(lf1b, text="Validation data ratio over all asd gauges").grid(
        row=1, column=2, padx=10, sticky='w')

    tk.Label(lf1b, text="Class Column", font=font).grid(row=2, column=0, sticky='w')
    tk.Entry(lf1b).grid(row=2, column=1, padx=5)
    tk.Label(lf1b, text="ASD column name for 1D and 2D gauges, default=class").grid(
        row=2, column=2, padx=10, sticky='w')
    
def _contour_data_option():
    font = tkfont.Font(underline=1)
    top = tk.Toplevel()
    # label frame 1c for contour data support
    lf1c = tk.LabelFrame(top, text="Contour Data Options")
    lf1c.pack(padx=20, pady=5, fill='both')
    tk.Label(lf1c, text="Edge Point File", font=font).grid(row=0, column=0, sticky='w')
    tk.Entry(lf1c).grid(row=0, column=1, padx=5, sticky='w')
    tk.Button(lf1c, text="Browse").grid(row=0, column=2, padx=5, sticky='w')

    tk.Label(lf1c, text="Contour GDS Folder", font=font).grid(row=1, column=0, sticky='w')
    tk.Entry(lf1c).grid(row=1, column=1, padx=5, sticky='w')
    tk.Button(lf1c, text="Browse").grid(row=1, column=2, padx=5, sticky='w')

    tk.Label(lf1c, text="GDS Layer Values", font=font).grid(row=2, column=0, sticky='w')
    tk.Entry(lf1c).grid(row=2, column=1, padx=5, sticky='w')
    tk.Label(lf1c, text="integer array").grid(row=2, column=2, padx=10, sticky='w')
    
    tk.Label(lf1c, text="Contour Split Ratio", font=font).grid(row=3, column=0, sticky='w')
    tk.Entry(lf1c).grid(row=3, column=1, padx=5, sticky='w')
    tk.Label(lf1c, text="float, the ration of the validation data for contour GDS").grid(
        row=3, column=2, padx=10, sticky='w')

    tk.Label(lf1c, text="GDS Sampling", font=font).grid(row=4, column=0, sticky='w')
    tk.Entry(lf1c).grid(row=4, column=1, padx=5, sticky='w')
    

def _advanced_set():
    font = tkfont.Font(underline=1)
    # label frame 1e for advanced setting (should hide from common users)
    top = tk.Toplevel()
    lf1e = tk.LabelFrame(top, text="Advanced Settings", font="bold")
    lf1e.pack(padx=20, pady=5, fill='both')
    #lf1e.pack_forget()
    tk.Label(lf1e, text="SIM_MATRIX_SIZE", font=font).grid(row=0, column=0, sticky='w')
    tk.Entry(lf1e).grid(row=0, column=1)
    tk.Label(lf1e, text="default:1024x1024, has to be power of 2").grid(row=0, column=2, padx=20, sticky='w')
    tk.Label(lf1e, text="TRAINING_FIELD_SIZE", font=font).grid(row=1, column=0, sticky='w')
    tk.Entry(lf1e).grid(row=1, column=1)
    tk.Label(lf1e, text="default:61x61, must be odd number").grid(row=1, column=2, padx=20, sticky='w')
    tk.Label(lf1e, text="VALIDATION_FIELD_SIZE", font=font).grid(row=2, column=0, sticky='w')
    tk.Entry(lf1e).grid(row=2, column=1)
    tk.Label(lf1e, text="default:159x159, must be odd number").grid(row=2, column=2, padx=20, sticky='w')
    tk.Label(lf1e, text="JCL_KEYWORD_FILE", font=font).grid(row=3, column=0, sticky='w')
    tk.Entry(lf1e).grid(row=3, column=1)
    tk.Label(lf1e, text="").grid(row=3, column=2)



def _dataprep_tab(parent):
    font = tkfont.Font(underline=1)
    lf1a = tk.LabelFrame(parent, text="Input Data", font="bold")
    lf1a.pack(padx=20, pady=5, fill='both')
    #ltext="Model File (Choose your initial model for ResistML, amdl is supported only)"
    tk.Label(lf1a, text="Model File", font=font).grid(row=0, column=0)
    tk.Entry(lf1a, width=20).grid(row=0, column=1, padx=5)
    tk.Button(lf1a, text="Browse").grid(row=0, column=2, padx=5, sticky='w')
    tk.Label(lf1a, text="Choose your initial model for ResistML, amdl is supported only").grid(
        row=0,column=3, padx=10, sticky='w')
    tk.Checkbutton(lf1a, text="EUV model").grid(row=0, column=4)

    tk.Label(lf1a, text="Test Pattern", font=font).grid(row=1, column=0)
    tk.Entry(lf1a, width=20).grid(row=1, column=1, padx=5)
    tk.Button(lf1a, text="Browse").grid(row=1, column=2, padx=5, sticky='w')
    tk.Label(lf1a, text="Load test pattern (.oas, .gds), VRT not support").grid(
        row=1, column=3, padx=10, sticky='w')

    tk.Label(lf1a, text="Layer Map", font=font).grid(row=2, column=0)
    tk.Entry(lf1a, width=20).grid(row=2, column=1, padx=5)
    tk.Label(lf1a, text="Specify the layer/model map, ex: myMain(0:0),myAF(1:0),myAF2(2:0)").grid(
        row=2, column=3, padx=10, sticky='w')
   
    tk.Label(lf1a, text="CD Data", font=font).grid(row=3, column=0)
    tk.Entry(lf1a, width=20).grid(row=3, column=1, padx=5)
    tk.Button(lf1a, text="Setting", command=_CD_data_option).grid(row=3, column=2, padx=5, sticky='w')
    tk.Label(lf1a, text="ASD file contains print_basex, print_basey, print_headx, print_heady columns").grid(
        row=3, column=3, padx=10 , sticky='w')
    tk.Button(lf1a, text="?").grid(row=3, column=4)

    nonetext=tk.StringVar(parent, "NONE")
    tk.Label(lf1a, text="Contour Data", font=font).grid(row=4, column=0)
    tk.Entry(lf1a, width=20, textvariable=nonetext).grid(row=4, column=1, padx=5)
    tk.Button(lf1a, text="Setting", command=_contour_data_option).grid(row=4, column=2, padx=5, sticky='w')
    tk.Label(lf1a, text="Edge point data or contour gds/oas data").grid(row=4, column=3, padx=10, sticky='w')

    # label frame 1d for DP control
    var_dataprepDP = tk.IntVar()
    var_dataprepDP.set(1)
    lf1d = tk.LabelFrame(parent, text="DP Options", font="bold")
    lf1d.pack(padx=20, pady=5, fill='both')
    tk.Radiobutton(lf1d, text="NONE", variable=var_dataprepDP, value=1).grid(row=0, column=0, padx=5)
    tk.Radiobutton(lf1d, text="RSH", variable=var_dataprepDP, value=2).grid(row=0, column=1, padx=5)
    tk.Entry(lf1d).grid(row=0, column=2, padx=5)
    tk.Button(lf1d, text="Browse").grid(row=0, column=3, padx=5)
    tk.Radiobutton(lf1d, text="SGE", variable=var_dataprepDP, value=3).grid(row=0, column=4, padx=5)
    tk.Spinbox(lf1d, from_=1,to=100,increment=1).grid(row=0, column=5, padx=5)

    lf1b = tk.LabelFrame(parent, text="Output", font="bold")
    lf1b.pack(padx=20, pady=5, fill='both')
    tk.Label(lf1b, text="Save Directory", font=font).grid(row=0, column=0)
    tk.Entry(lf1b).grid(row=0, column=1, padx=5)
    tk.Button(lf1b, text="Browse").grid(row=0, column=2, padx=5)
    tk.Checkbutton(lf1b, text="Overwrite Data").grid(row=0, column=3, padx=10)

    fr1 = tk.Frame(parent)
    fr1.pack(pady=10)
    tk.Button(fr1, text="Advanced Settings", font="bold", command=_advanced_set).pack(padx=5, side='left')
    tk.Button(fr1, text="Load CFG", font="bold").pack(padx=5, side='left')
    tk.Button(fr1, text="SAVE CFG", font="bold").pack(padx=5, side='left')
    
    # label frame 1f for message, etc...
    lf1f = tk.LabelFrame(parent, text="Message, Status Bar, etc...", font="bold")
    lf1f.pack(padx=20, pady=5, fill='both')
    tk.Button(lf1f, text="RUN", font="bold").pack()
    tk.Label(lf1f, text="pre-calc initial model image for ML training later, barabara...", font=("bold")).pack()
    ttk.Progressbar(lf1f, value=50).pack()


def _optimizer_cfg():
    top = tk.Toplevel()
    lf2c = tk.LabelFrame(top, text="Optimizer Configs", font="bold")
    lf2c.grid(padx=20, pady=5, row=0,column=0, rowspan=2)
    tk.Label(lf2c, text="RANDOM_SEED").grid(row=0, column=0)
    tk.Entry(lf2c).grid(row=0, column=1)
    tk.Label(lf2c, text="MAX_ITER").grid(row=1, column=0)
    tk.Entry(lf2c).grid(row=1, column=1)
    tk.Label(lf2c, text="BATCH_SIZE").grid(row=2, column=0)
    tk.Entry(lf2c).grid(row=2, column=1)
    tk.Label(lf2c, text="SAVE_STEP").grid(row=3, column=0)
    tk.Entry(lf2c).grid(row=3, column=1)
    tk.Label(lf2c, text="SHOW_STEP").grid(row=4, column=0)
    tk.Entry(lf2c).grid(row=4, column=1)
    tk.Label(lf2c, text="INITIAL_LEARNING_RATE").grid(row=5, column=0)
    tk.Entry(lf2c).grid(row=5, column=1)
    tk.Label(lf2c, text="NUM_EPOCHS_PER_DECAY").grid(row=6, column=0)
    tk.Entry(lf2c).grid(row=6, column=1)
    tk.Label(lf2c, text="LEARNING_RATE_DECAY_FCATOR").grid(row=7, column=0)
    tk.Entry(lf2c).grid(row=7, column=1)
    tk.Label(lf2c, text="REGULARIZATION_WEIGHT").grid(row=8, column=0)
    tk.Entry(lf2c).grid(row=8, column=1)

def _network_cfg():
    top = tk.Toplevel()
    lf2d = tk.LabelFrame(top, text="ML Network Configs", font="bold")
    lf2d.grid(padx=20, pady=5, row=0, column=1)
    tk.Label(lf2d, text="MODEL_VERSION").grid(row=0, column=0)
    tk.Entry(lf2d).grid(row=0, column=1)
    tk.Label(lf2d, text="THRESH_ADJUST").grid(row=1, column=0)
    tk.Entry(lf2d).grid(row=1, column=1)
    tk.Label(lf2d, text="CHANNELS").grid(row=2, column=0)
    tk.Entry(lf2d).grid(row=2, column=1)
    tk.Label(lf2d, text="_LAYER_OP").grid(row=3, column=0)
    tk.Entry(lf2d).grid(row=3, column=1)


def _train_options():
    top = tk.Toplevel()
    font = tkfont.Font(underline=1)
    lfto = tk.LabelFrame(top, text="optional setting")
    lfto.pack(padx=20, pady=5, fill='both')
    tk.Label(lfto, text="Anchor Gauge Names", font=font).grid(row=0, column=0, padx=5)
    tk.Entry(lfto).grid(row=0, column=1, padx=5)
    tk.Label(lfto, text="(string) anchor gauge names separated by comma").grid(row=0, column=2, padx=10)
    #tk.Label(lfto, text="Anchor Gauge Cost Weight", font=font).grid(row=1, column=0, padx=5)
    #tk.Entry(lfto).grid(row=1, column=1, padx=5)
    tk.Label(lfto, text="Gauge Weight", font=font).grid(row=2, column=0, padx=5)
    tk.Entry(lfto).grid(row=2, column=1, padx=5)
    tk.Label(lfto, text="asd column name. use NONE if you want equal weight").grid(row=2, column=2, padx=10)
    tk.Label(lfto, text="Cost Weight EP", font=font).grid(row=3, column=0)
    tk.Entry(lfto).grid(row=3, column=1)
    tk.Label(lfto, text="(float) cost weight for edge point data").grid(row=3, column=2)
    tk.Label(lfto, text="Cost Weight CT", font=font).grid(row=4, column=0)
    tk.Entry(lfto).grid(row=4, column=1)
    tk.Label(lfto, text="(float) cost weight for contour gds data").grid(row=4, column=2)
    tk.Label(lfto, text="GPU index", font=font).grid(row=5, column=0)
    tk.Entry(lfto).grid(row=5, column=1)
    tk.Label(lfto, text="integer separated by comma (0,1,2,...), use NONE for CPU").grid(row=5, column=2)
    #tk.Label(lfto, text="Range Spec", font=font).grid(row=6, column=0)
    #tk.Entry(lfto).grid(row=6, column=1)
    #tk.Label(lfto, text="range spec").grid(row=6, column=2)



def _training_tab(parent):
    font = tkfont.Font(underline=1)
    lf2a = tk.LabelFrame(parent, text="Training", font="bold")
    lf2a.pack(padx=20, pady=10, fill='both')
    tk.Label(lf2a, text="Gauge CD Data (.npz)", font=font).grid(row=0, column=0)
    tk.Entry(lf2a, width=30).grid(row=0, column=1, padx=5)
    tk.Button(lf2a, text="Browse").grid(row=0, column=2, padx=5, sticky='w')
    # edge point 
    tk.Label(lf2a, text="Edge Point Data (.npz)", font=font).grid(row=1, column=0)
    tk.Entry(lf2a, width=30).grid(row=1, column=1, padx=5)
    tk.Button(lf2a, text="Browse").grid(row=1, column=2, padx=5, sticky='w')
    # contour gds
    tk.Label(lf2a, text="Contour GDS Data (.npz)", font=font).grid(row=2, column=0)
    tk.Entry(lf2a, width=30).grid(row=2, column=1, padx=5)
    tk.Button(lf2a, text="Browse").grid(row=2, column=2, padx=5, sticky='w')
    # optional parameters
    tk.Button(lf2a, text="Options", height=3,command=_train_options).grid(row=0, column=3, rowspan=3, padx=5)
    tk.Label(lf2a, text="Anchor, Weighting, GPU enable, etc ...").grid(row=1, column=4, padx=5)
    # save dir
    tk.Label(lf2a, text="Output Directory", font=font).grid(row=3, column=0, padx=5)
    tk.Entry(lf2a, width=30).grid(row=3, column=1, padx=5)
    tk.Button(lf2a, text="Browse").grid(row=3, column=2, padx=5, sticky='w')
    text="Job folder to save training results, records, and checkpoint files."
    text+="\nRestore the training from latest one if checkpoint file exist."
    tk.Label(lf2a, text=text).grid(row=3, column=3, columnspan=2)

    fr2 = tk.Frame(parent)
    fr2.pack()
    tk.Button(fr2, text="Optimizer Settings", font="bold", command=_optimizer_cfg).pack(
        padx=10, pady=10, side='left')
    tk.Button(fr2, text="Network Settings", font="bold", command=_network_cfg).pack(
        padx=10, pady=10, side='left')
    tk.Button(fr2, text="Load CFG", font="bold").pack(padx=10, pady=10, side='left')
    tk.Button(fr2, text="SAVE CFG", font="bold").pack(padx=10, pady=10, side='left')
    
    # label frame 2f for message, etc...
    lf2f = tk.LabelFrame(parent, text="Message, Status Bar, Interactive Plots, etc...", font="bold")
    lf2f.pack(padx=20, pady=5, fill='both')
    
    tk.Button(lf2f, text="RUN", font="bold").pack(padx=10, pady=5)
    tk.Label(lf2f, text="Launch ResistML training to get a Model!!", font="bold").pack()
    ttk.Progressbar(lf2f, value=30).pack()
    canvas1 = tk.Canvas(lf2f, width=300, height=200, bg='white')
    #canvas2 = tk.Canvas(lf2f, width=300, height=200, bg='yellow')
    canvas1.pack()
    #canvas2.pack(side='left')
    canvas1.create_line(50,180,250,180, arrow=tk.LAST)
    canvas1.create_line(50,180,50,20, arrow=tk.LAST)
    canvas1.create_text(150,100, text="Inline-inference result \n(RMS vs. steps),\n interactive plots")
    tk.Button(lf2f, text="Model Export (AMDL)", font="bold").pack(pady=10)


def _model_check_export_tab(parent):
    font = tkfont.Font(underline=1)
    lf3a = tk.LabelFrame(parent, text="Check Train", font="bold")
    lf3a.pack(padx=20, pady=10, fill='both')
    tk.Label(lf3a, text="Trained Job Folder", font=font).grid(row=0, column=0)
    #tk.Entry(lf3a, width=30).grid(row=0, column=1, padx=5)
    #tk.Button(lf2a, text="Browse").grid(row=0, column=2, padx=5, sticky='w')
    

def _misc_tab(parent):
    font = tkfont.Font(underline=1)
    lf5a = tk.LabelFrame(parent, text="Utilities", font="bold")
    lf5a.pack(padx=20, pady=10, fill='both')
    tk.Label(lf5a, text="put some utilities here", font="bold").pack()
    tk.Label(lf5a, 
        text="contour-check, file format check, sanity check, asd file preprocessing, etc...", font="bold").pack()
    
    

def main():
    root = tk.Tk()
    root.title("REML GUI Mockup:1024x768")
    root.geometry("1024x768")
    #root.option_add("*Font", "courier 10")
    
    tab_parent = ttk.Notebook(root)
    tab0 = ttk.Frame(tab_parent)
    tab1 = ttk.Frame(tab_parent)
    tab2 = ttk.Frame(tab_parent)
    tab3 = ttk.Frame(tab_parent)
    tab4 = ttk.Frame(tab_parent)
    tab5 = ttk.Frame(tab_parent)
    
    tab_parent.add(tab0, text="Quick Start")
    tab_parent.add(tab1, text="Data Preparation")
    tab_parent.add(tab2, text="Single ML Training")
    #tab_parent.add(tab3, text="Result Check and Model Export")    
    tab_parent.add(tab4, text="Advanced ML Training (HP Tuner)")
    tab_parent.add(tab5, text="Misc")
    tab_parent.pack(expand=1, fill='both')
   
    _quick_start_tab(tab0)
    _dataprep_tab(tab1)
    _training_tab(tab2)
    #_model_check_export_tab(tab3)
    _misc_tab(tab5)

    menu_bar(root)    
    root.mainloop()
    
if __name__=="__main__":
    main()
