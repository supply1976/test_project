import tkinter as tk
from tkinter import ttk


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
    font = ('Helvetica', 16)
    tk.Button(parent, 
        text="Welcome to QUANTUM! \n(Qualified Ultra Accurate Network TUnable Model)",
        font=font).pack(pady=50)
    # label frame for usecase control
    var_usecase = tk.IntVar()
    var_usecase.set(1)

    lf0a = tk.LabelFrame(parent, text="use case", font=font)
    lf0a.pack(padx=20, pady=20)
    text="I want a fast and stable ML model with moderate accuracy."
    text+= "\n(simple network, get a model in 30 min)"
    tk.Radiobutton(lf0a, text=text, variable=var_usecase, value=1, font=font).pack(pady=10)
    
    text="I need a very accurate ML model with acceptable stability."
    text+="\n(more steps and hyper-parameters to tune)"
    tk.Radiobutton(lf0a, text=text, variable=var_usecase, value=2, font=font).pack(pady=10)

    tk.Radiobutton(lf0a, 
        text="I am a super user, give all parameters control to me.",
        variable=var_usecase, value=3, font=font).pack(pady=10)
    msgTest="message/text here to explain more for each use case..."
    tk.Message(parent, text=msgTest, width=250, font=font).pack(pady=10)
    tk.Button(parent, text="Next", font=font).pack(pady=10)


def _dataprep_gui(parent):
    lf1a = tk.LabelFrame(parent, text="Necessary Input")
    lf1a.pack(padx=20, pady=5, fill='both')
    tk.Button(lf1a, text="MODEL_FILE").grid(row=0, column=0, sticky='w')
    tk.Entry(lf1a).grid(row=0, column=1)
    tk.Label(lf1a, text="choose your initial model (.amdl) for ResistML").grid(
        row=0, column=2, padx=20, sticky='w')
    tk.Checkbutton(lf1a, text="EUV model").grid(row=0, column=3)

    tk.Button(lf1a, text="TEST_PATTERN").grid(row=1, column=0, sticky='w')
    tk.Entry(lf1a).grid(row=1, column=1)
    tk.Label(lf1a, text="load test pattern (.oas, .gds), VRT not support").grid(
        row=1, column=2, padx=20, sticky='w')

    tk.Button(lf1a, text="ASD_FILE").grid(row=2, column=0, sticky='w')
    tk.Entry(lf1a).grid(row=2, column=1)
    asdmsg="asd file must contains 4 new columns: print_basex, print_basey, print_headx, print_heady"
    tk.Label(lf1a, text=asdmsg).grid(row=2, column=2, padx=20, sticky='w')
    tk.Button(lf1a, text="help me").grid(row=2, column=3)
    
    tk.Label(lf1a, text="LAYER_MAP").grid(row=3, column=0, sticky='w')
    tk.Entry(lf1a).grid(row=3, column=1)
    tk.Label(lf1a, text="specify the layer/model map, ex: myMain(0:0),myAF(1:0),myAF2(2:0)").grid(
        row=3, column=2, padx=20, sticky='w')
    
    tk.Button(lf1a, text="WORKING_DIR").grid(row=4, column=0, sticky='w')
    tk.Entry(lf1a).grid(row=4, column=1)
    tk.Label(lf1a, text="choose or create a folder to save the training data images").grid(
        row=4, column=2, padx=20, sticky='w')

    lf1b = tk.LabelFrame(parent, text="CD Data Options")
    lf1b.pack(padx=20, pady=5, fill='both')
    tk.Label(lf1b, text="VALIDATION_ASD_SPLIT_RATIO").grid(row=0, column=0, sticky='w')
    tk.Entry(lf1b).grid(row=0, column=1)
    tk.Label(lf1b, text="(0~1), ratio to split all gauges into 2 sets randomly (validation / train)").grid(
        row=0, column=2, padx=20, sticky='w')

    tk.Label(lf1b, text="ASD_CLASSIFICATION_COL").grid(row=1, column=0, sticky='w')
    tk.Entry(lf1b).grid(row=1, column=1)
    tk.Label(lf1b, text="column name in .asd file to group the gauges, default=NONE").grid(
        row=1, column=2, padx=20, sticky='w')
    
    tk.Label(lf1b, text="ANCHOR_GAUGE_NAME").grid(row=2, column=0, sticky='w')
    tk.Entry(lf1b).grid(row=2, column=1)
    tk.Label(lf1b, text="gauge name for anchor, use NONE if you don't want to").grid(
        row=2, column=2, padx=20, sticky='w')

    # label frame 1c for contour data support
    lf1c = tk.LabelFrame(parent, text="Contour Data (optional)")
    lf1c.pack(padx=20, pady=5, fill='both')
    tk.Button(lf1c, text="SEM_CONTOUR_DATA_DIR").grid(row=0, column=0, sticky='w')
    tk.Entry(lf1c).grid(row=0, column=1)
    tk.Button(lf1c, text="?").grid(row=0, column=2)
    tk.Button(lf1c, text="SEM_CONTOUR_DATA_LAYOUT_VALUES").grid(row=1, column=0, sticky='w')
    tk.Entry(lf1c).grid(row=1, column=1)
    tk.Button(lf1c, text="?").grid(row=1, column=2)
    tk.Button(lf1c, text="CONTOUR_GDS_RESAMPLING_PARAMS").grid(row=2, column=0, sticky='w')
    tk.Entry(lf1c).grid(row=2, column=1)
    tk.Button(lf1c, text="?").grid(row=2, column=2)
    
    # label frame 1d for DP control
    var_dataprepDP = tk.IntVar()
    var_dataprepDP.set(1)
    lf1d = tk.LabelFrame(parent, text="USE_DP ?")
    lf1d.pack(padx=20, pady=5, fill='both')
    tk.Radiobutton(lf1d, text="NONE", variable=var_dataprepDP, value=1).grid(row=0, column=0)
    tk.Radiobutton(lf1d, text="SGE (SGE_NUM_WORKERS)", variable=var_dataprepDP, value=2).grid(row=0, column=1)
    tk.Spinbox(lf1d, from_=1,to=100,increment=1).grid(row=0, column=2)
    tk.Radiobutton(lf1d, text="RSH (RSH_WORKERS_HOST_FILE)", variable=var_dataprepDP, value=3).grid(row=0, column=3)
    tk.Entry(lf1d).grid(row=0, column=4)
   
    # label frame 1e for advanced setting (should hide from common users)
    lf1e = tk.LabelFrame(parent, text="Advanced Setting, should be hided by default and only for advanced users")
    lf1e.pack(padx=20, pady=5, fill='both')
    tk.Label(lf1e, text="SIM_MATRIX_SIZE").grid(row=0, column=0, sticky='w')
    tk.Entry(lf1e).grid(row=0, column=1)
    tk.Label(lf1e, text="default:1024x1024, has to be power of 2").grid(row=0, column=2, padx=20, sticky='w')
    tk.Label(lf1e, text="TRAINING_FIELD_SIZE").grid(row=1, column=0, sticky='w')
    tk.Entry(lf1e).grid(row=1, column=1)
    tk.Label(lf1e, text="default:61x61, must be odd number").grid(row=1, column=2, padx=20, sticky='w')
    tk.Label(lf1e, text="VALIDATION_FIELD_SIZE").grid(row=2, column=0, sticky='w')
    tk.Entry(lf1e).grid(row=2, column=1)
    tk.Label(lf1e, text="default:159x159, must be odd number").grid(row=2, column=2, padx=20, sticky='w')

    fr1 = tk.Frame(parent)
    fr1.pack()
    tk.Button(fr1, text="load default dataprep config").pack(side='left')
    tk.Button(fr1, text="export my current setting").pack(side='left')
    tk.Button(fr1, text="RUN").pack()
    
    # label frame 1f for message, etc...
    lf1f = tk.LabelFrame(parent, text="Message, Status Bar, etc...")
    lf1f.pack(padx=20, pady=5, fill='both')
    tk.Label(lf1f, text="pre-calc initial model image for ML training later, barabara...").pack()
    ttk.Progressbar(lf1f, value=50).pack()

def _train_gui(parent):
    lf2a = tk.LabelFrame(parent, text="Training/Validation with Gauge CD Data")
    lf2a.pack(padx=20, pady=5, fill='both')
    tk.Button(lf2a, text="CDML_DATA").grid(row=0, column=0, sticky='w')
    tk.Entry(lf2a).grid(row=0, column=1)
    tk.Label(lf2a, text="numpy file (.npz) generated from Data Preparation, used for ML training").grid(
        row=0, column=2, padx=20, sticky='w')
    tk.Button(lf2a, text="CDML_VALIDATION_DATA").grid(row=1, column=0, sticky='w')
    tk.Entry(lf2a).grid(row=1, column=1)
    tk.Label(lf2a, text="numpy file (.npz) generated from Data Preparation, used for ML in-line validation").grid(
        row=1, column=2, padx=20, sticky='w')
   
    lf2b = tk.LabelFrame(parent, text="Training/Validation with Contour Data (optional)")
    lf2b.pack(padx=20, pady=5, fill='both')
    tk.Button(lf2b, text="CTML_DATA").grid(row=0, column=0, sticky='w')
    tk.Entry(lf2b).grid(row=0, column=1)
    tk.Label(lf2b, text="numpy file (.npz) generated from Data Preparation, used for ML training").grid(
        row=0, column=2, padx=20, sticky='w')
    tk.Button(lf2b, text="CTML_VALIDATION_DATA").grid(row=1, column=0, sticky='w')
    tk.Entry(lf2b).grid(row=1, column=1)
    tk.Label(lf2b, text="numpy file (.npz) generated from Data Preparation, used for ML in-line validation").grid(
        row=1, column=2, padx=20, sticky='w')

    fr2 = tk.Frame(parent)
    fr2.pack()
    lf2c = tk.LabelFrame(fr2, text="Optimizer Configs")
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

    lf2d = tk.LabelFrame(fr2, text="ML Network Configs")
    lf2d.grid(padx=20, pady=5, row=0, column=1)
    tk.Label(lf2d, text="MODEL_VERSION").grid(row=0, column=0)
    tk.Entry(lf2d).grid(row=0, column=1)
    tk.Label(lf2d, text="THRESH_ADJUST").grid(row=1, column=0)
    tk.Entry(lf2d).grid(row=1, column=1)
    tk.Label(lf2d, text="CHANNELS").grid(row=2, column=0)
    tk.Entry(lf2d).grid(row=2, column=1)
    tk.Label(lf2d, text="_LAYER_OP").grid(row=3, column=0)
    tk.Entry(lf2d).grid(row=3, column=1)

    lf2e = tk.LabelFrame(fr2, text="Others")
    lf2e.grid(padx=20, pady=5, row=1, column=1)
    tk.Label(lf2e, text="GPU_INDICES").grid(row=0, column=0)
    tk.Entry(lf2e).grid(row=0, column=1)
    tk.Label(lf2e, text="WORKING_DIR").grid(row=1, column=0)
    tk.Entry(lf2e).grid(row=1, column=1)
    tk.Label(lf2e, text="RANGE_SPEC").grid(row=2, column=0)
    tk.Entry(lf2e).grid(row=2, column=1)

    fr2b = tk.Frame(parent)
    fr2b.pack()
    tk.Button(fr2b, text="load default training configs").pack(side='left')
    tk.Button(fr2b, text="export my current setting").pack(side='left')
    tk.Button(fr2b, text="RUN").pack()
    # label frame 2f for message, etc...
    lf2f = tk.LabelFrame(parent, text="Message, Status Bar, Interactive Plots, etc...")
    lf2f.pack(padx=20, pady=5, fill='both')
    tk.Label(lf2f, text="Launch ResistML training to get a Model!!").pack()
    ttk.Progressbar(lf2f, value=30).pack()
    canvas1 = tk.Canvas(lf2f, width=300, height=200, bg='white')
    #canvas2 = tk.Canvas(lf2f, width=300, height=200, bg='yellow')
    canvas1.pack()
    #canvas2.pack(side='left')
    canvas1.create_line(50,180,250,180, arrow=tk.LAST)
    canvas1.create_line(50,180,50,20, arrow=tk.LAST)
    canvas1.create_text(150,100, text="RMS vs. steps, interactive plots")




    

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
    tab_parent.add(tab3, text="Result Check and Model Export")    
    tab_parent.add(tab4, text="Advanced ML Training (HP Tuner)")
    tab_parent.add(tab5, text="Misc")
    tab_parent.pack(expand=1, fill='both')
   
    _quick_start_tab(tab0)
    _dataprep_gui(tab1)
    _train_gui(tab2)

    menu_bar(root)    
    root.mainloop()
    
if __name__=="__main__":
    main()
