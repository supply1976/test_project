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
    #editMenu = tk.Menu(menuBar, tearoff=0)
    editMenu = tk.Menu(menuBar)
    editMenu.add_command(label="Cut")
    editMenu.add_command(label="Copy")
    editMenu.add_command(label="Paste")
    menuBar.add_cascade(label="File", menu=fileMenuItems)
    menuBar.add_cascade(label="Edit", menu=editMenu)
    root.config(menu=menuBar)
    


def main():
    root = tk.Tk()
    root.title("MLResist GUI Mockup")
    root.geometry("800x600")
    
    tab_parent = ttk.Notebook(root)
    tab1 = ttk.Frame(tab_parent)
    tab2 = ttk.Frame(tab_parent)
    tab3 = ttk.Frame(tab_parent)
    tab4 = ttk.Frame(tab_parent)
    tab5 = ttk.Frame(tab_parent)
    
    tab_parent.add(tab1, text="Data Preparation")
    tab_parent.add(tab2, text="Single ML Training")
    tab_parent.add(tab3, text="Model Export")    
    tab_parent.add(tab4, text="Advanced ML Training (HP Tuner)")
    tab_parent.add(tab5, text="Misc") 
    
    tab_parent.pack(expand=1, fill='both')
    
    labfr1a = tk.LabelFrame(tab1, text="Necessary Input")
    labfr1a.pack(fill='both')
    tk.Button(labfr1a, text="input model (.adml)").grid(row=0, column=0)
    tk.Entry(labfr1a).grid(row=0, column=1)
    tk.Button(labfr1a, text="test pattern (.oas, .gds)").grid(row=1, column=0)
    tk.Entry(labfr1a).grid(row=1, column=1)
    tk.Label(labfr1a, text="layer map (ex: myMain(0:0),myAF(1:0))").grid(row=2, column=0)
    tk.Entry(labfr1a).grid(row=2, column=1)
    tk.Button(labfr1a, text="save directory").grid(row=3, column=0)
    tk.Entry(labfr1a).grid(row=3, column=1)
    
    labfr1b = tk.LabelFrame(tab1, text="CD Gauge Data")
    labfr1b.pack(fill='both')
    tk.Button(labfr1b, text="gauge file (.asd)").grid(row=0, column=0)
    tk.Entry(labfr1b).grid(row=0, column=1)
    tk.Label(labfr1b, text="asd split ratio").grid(row=1, column=0)
    tk.Entry(labfr1b).grid(row=1, column=1)
    
    labfr1c = tk.LabelFrame(tab1, text="Contour Data (optional)")
    labfr1c.pack(fill='both')
    tk.Button(labfr1c, text="SEM_CONTOUR_DATA_DIR").grid(row=0, column=0)
    tk.Entry(labfr1c).grid(row=0, column=1)
    tk.Button(labfr1c, text="SEM_CONTOUR_DATA_LAYOUT_VALUES").grid(row=1, column=0)
    tk.Entry(labfr1c).grid(row=1, column=1)
    tk.Button(labfr1c, text="CONTOUR_GDS_RESAMPLING_PARAMS").grid(row=2, column=0)
    tk.Entry(labfr1c).grid(row=2, column=1)
    
    var = tk.IntVar()
    var.set(1)
    labfr1d = tk.LabelFrame(tab1, text="USE_DP")
    labfr1d.pack(fill='both')
    tk.Radiobutton(labfr1d, text="NONE", variable=var, value=1).grid(row=0, column=0)
    tk.Radiobutton(labfr1d, text="SGE", variable=var, value=2).grid(row=0, column=1)
    tk.Entry(labfr1d).grid(row=0, column=2)
    tk.Radiobutton(labfr1d, text="RSH", variable=var, value=3).grid(row=0, column=3)
    tk.Entry(labfr1d).grid(row=0, column=4)
    
    tk.Button(tab1, text="export dataprep config file").pack()
    tk.Button(tab1, text="Run").pack()
    
    labfr1e = tk.LabelFrame(tab1, text="Message")
    labfr1e.pack(fill='both')
    tk.Label(labfr1e, text="pre-calc initial model image for ML training later, barabara...").pack()
    ttk.Progressbar(labfr1e, value=50).pack()
    

    #menu_bar(root)    
    root.mainloop()
    
if __name__=="__main__":
    main()