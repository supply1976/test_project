import os, sys
#import tensorflow as tf
#from tensorflow import keras
import tkinter as tk
import numpy as np


class MyDraw(tk.Frame):
    def __init__(self, master, height, width):
        super(MyDraw, self).__init__(master)
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self, bg='white', width=width, height=height, 
            scrollregion=(0,0,height,width))
        self.scrollY = tk.Scrollbar(self, orient='vertical')
        self.scrollY.pack(side='right', fill='y')
        self.scrollY.config(command=self.canvas.yview)
        self.canvas.config(yscrollcommand=self.scrollY.set)
        
        self.canvas.pack()
        self.pack()
        self.colors = list('123456789abcde')

    def draw_multi_lines(self, x0, y0, x1, y1, width, out_dir):
        if width==1:
            self.canvas.create_line(x0, y0, x1, y1, width=1)
            return 0
        for i in range(-width+1, width+1, 2):
            c = "#"+"".join(np.random.choice(self.colors, 3))
            if out_dir=='e':
                self.canvas.create_line(x0, y0+i, x1, y1+i, width=2, fill=c)
            elif out_dir=='s':
                self.canvas.create_line(x0+i, y0, x1+i, y1, width=2, fill=c)
            elif out_dir=='n':
                self.canvas.create_line(x0+i, y0, x1+i, y1, width=2, fill=c)
            else:
                self.canvas.create_line(x0, y0, x1, y1)
            
    def draw_node(self, x0, y0, dr, text, out_len, out_width, out_dir):
        """
        """
        self.canvas.create_oval(x0-dr, y0-dr, x0+dr, y0+dr, fill='black')
        x1, y1 = (0,0)
        if out_dir=='e':
            x1, y1 = x0+dr+out_len, y0
            self.canvas.create_text(x0-15, y0-20, text=text)
        elif out_dir=='s':
            x1, y1 = x0, y0+dr+out_len
            self.canvas.create_text(x0, y0-15, text=text)
        elif out_dir=='n':
            x1, y1 = x0, y0-dr-out_len
            self.canvas.create_text(x0, y0+15, text=text)
        elif out_dir=='w':
            x1, y1 = x0-dr-out_len, y0
            self.canvas.create_text(x0+15, y0-20, text=text)
        else:
            pass
        self.draw_multi_lines(x0, y0, x1, y1, width=out_width, out_dir=out_dir)
        return (x1, y1)


    def draw_conv2d_block(self, x0, y0, size=50, text='Conv2D',
        actf=False, out_len=20, out_width=4, out_dir='s'):
        """
        ul: upper-left
        lr: lower-right
        """
        if out_dir=='e':
            ulx, uly = (x0, y0-size//2)
            lrx, lry = (x0+size, y0+size//2)
            x1, y1 = (x0+size, y0)
            x2, y2 = (x1+out_len, y1)
        elif out_dir=='s':
            ulx, uly = (x0-size//2, y0)
            lrx, lry = (x0+size//2, y0+size)
            x1, y1 = (x0, y0+size)
            x2, y2 = (x1, y1+out_len)
        elif out_dir=='n':
            ulx, uly = (x0-size//2, y0-size)
            lrx, lry = (x0+size//2, y0)
            x1, y1 = (x0, y0-size)
            x2, y2 = (x1, y1-out_len)
        elif out_dir=='s2e':
            ulx, uly = (x0-size//2, y0)
            lrx, lry = (x0+size//2, y0+size)
            x1, y1 = (x0+size//2, y0+size//2)
            x2, y2 = (x1+out_len, y1)
        else:
            ulx, uly = (x0, y0)
            lrx, lry = (x0, x0)
            x1, y1 = (x0, y0)
        xc, yc = (ulx+lrx)//2, (uly+lry)//2
        self.canvas.create_rectangle(ulx, uly, lrx, lry)
        self.canvas.create_text(xc, yc, text=text)
        if actf:
            self.canvas.create_line(
                xc-10, yc+20, xc, yc+20, xc+10, yc+10, width=1, fill='blue')
            #self.canvas.create_line(xc, yc+20, xc+10, yc+10, width=1, fill='blue')
        #self.canvas.create_line(x1, y1, x2, y2, width=out_width)
        self.draw_multi_lines(x1, y1, x2, y2, width=out_width, out_dir=out_dir)
        return (x2, y2)
    
    def draw_skip_arc(self, x0, y0, x1, y1, dr, st=180, ext=180, ifdash=False, out_dir='s'):
        if out_dir=='e':
            ulx, uly = x0-10, y0-dr
            lrx, lry = x1-10, y1+dr
            st, ext = st, ext
        elif out_dir=='s':
            ulx, uly = x0-dr, y0-10
            lrx, lry = x1+dr, y1-10
            st, ext = 90, 180
        elif out_dir=='n':
            ulx, uly = x1-dr, y1
            lrx, lry = x0+dr, y0
            st, ext = 270, 180
        else:
            pass
        if ifdash:
            self.canvas.create_arc(ulx, uly, lrx, lry, width=2, start=st, dash=(4,4), extent=ext, style=tk.ARC)
        else:
            self.canvas.create_arc(ulx, uly, lrx, lry, width=2, start=st, extent=ext, style=tk.ARC)
    
    def cnn(self, x0, y0, num_conv=6, skips=None, flowdir='e'):
        # input node
        x1, y1 = self.draw_node(x0, y0, dr=3, text="vt0 (input image)", 
            out_len=30, out_width=1, out_dir=flowdir)
        # conv2d layers
        coords=[(x1, y1)]
        for i in range(num_conv):
            xf, yf = self.draw_conv2d_block(*coords[-1], actf=True, out_dir=flowdir)
            coords.append((xf, yf))
        # draw skips
        if skips is not None:
            assert type(skips) is int
            for j in range(0, num_conv, skips):    
                self.draw_skip_arc(*coords[j], *coords[j+skips], dr=60, out_dir=flowdir)
        # final 1x1 layer linear sum
        x8, y8 = self.draw_conv2d_block(xf, yf, out_dir=flowdir, size=20, text='1x1', out_len=40, out_width=1)
        # output node
        flowdir = 'n' if flowdir=='s' else flowdir
        self.draw_node(x8, y8, dr=3, text="final output", out_len=10, out_width=1, out_dir='w')
        #mydraw.draw_skip_arc(x0, y0+15, x8, y8-15, dr=70, ifdash=True, out_dir='n')


    def cnnMLP(self, x0, y0, num_conv=6, skips=None, flowdir='e'):
        # input node
        x1, y1 = self.draw_node(x0, y0, dr=3, text="vt0 (input image)", 
            out_len=30, out_width=1, out_dir=flowdir)
        # conv2d layers
        coords=[(x1, y1)]
        for i in range(num_conv):
            xf, yf = self.draw_conv2d_block(*coords[-1], actf=True, out_dir=flowdir)
            coords.append((xf, yf))
        # draw skips
        if skips is not None:
            assert type(skips) is int
            for j in range(0, num_conv, skips):    
                self.draw_skip_arc(*coords[j], *coords[j+skips], dr=60, out_dir=flowdir)
        # create few 1x1 layers
        ann_x0, ann_y0 = (xf, yf)
        for _ in range(3):
            xf, yf = self.draw_conv2d_block(xf, yf, out_dir=flowdir, 
                size=20, text='1x1', out_len=30, out_width=4, actf=True)
        # final 1x1 layer linear sum
        xf, yf = self.draw_conv2d_block(xf, yf, out_dir=flowdir, size=20, text='1x1', out_len=40, out_width=1)      
        # output node
        flowdir = 'n' if flowdir=='s' else flowdir
        self.draw_node(xf, yf, dr=3, text="final output", out_len=0, out_width=1, out_dir=flowdir)
        mydraw.draw_skip_arc(ann_x0, ann_y0, xf, yf, dr=60, ifdash=True, out_dir='e')


    def unet_type1(self, x0, y0, num_conv=6, skips=None, flowdir='e'):
        # input node
        x1, y1 = self.draw_node(x0, y0, dr=3, text="vt0 (input image)", 
            out_len=30, out_width=1, out_dir=flowdir)
        # conv2d layers
        coords=[(x1, y1)]
        for i in range(num_conv):
            xf, yf = self.draw_conv2d_block(*coords[-1], actf=True, out_dir=flowdir)
            coords.append((xf, yf))
        # draw skips
        if skips is not None:
            assert type(skips) is int
            for j in range(0, num_conv, skips):    
                self.draw_skip_arc(*coords[j], *coords[j+skips], dr=60, out_dir=flowdir)
        # create equal number of 1x1 layers as num_conv
        for j in range(num_conv):
            xf, yf = self.draw_conv2d_block(xf, yf, out_dir=flowdir, size=20, text='1x1', out_len=30, out_width=1)
            coords.append((xf, yf))
        for j in range(1, num_conv):
            self.draw_skip_arc(*coords[j], *coords[2*num_conv-j], dr=100-j*10, st=0, ext=180, out_dir='e')
        #self.draw_skip_arc(*coords[2], *coords[10], dr=90, st=0, ext=180, out_dir='e')
        #self.draw_skip_arc(*coords[3], *coords[-4], dr=80, st=0, ext=180, out_dir='e')
        #self.draw_skip_arc(*coords[4], *coords[-5], dr=70, st=0, ext=180, out_dir='e')
        #self.draw_skip_arc(*coords[5], *coords[-6], dr=60, st=0, ext=180, out_dir='e')
        # output node
        flowdir = 'n' if flowdir=='s' else flowdir
        self.draw_node(xf+10, yf, dr=3, text="final output", out_len=10, out_width=1, out_dir='w')

    def unet_Vview(self, x0, y0, num_conv=6, skips=None, flowdir='s'):
        # input node
        x1, y1 = self.draw_node(x0, y0, dr=3, text="vt0 (input image)", 
            out_len=30, out_width=1, out_dir=flowdir)
        # conv2d layers
        coords=[(x1, y1)]
        for i in range(num_conv):
            xf, yf = self.draw_conv2d_block(*coords[-1], actf=False, out_dir=flowdir)
            if i==num_conv-1:
                self.draw_multi_lines(xf, yf, xf+100, yf, out_dir='e', width=3)
            else:
                self.draw_multi_lines(xf, yf-10, xf+100, yf-10, out_dir='e', width=3)
            coords.append((xf, yf))
        self.draw_multi_lines(xf+100, yf, xf+100, yf-38, width=3, out_dir='n')
        # draw skips
        if skips is not None:
            assert type(skips) is int
            for j in range(0, num_conv, skips):    
                self.draw_skip_arc(*coords[j], *coords[j+skips], dr=60, out_dir=flowdir)
        # create equal number of 1x1 layers as num_conv
        xf, yf = (xf+100, yf-40)
        for j in range(num_conv):
            if j==num_conv-1:
                w=1
            else:
                w=3
            xf, yf = self.draw_conv2d_block(xf, yf, out_dir='n', actf=True, size=20, text='1x1', out_len=50, out_width=w)
            coords.append((xf, yf))
        #for j in range(1, num_conv):
        #    self.draw_skip_arc(*coords[j], *coords[2*num_conv-j], dr=100-j*10, st=0, ext=180, out_dir='e')
        #self.draw_skip_arc(*coords[2], *coords[10], dr=90, st=0, ext=180, out_dir='e')
        #self.draw_skip_arc(*coords[3], *coords[-4], dr=80, st=0, ext=180, out_dir='e')
        #self.draw_skip_arc(*coords[4], *coords[-5], dr=70, st=0, ext=180, out_dir='e')
        #self.draw_skip_arc(*coords[5], *coords[-6], dr=60, st=0, ext=180, out_dir='e')
        # output node
        self.draw_node(xf, yf, dr=5, text="final output", out_len=10, out_width=1, out_dir='s')   

if __name__=="__main__":
    root = tk.Tk()
    root.title("")
    root.geometry('1600x900')
    
    mydraw = MyDraw(root, height=1600, width=1600)
    
    # simple cnn, no skip
    mydraw.canvas.create_text(40, 40, text="simple CNN, no skip",  anchor='nw', font=16)
    mydraw.cnn(x0=150, y0=100, skips=None, num_conv=6, flowdir='e')
    
    # sequential resNet, 2 or 3 conv2d form a resNetBlock
    mydraw.canvas.create_text(40, 180, 
        text="sequantial resNet CNN, 2 or 3 Conv2D() form a resNetBlock",  anchor='nw', font=16)
    mydraw.cnn(x0=150, y0=240, skips=2, num_conv=6, flowdir='e')
    mydraw.cnn(x0=800, y0=240, skips=2, num_conv=6, flowdir='e')
    mydraw.draw_skip_arc(903, 240, 1043, 240, dr=60, out_dir='e', st=0, ext=180)
    mydraw.draw_skip_arc(1043, 240, 1183, 240, dr=60, out_dir='e', st=0, ext=180)
    mydraw.draw_skip_arc(1183, 240, 1300, 240, dr=60, out_dir='e', st=0, ext=180)
    #mydraw.cnn(x0=800, y0=240, skips=3, num_conv=6, flowdir='e')
    
    # sequential resNet + ANN
    mydraw.canvas.create_text(40, 380, 
        text="sequential resNet CNN + 3-layer MLP (ANN)",  anchor='nw', font=16)
    mydraw.cnnMLP(x0=150, y0=450, skips=2, num_conv=6, flowdir='e')
    
    # UNet-like CNN
    mydraw.canvas.create_text(40, 600, 
        text="UNet-like CNN (adaptive sum of CNNs from deep to shallow)", anchor='nw', font=16) 
    mydraw.unet_type1(x0=100, y0=750, skips=2, num_conv=6, flowdir='e')
    mydraw.unet_Vview(x0=1100, y0=500, skips=2, num_conv=6, flowdir='s')
    #mydraw.cnn(x0=800, y0=50, skips=2, num_conv=6, flowdir='s')



    flowdir='s'


    """
    example 4: UNet
    x0, y0 = 800, 100
    x1, y1 = mydraw.draw_node(x0, y0, dr=3, text="vt0 (input image)",
        out_len=30, out_width=1, out_dir=flowdir)
    x2, y2 = mydraw.draw_conv2d_block(x1, y1, out_dir=flowdir)
    x3, y3 = mydraw.draw_conv2d_block(x2, y2, out_dir=flowdir)
    x4, y4 = mydraw.draw_conv2d_block(x3, y3, out_dir=flowdir)
    x5, y5 = mydraw.draw_conv2d_block(x4, y4, out_dir=flowdir)
    x6, y6 = mydraw.draw_conv2d_block(x5, y5, out_dir=flowdir)
    x7, y7 = mydraw.draw_conv2d_block(x6, y6, out_dir=flowdir)
    mydraw.draw_skip_arc(x1, y1-10, x3, y3-10, dr=60, out_dir=flowdir)
    mydraw.draw_skip_arc(x3, y3-10, x5, y5-10, dr=60, out_dir=flowdir)
    mydraw.draw_skip_arc(x5, y5-10, x7, y7-10, dr=60, out_dir=flowdir)
    mydraw.canvas.create_line(x7, y7, x7+100, y7, width=4)
    mydraw.canvas.create_line(x7+100, y7, x7+100, y7-40, width=4)
    x7, y7 = x7+100, y7-40
    x8, y8 = mydraw.draw_conv2d_block(x7, y7, out_dir='n', size=20, 
            text='1x1', out_len=50, out_width=1)
    xf, yf = x8, y8
    for _ in range(5):
        xf, yf = mydraw.draw_conv2d_block(xf, yf, out_dir='n', size=20, 
                text='1x1', out_len=50, out_width=1)
    mydraw.canvas.create_line(x2, y2-10, x2+98, y2-10, width=4, arrow=tk.LAST)
    mydraw.canvas.create_line(x3, y3-10, x3+98, y3-10, width=4, arrow=tk.LAST)
    mydraw.canvas.create_line(x4, y4-10, x4+98, y4-10, width=4, arrow=tk.LAST)
    mydraw.canvas.create_line(x5, y5-10, x5+98, y5-10, width=4, arrow=tk.LAST)
    mydraw.canvas.create_line(x6, y6-10, x6+98, y6-10, width=4, arrow=tk.LAST)
    mydraw.draw_node(xf, yf, out_dir='s', dr=3, text="cnn output", out_len=1, out_width=1)
    mydraw.canvas.create_text(x0, y0+500, text="adaptive sum of networks from deep to shallow")
    """
  
    mydraw.mainloop()

