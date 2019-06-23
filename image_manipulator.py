import numpy as np
import sys
from numpy import linalg as LA
from scipy import linalg as LA
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
from scipy import signal
import random


class Image(object):
    """
    Class for manipulating images
    """

    #def __init__(self):
    #    print("test")

    def load(self,filename_in=None,x_in=0,y_in=0,size_in=0):
        """
        Loads image from file. filename variable should be path to image file
        
        x,y and size variables select midpoint and size of square cropped image.
        if left blank, image is not cropped.
        
        However, many of the methods here will run slowly on larger images, so smaller cropped images
        would be advised
        
        """
        try:
            filename = sys.argv[1]
        except:
            filename = filename_in

        try:
            x = int(sys.argv[2])
            y = int(sys.argv[3])
            size = int(sys.argv[4])
        except:
            x = x_in
            y = y_in
            size = size_in
        size = size//2




        #f = open(filename,"r")
        #im = ndimage.imread(f)
        #f.close()
        with open(filename,'rb') as f:
            im = ndimage.imread(f)


        
        if ((x==0) or (y==0) or (size==0)):
            self.image = im
        else:
            r = im[(x-size):(x+size),(y-size):(y+size),0]
            g = im[(x-size):(x+size),(y-size):(y+size),1]
            b = im[(x-size):(x+size),(y-size):(y+size),2]
            im = np.vstack(([r],[g],[b]))
            self.image = np.moveaxis(im,0,-1)


    def show(self):
        #Displays image
        plt.imshow(self.image)
        plt.show()


    def save(self,filename=0):
        
        if str(raw_input("Save image? (y/n) "))=="y":


            if filename==0:
                filename = str(raw_input("Enter filename: ")) + ".jpg"

            misc.imsave(filename,self.image)

    def copy_to(self,other):
        r,g,b = self._split()
        other._comb(r,g,b)
    def copy_from(self,other):
        self.image = other.image[:,:,:]

#--- Helper methods to split image to r,g,b components and combine r,g,b components back to image
    def _split(self):
        #Returns seperate r,g,b matrices
        r = self.image[:,:,0]
        g = self.image[:,:,1]
        b = self.image[:,:,2]
        return r,g,b
    def _comb(self,r,g,b):
        #Combines seperate r,g,b matrics together to an image
        im = np.vstack(([r],[g],[b]))
        self.image = np.moveaxis(im,0,-1)
    def _fft(self):
        #Returns fourier transforms of r,g,b components
        r,g,b = self._split()
        fr = np.fft.fftshift(np.fft.fft2(self.image[:,:,0]))
        fg = np.fft.fftshift(np.fft.fft2(self.image[:,:,1]))
        fb = np.fft.fftshift(np.fft.fft2(self.image[:,:,2]))
        return fr,fg,fb
    def _ifft_comb(self,fr,fg,fb):
        #Performs inverse fourier transform, and combines r,g,b components into image

        r = np.fft.ifft2(np.fft.ifftshift(fr)).real.astype(int)
        g = np.fft.ifft2(np.fft.ifftshift(fg)).real.astype(int)
        b = np.fft.ifft2(np.fft.ifftshift(fb)).real.astype(int)
        self._comb(r,g,b)
#---------------------------------------------------------------------------------------------
    




#--- Fairly normal/standard image editing methods
    def smooth(self,it):
        #Convolution smoothing 
        r,g,b = self._split()
        k = np.ones((it,it))
        k = k/np.sum(k)
        
        r = ndimage.convolve(r,k)
        g = ndimage.convolve(g,k)
        b = ndimage.convolve(b,k)
        
        self._comb(r,g,b)
    def edge(self,mode=None):
        #Edge detection using convolution

        if mode == "h":
            e = np.array([[1.0,0,-1.0],
                          [20.0,0,-20.0],
                          [1.0,0,-1.0]])
        elif mode =="v":
            e = np.array([[1.0,20.0,1.0],
                          [0,0,0],
                          [-1.0,-20.0,-1.0]])
        else:
            e = np.array([[-3.0-3.0j,0-10.0j,3.0-3.0j],
                          [-10.0,0,10.0],
                          [-3.0+3.0j,0+10.0j,3.0+3.0j]])
        if np.sum(e)!=0:
            e = e/float(np.sum(e))
    
        r,g,b = self._split()
        r = signal.convolve2d(r,e,boundary="symm",mode="same")
        g = signal.convolve2d(g,e,boundary="symm",mode="same")
        b = signal.convolve2d(b,e,boundary="symm",mode="same")
        
        self._comb(r.real.astype(int)+r.imag.astype(int),g.real.astype(int)+g.imag.astype(int),b.real.astype(int)+b.imag.astype(int))
    def thresh(self,t):
        #If pixel data is below a threshold, set to 0

        r,g,b = self._split()
        f = np.vectorize(lambda x:0 if x<t else x)
        r = f(r) 
        g = f(g) 
        b = f(b) 
        self._comb(r,g,b)
    def key(self,other,t):
        #hard keying of 2 images
        assert self.image.shape==other.image.shape
        r1,g1,b1 = self._split()
        r2,g2,b2 = other._split()
        k = (np.sqrt(r1**2+g1**2+b1**2))>t
        #print(k)
        r = np.where(k,r1,r2)
        g = np.where(k,g1,g2)
        b = np.where(k,b1,b2)
        self._comb(r,g,b)
    def col_rot(self):

        #rotates all colour values

        r,g,b = self._split()
        self._comb(g,b,r)
    def col_swap(self):

        #swaps first 2 colour values

        r,g,b = self._split()
        self._comb(g,r,b)
    def col_inv(self):

        #Inverts colours

        r,g,b = self._split()
        r = 256-r
        g = 256-g
        b = 256-b
        self._comb(r,g,b)
    def norm(self):
        #Scales brightness of image until any pixel reaches 255
        #Might change colour
        r,g,b = self._split()
        r = r*256/np.max(r)
        g = g*256/np.max(g)
        b = b*256/np.max(b)
        self._comb(r.astype(int),g.astype(int),b.astype(int))
    def thresh_split(self,upper,lower,t):
        #Splits image into 2 - one with every value above threshold, one below
        r,g,b = self._split()
        zs = np.zeros(r.shape)
        ur = np.where(r>t,r,zs).astype(int)
        ug = np.where(g>t,g,zs).astype(int)
        ub = np.where(b>t,b,zs).astype(int)

        lr = np.where(r<t,r,zs).astype(int)
        lg = np.where(g<t,g,zs).astype(int)
        lb = np.where(b<t,b,zs).astype(int)

        upper._comb(ur,ug,ub)
        lower._comb(lr,lg,lb)
    def add(self,im1,im2):
        self.image = (im1.image+im2.image)
    def rotate(self,angle):
        #Rotate image
        self.image = ndimage.rotate(self.image,angle,reshape=False)
    def fold(self,amount):
        #Like increasing brightness, but loops round
        # i.e. once pixel r,g,b value reaches 256 it loops back to 0
        r,g,b = self._split()
        r = (r*amount).astype(int)%256
        g = (g*amount).astype(int)%256
        b = (b*amount).astype(int)%256
        self._comb(r,g,b)
    def bright(self,amount):
        #Increase brightness of image

        amount = float(amount)
        r,g,b = self._split()
        r = (r*amount).astype(int)
        g = (g*amount).astype(int)
        b = (b*amount).astype(int)

        full = np.full((r.shape),255)
        r = np.where(r>255,full,r)
        g = np.where(g>255,full,g)
        b = np.where(b>255,full,b)


        self._comb(r,g,b)
    def noise(self,amount):
        #Add noise to image
        r,g,b = self._split()
        r =(r+ np.random.randint(-amount,amount,size=r.shape))
        g =(g+ np.random.randint(-amount,amount,size=g.shape))
        b =(b+ np.random.randint(-amount,amount,size=b.shape))

        full = np.full((r.shape),255)
        r = np.where(r>255,full,r)
        g = np.where(g>255,full,g)
        b = np.where(b>255,full,b)

        zs = np.zeros((r.shape))
        r = np.where(r<0,zs,r)
        g = np.where(g<0,zs,g)
        b = np.where(b<0,zs,b)

        self._comb(r.astype(int),g.astype(int),b.astype(int))
    def contrast(self,amount):
        #Increase contrast of image
        r,g,b = self._split()
        r = np.where(r>128,r*amount,r)
        g = np.where(g>128,g*amount,g)
        b = np.where(b>128,b*amount,b)

        r = np.where(r<128,r/amount,r)
        g = np.where(g<128,g/amount,g)
        b = np.where(b<128,b/amount,b)
        self._comb(r.astype(int),g.astype(int),b.astype(int))
    
    def pixelate(self,amount):
        a = 1.0/float(amount)
        imzoom = ndimage.zoom(self.image,[a,a,1],order=0)
        #print(self.image.shape)
        #print(imzoom.shape)
        self.image = imzoom


    def flip(self,ax=1):
        r,g,b = self._split()

        r = np.flip(r,ax)
        g = np.flip(g,ax)
        b = np.flip(b,ax)
        self._comb(r,g,b)

    def transpose(self):
        r,g,b = self._split()
        r = r.T
        g = g.T
        b = b.T
        self._comb(r,g,b)

    def monochrome(self,col):
        #sets 2 colours to 0
        if col=="r":
            self.image[:,:,1]=0
            self.image[:,:,2]=0
        if col=="g":
            self.image[:,:,0]=0
            self.image[:,:,2]=0
        if col=="b":
            self.image[:,:,0]=0
            self.image[:,:,1]=0

#------------------------------------------------------





#--- Methods that involve manipulation of image data in fourier space
    def fft_smooth(self,size):
        #Convolution smoothing in fourier space
        r,g,b = self._fft()
        k = np.ones((size,size))
        #k = k/np.sum(k)
        r = ndimage.convolve((r.real),k)
        g = ndimage.convolve((g.real),k)
        b = ndimage.convolve((b.real),k)
        self._ifft_comb(r,g,b)
    def fft_combine(self,other):
        #Convolution of images in fourier space
        #Runs very slowly for large images
        r1,g1,b1 = self._fft()
        r2,g2,b2 = other._fft()

        
        r = signal.convolve2d(r1,r2,boundary="symm",mode="same")
        g = signal.convolve2d(g1,g2,boundary="symm",mode="same")
        b = signal.convolve2d(b1,b2,boundary="symm",mode="same")
        

        self._ifft_comb(r,g,b)
    def fft_rgb_partition(self,sorts,axes,k):
        #Partial sort of r,g,b channels in fourier space
        #k variable determines how sorted output is

        r,g,b = self._fft()
        if 'r' in sorts:
            r = np.partition(r,k[0],axis=axes[0])
        if 'g' in sorts:
            g = np.partition(g,k[1],axis=axes[1])
        if 'b' in sorts:
            b = np.partition(b,k[2],axis=axes[2])
        self._ifft_comb(r,g,b)
    def fft_offset(self,offsets,axes):
        #Offsets r,g,b channels in fourier space

        r,g,b = self._fft()
        r = np.roll(r,offsets[0],axis=axes[0])
        g = np.roll(g,offsets[1],axis=axes[1])
        b = np.roll(b,offsets[2],axis=axes[2])
        self._ifft_comb(r,g,b)
    def fft(self):
        #Fourier transform of image, but scaled so that it can be viewed
        fr,fg,fb = self._fft()

        r = np.log10(fr.real*0.001+1)
        g = np.log10(fg.real*0.001+1)
        b = np.log10(fb.real*0.001+1)
        self._comb(r,g,b) 
    def fft_filter(self,amount,mode="l"):
        fr,fg,fb = self._fft()
        
        #Filter data around center. Either lowpass or highpass
        
        zs = np.zeros(fr.shape)
        lx = fr.shape[0]
        ly = fr.shape[1]
        condition = np.mean((np.abs(np.mgrid[-lx/2:lx/2:1, -ly/2:ly/2:1])),axis=0)<amount
        
        if mode=="l":
            fr = np.where(condition,fr,zs)
            fg = np.where(condition,fg,zs)
            fb = np.where(condition,fb,zs)
        if mode=="h":
            fr = np.where(condition,zs,fr)
            fg = np.where(condition,zs,fg)
            fb = np.where(condition,zs,fb)
        if mode=="c":
            fr = np.where(condition,fr,-fr)
            fg = np.where(condition,fg,-fg)
            fb = np.where(condition,fb,-fb)
        self._ifft_comb(fr,fg,fb)
    def fft_perm_chunk(self,mode=0,n=1):

        #Swap random horizontal, vertical or rectangular chunks of image in fourier space

        r,g,b = self._fft()
        for its in range(n):
            s1h_max = random.randint(0,r.shape[0])
            s1h_min = random.randint(0,s1h_max)
            width = s1h_max-s1h_min
            s2h_max = random.randint(s1h_max,r.shape[0])
            s2h_min = s2h_max-width

            if mode!="v"and mode!="b":
                r_swap = r[s1h_min:s1h_max,:].copy()
                g_swap = g[s1h_min:s1h_max,:].copy()
                b_swap = b[s1h_min:s1h_max,:].copy()

                r[s1h_min:s1h_max,:] = r[s2h_min:s2h_max,:]#r_swap2[:]
                g[s1h_min:s1h_max,:] = g[s2h_min:s2h_max,:]#g_swap2[:]
                b[s1h_min:s1h_max,:] = b[s2h_min:s2h_max,:]#b_swap2[:]
                
                r[s2h_min:s2h_max,:] = r_swap
                g[s2h_min:s2h_max,:] = g_swap
                b[s2h_min:s2h_max,:] = b_swap
                
            s1v_max = random.randint(0,r.shape[1])
            s1v_min = random.randint(0,s1v_max)
            width = s1v_max-s1v_min
            s2v_max = random.randint(s1v_max,r.shape[1])
            s2v_min = s2v_max-width

            if mode!="h"and mode!="b":
                r_swap = r[:,s1v_min:s1v_max].copy()
                g_swap = g[:,s1v_min:s1v_max].copy()
                b_swap = b[:,s1v_min:s1v_max].copy()

                r[:,s1v_min:s1v_max] = r[:,s2v_min:s2v_max]
                g[:,s1v_min:s1v_max] = g[:,s2v_min:s2v_max]
                b[:,s1v_min:s1v_max] = b[:,s2v_min:s2v_max]

                r[:,s2v_min:s2v_max] = r_swap
                g[:,s2v_min:s2v_max] = g_swap
                b[:,s2v_min:s2v_max] = b_swap

            if mode=="b":
                r_swap = r[s1h_min:s1h_max,s1v_min:s1v_max].copy()
                g_swap = g[s1h_min:s1h_max,s1v_min:s1v_max].copy()
                b_swap = b[s1h_min:s1h_max,s1v_min:s1v_max].copy()

                r[s1h_min:s1h_max,s1v_min:s1v_max] = r[s2h_min:s2h_max,s2v_min:s2v_max]
                g[s1h_min:s1h_max,s1v_min:s1v_max] = g[s2h_min:s2h_max,s2v_min:s2v_max]
                b[s1h_min:s1h_max,s1v_min:s1v_max] = b[s2h_min:s2h_max,s2v_min:s2v_max]

                r[s2h_min:s2h_max,s2v_min:s2v_max] = r_swap
                g[s2h_min:s2h_max,s2v_min:s2v_max] = g_swap
                b[s2h_min:s2h_max,s2v_min:s2v_max] = b_swap


        self._ifft_comb(r,g,b)
#----------------------------------------------------------------------




#--- Methods that permute or shuffle images in some way
    def rgb_offset(self,offsets,axes):
        r,g,b = self._split()
        r = np.roll(r,offsets[0],axis=axes[0])
        g = np.roll(g,offsets[1],axis=axes[1])
        b = np.roll(b,offsets[2],axis=axes[2])
        self._comb(r,g,b)
    def rgb_sort(self,sorts,axes):
        #sort r,g,b channels along specified axes
        r,g,b = self._split()
        if 'r' in sorts:
            r = np.sort(r,axis=axes[0])
        if 'g' in sorts:
            g = np.sort(g,axis=axes[1])
        if 'b' in sorts:
            b = np.sort(b,axis=axes[2])
        self._comb(r,g,b)
    def rgb_partition(self,sorts,axes,k):
        #Partial sort of r,g,b channels
        #k variable determines how sorted output is

        r,g,b = self._split()
        if 'r' in sorts:
            r = np.partition(r,k,axis=axes[0])
        if 'g' in sorts:
            g = np.partition(g,k,axis=axes[1])
        if 'b' in sorts:
            b = np.partition(b,k,axis=axes[2])
        self._comb(r,g,b)
    def perm(self,mode=0):
        #Permute the rows and columns of pixel matrices
        r,g,b = self._split()
        i = np.identity(r.shape[0]).astype(int)
        rr = range(r.shape[0])
        np.random.shuffle(rr)
        ph = np.take(i, rr, axis=0)
        np.random.shuffle(rr)
        pv = np.take(i, rr, axis=0)
        if mode!="v":
            r = np.matmul(ph,r)
            g = np.matmul(ph,g)
            b = np.matmul(ph,b)
        if mode!="h":
            r = np.matmul(r,pv)
            g = np.matmul(g,pv)
            b = np.matmul(b,pv)


        self._comb(r,g,b)
    def perm_chunk(self,mode=0):

        #Swap random horizontal, vertical or rectangular chunks of image


        r,g,b = self._split()
        s1h_max = random.randint(0,r.shape[0])
        s1h_min = random.randint(0,s1h_max)
        width = s1h_max-s1h_min
        s2h_max = random.randint(s1h_max,r.shape[0])
        s2h_min = s2h_max-width

        if mode!="v"and mode!="b":
            r_swap = r[s1h_min:s1h_max,:].copy()
            g_swap = g[s1h_min:s1h_max,:].copy()
            b_swap = b[s1h_min:s1h_max,:].copy()

            r[s1h_min:s1h_max,:] = r[s2h_min:s2h_max,:]#r_swap2[:]
            g[s1h_min:s1h_max,:] = g[s2h_min:s2h_max,:]#g_swap2[:]
            b[s1h_min:s1h_max,:] = b[s2h_min:s2h_max,:]#b_swap2[:]
            
            r[s2h_min:s2h_max,:] = r_swap
            g[s2h_min:s2h_max,:] = g_swap
            b[s2h_min:s2h_max,:] = b_swap
            
        s1v_max = random.randint(0,r.shape[1])
        s1v_min = random.randint(0,s1v_max)
        width = s1v_max-s1v_min
        s2v_max = random.randint(s1v_max,r.shape[1])
        s2v_min = s2v_max-width

        if mode!="h"and mode!="b":
            r_swap = r[:,s1v_min:s1v_max].copy()
            g_swap = g[:,s1v_min:s1v_max].copy()
            b_swap = b[:,s1v_min:s1v_max].copy()

            r[:,s1v_min:s1v_max] = r[:,s2v_min:s2v_max]
            g[:,s1v_min:s1v_max] = g[:,s2v_min:s2v_max]
            b[:,s1v_min:s1v_max] = b[:,s2v_min:s2v_max]

            r[:,s2v_min:s2v_max] = r_swap
            g[:,s2v_min:s2v_max] = g_swap
            b[:,s2v_min:s2v_max] = b_swap

        if mode=="b":
            r_swap = r[s1h_min:s1h_max,s1v_min:s1v_max].copy()
            g_swap = g[s1h_min:s1h_max,s1v_min:s1v_max].copy()
            b_swap = b[s1h_min:s1h_max,s1v_min:s1v_max].copy()

            r[s1h_min:s1h_max,s1v_min:s1v_max] = r[s2h_min:s2h_max,s2v_min:s2v_max]
            g[s1h_min:s1h_max,s1v_min:s1v_max] = g[s2h_min:s2h_max,s2v_min:s2v_max]
            b[s1h_min:s1h_max,s1v_min:s1v_max] = b[s2h_min:s2h_max,s2v_min:s2v_max]

            r[s2h_min:s2h_max,s2v_min:s2v_max] = r_swap
            g[s2h_min:s2h_max,s2v_min:s2v_max] = g_swap
            b[s2h_min:s2h_max,s2v_min:s2v_max] = b_swap


        self._comb(r,g,b)   
    def interleave(self,offsets,mode=0):
        #Offsets odd and even rows or coloumns in opposite directions

        r,g,b = self._split()
        if mode!="v":
            r_even = np.roll(r[::2,:],offsets[0],axis=1)
            g_even = np.roll(g[::2,:],offsets[1],axis=1)
            b_even = np.roll(b[::2,:],offsets[2],axis=1)

            r_odd = np.roll(r[1::2,:],-offsets[0],axis=1)
            g_odd = np.roll(g[1::2,:],-offsets[1],axis=1)
            b_odd = np.roll(b[1::2,:],-offsets[2],axis=1)

            r[::2,:] = r_even
            g[::2,:] = g_even
            b[::2,:] = b_even

            r[1::2,:] = r_odd
            g[1::2,:] = g_odd
            b[1::2,:] = b_odd

        if mode!="h":
            r_even = np.roll(r[:,::2],offsets[0],axis=0)
            g_even = np.roll(g[:,::2],offsets[1],axis=0)
            b_even = np.roll(b[:,::2],offsets[2],axis=0)

            r_odd = np.roll(r[:,1::2],-offsets[0],axis=0)
            g_odd = np.roll(g[:,1::2],-offsets[1],axis=0)
            b_odd = np.roll(b[:,1::2],-offsets[2],axis=0)

            r[:,::2] = r_even
            g[:,::2] = g_even
            b[:,::2] = b_even

            r[:,1::2] = r_odd
            g[:,1::2] = g_odd
            b[:,1::2] = b_odd
        self._comb(r,g,b)
    def roll_chunk(self,amount,mode=0):
        #Rolls/offsets a horizontal or vertical chunk of an image
        r,g,b = self._split()
        



        if mode!="v":
            s1h_max = random.randint(0,r.shape[0])
            s1h_min = random.randint(0,s1h_max)
            r[s1h_min:s1h_max,:] = np.roll(r[s1h_min:s1h_max,:],amount,axis=1)
            g[s1h_min:s1h_max,:] = np.roll(g[s1h_min:s1h_max,:],amount,axis=1)
            b[s1h_min:s1h_max,:] = np.roll(b[s1h_min:s1h_max,:],amount,axis=1)
            
           

        if mode!="h":
            s1v_max = random.randint(0,r.shape[1])
            s1v_min = random.randint(0,s1v_max)
            r[:,s1v_min:s1v_max] = np.roll(r[:,s1v_min:s1v_max],amount,axis=0)
            g[:,s1v_min:s1v_max] = np.roll(g[:,s1v_min:s1v_max],amount,axis=0)
            b[:,s1v_min:s1v_max] = np.roll(b[:,s1v_min:s1v_max],amount,axis=0)

        self._comb(r,g,b)
    

#----------------------------------------------------------------

       
#--- Linear algebra methods -------------------------------------
    def eigen(self):
        #Finds eigenvectors of image matrix
        r,g,b = self._split()
        r = LA.eig(r)[1].real*10
        g = LA.eig(g)[1].real*10
        b = LA.eig(b)[1].real*10
        self._comb(r,g,b)
    def lu(self):
        #LU factorises pixel matrices
        r,g,b = self._split()
        r = LA.lu(r)[1]+LA.lu(r)[2]
        g = LA.lu(g)[1]+LA.lu(g)[2]
        b = LA.lu(b)[1]+LA.lu(b)[2]
        self._comb(r,g,b)
    def svd(self,mode="u"):
        r,g,b = self._split()
        u_r,s_r,v_r = LA.svd(r)
        u_g,s_g,v_g = LA.svd(g)
        u_b,s_b,v_b = LA.svd(b)

        s_r = np.diag(s_r)
        s_g = np.diag(s_g)
        s_b = np.diag(s_b)
        
        if mode=="u":

            r = np.absolute(np.dot(u_r,np.dot(s_r,u_r.T))).astype(int)
            g = np.absolute(np.dot(u_g,np.dot(s_g,u_g.T))).astype(int)
            b = np.absolute(np.dot(u_b,np.dot(s_b,u_b.T))).astype(int)

        if mode=="v":
        
            r = np.absolute(np.dot(v_r,np.dot(s_r,v_r.T))).astype(int)
            g = np.absolute(np.dot(v_g,np.dot(s_g,v_g.T))).astype(int)
            b = np.absolute(np.dot(v_b,np.dot(s_b,v_b.T))).astype(int)

        if mode=="c":
            r = np.absolute(np.dot(u_r,np.dot(s_b,v_g))).astype(int)
            g = np.absolute(np.dot(u_g,np.dot(s_r,v_b))).astype(int)
            b = np.absolute(np.dot(u_b,np.dot(s_g,v_r))).astype(int)


        self._comb(r,g,b)

