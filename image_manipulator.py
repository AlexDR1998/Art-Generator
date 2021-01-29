import numpy as np
import sys
from numpy import linalg as LA
from scipy import linalg as LA
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
from scipy import signal
from scipy import stats
import random






class Image(object):
    """
    Class for manipulating images
    """

    #def __init__(self):
    #    print("test")

    def load(self,n=0,filename_in=None,x_in=0,y_in=0,size_in=0,size2=0):
        """
        Loads image from file. filename variable should be path to image file
        
        x,y and size variables select midpoint and size of square cropped image.
        if left blank, image is not cropped.
        
        However, many of the methods here will run slowly on larger images, so smaller cropped images
        would be advised
        
        """
        try:
            filename = sys.argv[n+1]
        except:
            filename = filename_in

        try:
            x = int(sys.argv[n+2])
            y = int(sys.argv[n+3])
            size = int(sys.argv[n+4])
            try:
                sizey = int(sys.argv[n+5])
            except:
                sizey=size
        except:
            x = x_in
            y = y_in
            size = size_in
            sizey=size2
        size = size//2
        sizey=sizey//2



        #f = open(filename,"r")
        #im = ndimage.imread(f)
        #f.close()
        with open(filename,'rb') as f:
            im = ndimage.imread(f)


        
        if ((x==0) or (y==0) or (size==0)):
            self.image = im
        else:
            r = im[(x-size):(x+size),(y-sizey):(y+sizey),0]
            g = im[(x-size):(x+size),(y-sizey):(y+sizey),1]
            b = im[(x-size):(x+size),(y-sizey):(y+sizey),2]
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

    def smooth(self,it=3):
        #Convolution smoothing 
        r,g,b = self._split()
        k = np.ones((it,it))
        k = k/np.sum(k)
        
        r = ndimage.convolve(r,k)
        g = ndimage.convolve(g,k)
        b = ndimage.convolve(b,k)
        
        self._comb(r,g,b)
    
    def sharpen(self):
        r,g,b = self._split()
        k = np.array([[0,-1,0],
                      [-1,5,-1],
                      [0,-1,0]])
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
    def thresh(self,t,mode=0):
        #If pixel data is below a threshold, set to 0
        if mode==0:
            r,g,b = self._split()
            f = np.vectorize(lambda x:0 if x<t else x)
            r = f(r) 
            g = f(g) 
            b = f(b) 
            self._comb(r,g,b)
        elif mode==1:
            #print(self.image.shape)
            for x in range(self.image.shape[0]):
                for y in range(self.image.shape[1]):
                    if sum(self.image[x,y,:])<3*t:
                        self.image[x,y,:]=0
        elif mode==2:
            r,g,b = self._split()
            f = np.vectorize(lambda x:0 if x>t else x)
            r = f(r) 
            g = f(g) 
            b = f(b) 
            self._comb(r,g,b)         
        else:
            for x in range(self.image.shape[0]):
                for y in range(self.image.shape[1]):
                    if sum(self.image[x,y,:])>3*t:
                        self.image[x,y,:]=0           

    def thresh_colour(self,col,dist=5,mode=0,col2=None):
        #keeps only colours that are close enough to specified colour
        if col2==None:
            col2=col
        if mode==0:
            for x in range(self.image.shape[0]):
                for y in range(self.image.shape[1]):
                    if sum((self.image[x,y,:]-col)**2)>dist**2:
                        self.image[x,y,:]=col2
        else :
            for x in range(self.image.shape[0]):
                for y in range(self.image.shape[1]):
                    if sum((self.image[x,y,:]-col)**2)<dist**2:
                        self.image[x,y,:]=col2

    def key(self,other,t,mode=0):
        #hard keying of 2 images
        assert self.image.shape==other.image.shape
        r1,g1,b1 = self._split()
        r2,g2,b2 = other._split()
        if mode==0:
            k = ((r2+g2+b2))>t*3
        else:
            k = ((r2+g2+b2))<t*3
        #print(k)
        r = np.where(k,r1,r2)
        g = np.where(k,g1,g2)
        b = np.where(k,b1,b2)
        self._comb(r,g,b)

    def key_col_2(self,other,t,col,mode=0):
        #Image 1 when image 2 is close enough to colour col
        assert self.image.shape==other.image.shape
        r1,g1,b1 = self._split()
        r2,g2,b2 = other._split()
        if mode==0:
            k = ((r2-col[0])**2+(g2-col[1])**2+(b2-col[2])**2)>t**2
        else:
            k = ((r2-col[0])**2+(g2-col[1])**2+(b2-col[2])**2)<t**2
        #print(k)
        #r = np.where(k,r1,col[0])#r2)
        #g = np.where(k,g1,col[1])#g2)
        #b = np.where(k,b1,col[2])#b2)
        r = np.where(k,r1,r2)
        g = np.where(k,g1,g2)
        b = np.where(k,b1,b2)
        self._comb(r,g,b)



    def key3(self,other,key,t,mode=0):
        #Use 3rd image (key) to decide whether to include from picture 1 or 2
        assert self.image.shape==other.image.shape
        r1,g1,b1 = self._split()
        r2,g2,b2 = other._split()
        rk,gk,bk = key._split()
        if mode==0:
            k = ((rk+gk+bk))>t*3
        else:
            k = ((rk+gk+bk))<t*3
        #print(k)
        r = np.where(k,r1,r2)
        g = np.where(k,g1,g2)
        b = np.where(k,b1,b2)
        self._comb(r,g,b)

    def key_black(self,other,t,mode=0):
        #Like key but with black
        assert self.image.shape==other.image.shape
        r1,g1,b1 = self._split()
        r2,g2,b2 = other._split()
        zs = np.zeros(r1.shape).astype(int)
        if mode==0:
            k = ((r2+g2+b2))>t*3
        else:
            k = ((r2+g2+b2))<t*3
            
        #print(k)
        r = np.where(k,r1,zs)
        g = np.where(k,g1,zs)
        b = np.where(k,b1,zs)
        self._comb(r,g,b)


    def key_colour(self,other,d):
        #only keeps parts where both images have similar enough colours
        assert self.image.shape==other.image.shape
        #for x in range(self.image.shape[0]):
        #    for y in range(self.image.shape[1]):
        #        if sum((self.image[x,y,:]-other.image[x,y,:])**2)>d**2:
        #            self.image[x,y,:]=0;


        r1,g1,b1 = self._split()
        r2,g2,b2 = other._split()
        k = ((r1-r2)**2+(g1-g2)**2+(b1-b2)**2)<d**2
        zs = np.zeros(r1.shape).astype(int)
        r = np.where(k,r1,zs)
        g = np.where(k,g1,zs)
        b = np.where(k,b1,zs)
        self._comb(r,g,b)

    def split_col(self,other):
        #Sets self to black and white, sets colour info to other
        for x in range(self.image.shape[0]):
            for y in range(self.image.shape[1]):
                c = np.min(self.image[x,y,:])
                for z in range(3):
                    other.image[x,y,z] = self.image[x,y,z]-c
                self.image[x,y,:]=c

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
    def thresh_split(self,upper,lower,t,mode=0,col=None):
        #Splits image into 2 - one with every value above threshold, one below
        r,g,b = self._split()
        w = self.image.shape[0]
        h = self.image.shape[1]
        lr = np.zeros((w,h)).astype(int)
        lg= np.zeros((w,h)).astype(int)
        lb= np.zeros((w,h)).astype(int)
        ur= np.zeros((w,h)).astype(int)
        ub= np.zeros((w,h)).astype(int)
        ug= np.zeros((w,h)).astype(int)
        if mode==0:
            zs = np.zeros(r.shape)
            ur = np.where(r>t,r,zs).astype(int)
            ug = np.where(g>t,g,zs).astype(int)
            ub = np.where(b>t,b,zs).astype(int)

            lr = np.where(r<t,r,zs).astype(int)
            lg = np.where(g<t,g,zs).astype(int)
            lb = np.where(b<t,b,zs).astype(int)

            upper._comb(ur,ug,ub)
            lower._comb(lr,lg,lb)
        elif mode==1:
            for x in range(self.image.shape[0]):
                for y in range(self.image.shape[1]):
                    if sum(self.image[x,y,:])<3*t:
                        lr[x,y] = r[x,y]
                        lg[x,y] = g[x,y]
                        lb[x,y] = b[x,y]
                    else:
                        ur[x,y]=r[x,y]
                        ug[x,y]=g[x,y]
                        ub[x,y]=b[x,y]
            upper._comb(ur,ug,ub)
            lower._comb(lr,lg,lb)
        else:
            for x in range(self.image.shape[0]):
                for y in range(self.image.shape[1]):
                    if sum((self.image[x,y]-col)**2)<3*t:
                        lr[x,y] = r[x,y]
                        lg[x,y] = g[x,y]
                        lb[x,y] = b[x,y]
                    else:
                        ur[x,y]=r[x,y]
                        ug[x,y]=g[x,y]
                        ub[x,y]=b[x,y]
            upper._comb(ur,ug,ub)
            lower._comb(lr,lg,lb)
    def mean(self,other):
        self.image = (self.image+other.image)/2
    
    def add(self,im1,im2):
        self.image = (im1.image+im2.image)    
    def mul(self,other):
        self.image = self.image*other.image


    def col_mat(self,mat):
        #Treats each pixel as 3d vector, multiplies it by matrix
        w = self.image.shape[0]
        h = self.image.shape[1]

        for x in range(w):
            for y in range(h):
                self.image[x,y,:] = np.matmul(mat,self.image[x,y,:])

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
        #Add noise to image
        """
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
        """
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



    def crop(self,l,s):
        data = np.copy(self.image)
        #print(l[0])
        self.image = data[l[0]:l[0]+s[0],l[1]:l[1]+s[1],:]
        

    def zoom(self,amount):
        imzoom = ndimage.zoom(self.image,[amount,amount,1])
        self.image = imzoom

    def flip(self,ax=[1,1,1],rgb="rgb"):
        r,g,b = self._split()
        if 'r' in rgb:
            r = np.flip(r,ax[0])
        if 'g' in rgb:
            g = np.flip(g,ax[1])
        if 'b' in rgb:
            b = np.flip(b,ax[2])
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
        
    def black_and_white(self,col):
        col = np.array(col)
        col = col/(np.sum(col).astype('float'))
        a = Image()
        self.copy_to(a)
        self.image[:,:,0] = np.sum(a.image[:,:]*col,axis=2)
        self.image[:,:,1] = np.sum(a.image[:,:]*col,axis=2)
        self.image[:,:,2] = np.sum(a.image[:,:]*col,axis=2)
        #self.norm()
        """
        if col=="r":
            self.image[:,:,1]=self.image[:,:,0]
            self.image[:,:,2]=self.image[:,:,0]
        if col=="g":
            self.image[:,:,0]=self.image[:,:,1]
            self.image[:,:,2]=self.image[:,:,1]
        if col=="b":
            self.image[:,:,0]=self.image[:,:,2]
            self.image[:,:,1]=self.image[:,:,2]
        """
    def mirror(self,axis=0,side=0,g=0):
        w = self.image.shape[0]
        h = self.image.shape[1]
        if axis==0:

            for x in range((w/2)):
                for y in range(h):
                    if side==0:
                        self.image[x,y,:]=self.image[(w-x-1+g)%w,y,:]
                    else:
                        self.image[w-x-1,y,:]=self.image[(x+g)%w,(y-g)%h,:]
        elif axis==1:
            for x in range(w):
                for y in range(h/2):
                    if side==0:
                        self.image[x,y,:]=self.image[x,(h-y-1+g)%h,:]
                    else:
                        self.image[x,h-y-1,:]=self.image[x,(y-g)%h,:]
        elif axis==2:
            for x in range(w):
                for y in range(h):
                    if side==0:            
                        self.image[x,y,:]=self.image[(y+g)%w,(x-g)%h,:]
                    else:
                        self.image[y%w,x%h,:]=self.image[(x+y*g)%w,y,:]
        elif axis==3:
            for x in range(w):
                for y in range(h):
                    if side==0:
                        self.image[w-x-1,h-y-1,:]=self.image[y%w,x%h,:]
                    else:
                        self.image[y%w,x%h,:]=self.image[w-x-1,h-y-1,:]
    def streak(self,mode="h",glitch=0):
        w = self.image.shape[0]
        h = self.image.shape[1]
        #for x in range(w):
        #    for y in range(h):
        xpos1 = random.randint(0,w)
        xpos2 = random.randint(0,xpos1)
        ypos1 = random.randint(0,h)
        ypos2 = random.randint(0,ypos1)
        #print(xpos1)
        #print(xpos2)
        if mode!="v":
            cols1 = self.image[xpos1,:,:]
            cols2 = self.image[xpos2,:,:]
            for x in range(xpos2,xpos1):
                a = (glitch+x-xpos2)/float(xpos1-xpos2)
                self.image[x%w,:,:] = a*cols1+(1-a)*cols2
            
        if mode!="h":
            cols1 = self.image[:,ypos1,:]
            cols2 = self.image[:,ypos2,:]
            for x in range(ypos2,ypos1):
                a = (glitch+x-ypos2)/float(ypos1-ypos2)
                self.image[:,x%h,:] = a*cols1+(1-a)*cols2
            

    def voronoi(self,n,p=2,g=0):
        w = self.image.shape[0]
        h = self.image.shape[1]
        xs = np.random.randint(0,w,n)
        ys = np.random.randint(0,h,n)

        #self.image[xs,ys,:]=255
        
        #print(xs)
        #print(ys)
        for x in range(w):
            for y in range(h):
                dx = abs(x-xs)
                dy = abs(y-ys)
                dist = np.sqrt(dx**p+dy**p)
                i = np.argmin(dist)
                self.image[x,y] = self.image[xs[(i+g)%n],ys[(i+g)%n]]

            #print(dx)
    def voronoi_bi(self,n,p=2,centre=[0.5,0.5],width=1,w2=[1,1]):
        w = self.image.shape[0]
        h = self.image.shape[1]
        xs_peak = np.random.normal(centre[0],w2[0],int(width*n))
        ys_peak = np.random.normal(centre[1],w2[1],int(width*n))
        xs_wide = np.random.randint(0,w,int((1-width)*n))
        ys_wide = np.random.randint(0,h,int((1-width)*n))

        xs_peak=(xs_peak.astype(int))%w
        ys_peak=(ys_peak.astype(int))%h

        xs = np.concatenate((xs_peak,xs_wide),axis=0)
        ys = np.concatenate((ys_peak,ys_wide),axis=0)
        #self.image[xs,ys,:]=255
        
        #print(xs)
        #print(ys)
        for x in range(w):
            for y in range(h):
                dx = abs(x-xs)
                dy = abs(y-ys)
                dist = np.sqrt(dx**p+dy**p)
                i = np.argmin(dist)
                self.image[x,y] = self.image[xs[i],ys[i]]


    def voronoi_monte_carlo(self,n,col,p=2,g=0):
        w = self.image.shape[0]
        h = self.image.shape[1]
        xs = np.array([])
        ys = np.array([])

        for x in range(w):
            for y in range(h):
                dist = np.sum(((self.image[x,y]-col)/10.0)**2)
                #print(dist)
                if np.random.rand()>dist:
                    xs = np.append(xs,x)
                    ys = np.append(ys,y)
        #print(xs.shape)
        if n<xs.shape[0]:
            xs = np.random.choice(xs,n,replace=False)
            ys = np.random.choice(ys,n,replace=False)
        for x in range(w):
            for y in range(h):
                dx = abs(x-xs)
                dy = abs(y-ys)
                d = np.sqrt(dx**p+dy**p)
                i = np.argpartition(d,g)[g]
                #print(int(xs[i]))
                self.image[x,y] = self.image[int(xs[(i)]),int(ys[(i)])]



    def voronoi_monte_carlo2(self,n,col,peak=5,p=2,g=0):
        w = self.image.shape[0]
        h = self.image.shape[1]
        temp = np.sqrt(np.sum((self.image[:,:]-col)**2,axis=2))
        probs = np.ravel(temp)
        probs = max(probs)-probs
        probs = probs**peak
        probs = probs/np.sum(probs)
        coords = np.random.choice(np.arange(probs.shape[0]),n,replace=False,p=probs)
        xs,ys = np.unravel_index(coords,temp.shape)
        for x in range(w):
            for y in range(h):
                dx = abs(x-xs)
                dy = abs(y-ys)
                d = np.sqrt(dx**p+dy**p)
                i = np.argpartition(d,g)[g]
                #print(int(xs[i]))
                self.image[x,y] = self.image[int(xs[(i)]),int(ys[(i)])]
        #print(data.shape)
        #plt.plot(data)
        #plt.show()
        #data = np.unravel_index(np.argmax(np.ravel(self.image)),self.image.shape)

    def voronoi_blur(self,n,p=2,k=2):
        w = self.image.shape[0]
        h = self.image.shape[1]
        xs = np.random.randint(0,w,n)
        ys = np.random.randint(0,h,n)
        #self.image[xs,ys,:]=255
        
        #print(xs)
        #print(ys)
        for x in range(w):
            for y in range(h):
                dx = abs(x-xs)
                dy = abs(y-ys)
                dist = np.sqrt(dx**p+dy**p)
                i = np.argpartition(dist,k)[1:k]
                #print(i.shape)
                #print(xs[i].shape)
                col = np.mean(self.image[xs[i],ys[i]],axis=0)

                self.image[x,y] = col


    def voronoi_block(self,n,g=0):
        p=10.0
        w = self.image.shape[0]
        h = self.image.shape[1]
        xs = np.random.randint(0,w,n)
        ys = np.random.randint(0,h,n)
        for x in range(w):
            for y in range(h):
                dx = abs(x-xs)
                dy = abs(y-ys)
                #dist = np.minimum(dx,dy)
                
                i_x = np.argpartition(dx,g)[g]
                i_y = np.argpartition(dy,g)[g]
                self.image[x,y] = self.image[xs[i_x],ys[i_y]]


    def conv_mat(self,mat=np.ones((3,3))):
        r,g,b = self._split()
        r = ndimage.convolve((r),mat)
        g = ndimage.convolve((g),mat)
        b = ndimage.convolve((b),mat)
        self._comb(r,g,b)


    def fade(self,other,location,sharpness,mode="w"):
        #Smoothly fade between 2 images
        w = self.image.shape[0]
        h = self.image.shape[1]

        if mode=="v":
            for x in range(w):
                a = (np.tanh((x-location)*sharpness)+1)/2.0
                self.image[x,:,:] = self.image[x,:,:]*a+other.image[x,:,:]*(1-a)
        if mode=="h":
            for y in range(h):
                a = (np.tanh((y-location)*sharpness)+1)/2.0
                self.image[:,y,:] = self.image[:,y,:]*a+other.image[:,y,:]*(1-a)
#------------------------------------------------------





#--- Methods that involve manipulation of image data in fourier space
    def fft_smooth(self,size,g=0):
        #Convolution smoothing in fourier space
        r,g,b = self._fft()
        k = np.ones((size,size))+g*np.random.random((size,size))
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
        xys = np.mgrid[-lx/2:lx/2:1, -ly/2:ly/2:1]
        filt = np.exp(-np.sqrt(xys[0]**2+xys[1]**2)/amount**2)
        
        if mode=="l":
            #fr = np.where(condition,fr,zs)
            #fg = np.where(condition,fg,zs)
            #fb = np.where(condition,fb,zs)
            fr = fr*filt
            fg = fg*filt
            fb = fb*filt
        if mode=="h":
            #fr = np.where(condition,zs,fr)
            #fg = np.where(condition,zs,fg)
            #fb = np.where(condition,zs,fb)
            #fr = fr - np.where(condition,fr,zs)
            #fg = fg - np.where(condition,fg,zs)
            #fb = fb - np.where(condition,fb,zs)
            fr = fr*(1-filt)
            fg = fg*(1-filt)
            fb = fb*(1-filt)
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
    def fft_conv_mat(self,mat=np.ones((3,3))):
        #Convolves image with arbitrary matrix
        r,g,b = self._fft()
        r = ndimage.convolve((r.real),mat)
        g = ndimage.convolve((g.real),mat)
        b = ndimage.convolve((b.real),mat)
        self._ifft_comb(r,g,b)

    def fft_symm(self,axis,side):
        #Performs a symmetry operation on the fft of image, then ifft
        r,g,b = self._fft()
        self._comb(r,g,b)
        self.mirror(axis,side)
        r,g,b = self._split()
        self._ifft_comb(r,g,b)
        #r = np.roll(r,10)
        #g = np.roll(g,10)
        #b = np.roll(b,10)
        #self._ifft_comb(r,g,b)
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
    
    def sort(self,sorts,axes):
        #keeps pixels intact but sorts 1-2 colours along 1-2 axes
        
        pixels=np.array(self.image,dtype=[('r',int),('g',int),('b',int)])
        
        pixels = np.sort(pixels,axis=axes,order=sorts)
        
        r = pixels["r"][:,:,0]
        print(r.shape)
        g = pixels["g"][:,:,1]
        b = pixels["b"][:,:,2]
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
    
    def copy_chunk(self,mode=0):

        r,g,b = self._split()
        s1h_max = random.randint(0,r.shape[0])
        s1h_min = random.randint(0,s1h_max)
        width = s1h_max-s1h_min
        s2h_max = random.randint(s1h_max,r.shape[0])
        s2h_min = s2h_max-width

        if mode!="v"and mode!="b":

            r[s1h_min:s1h_max,:] = r[s2h_min:s2h_max,:]#r_swap2[:]
            g[s1h_min:s1h_max,:] = g[s2h_min:s2h_max,:]#g_swap2[:]
            b[s1h_min:s1h_max,:] = b[s2h_min:s2h_max,:]#b_swap2[:]
            
            
        s1v_max = random.randint(0,r.shape[1])
        s1v_min = random.randint(0,s1v_max)
        width = s1v_max-s1v_min
        s2v_max = random.randint(s1v_max,r.shape[1])
        s2v_min = s2v_max-width

        if mode!="h"and mode!="b":

            r[:,s1v_min:s1v_max] = r[:,s2v_min:s2v_max]
            g[:,s1v_min:s1v_max] = g[:,s2v_min:s2v_max]
            b[:,s1v_min:s1v_max] = b[:,s2v_min:s2v_max]

        if mode=="b":
            r[s1h_min:s1h_max,s1v_min:s1v_max] = r[s2h_min:s2h_max,s2v_min:s2v_max]
            g[s1h_min:s1h_max,s1v_min:s1v_max] = g[s2h_min:s2h_max,s2v_min:s2v_max]
            b[s1h_min:s1h_max,s1v_min:s1v_max] = b[s2h_min:s2h_max,s2v_min:s2v_max]


        self._comb(r,g,b)   


    def sum_chunk(self,mode=0):
        r,g,b = self._split()
        s1h_max = random.randint(0,r.shape[0])
        s1h_min = random.randint(0,s1h_max)
        width = s1h_max-s1h_min
        s2h_max = random.randint(s1h_max,r.shape[0])
        s2h_min = s2h_max-width
        s=1
        if mode!="v"and mode!="b":

            r[s1h_min:s1h_max,:] = (r[s1h_min:s1h_max,:] + r[s2h_min:s2h_max,:])/s#r_swap2[:]
            g[s1h_min:s1h_max,:] = (g[s1h_min:s1h_max,:] + g[s2h_min:s2h_max,:])/s#g_swap2[:]
            b[s1h_min:s1h_max,:] = (b[s1h_min:s1h_max,:] + b[s2h_min:s2h_max,:])/s#b_swap2[:]
            
            
        s1v_max = random.randint(0,r.shape[1])
        s1v_min = random.randint(0,s1v_max)
        width = s1v_max-s1v_min
        s2v_max = random.randint(s1v_max,r.shape[1])
        s2v_min = s2v_max-width

        if mode!="h"and mode!="b":

            r[:,s1v_min:s1v_max] = (r[:,s1v_min:s1v_max] + r[:,s2v_min:s2v_max])/s
            g[:,s1v_min:s1v_max] = (g[:,s1v_min:s1v_max] + g[:,s2v_min:s2v_max])/s
            b[:,s1v_min:s1v_max] = (b[:,s1v_min:s1v_max] + b[:,s2v_min:s2v_max])/s

        if mode=="b":
            r[s1h_min:s1h_max,s1v_min:s1v_max] = (r[s1h_min:s1h_max,s1v_min:s1v_max] + r[s2h_min:s2h_max,s2v_min:s2v_max])/s
            g[s1h_min:s1h_max,s1v_min:s1v_max] = (g[s1h_min:s1h_max,s1v_min:s1v_max] + g[s2h_min:s2h_max,s2v_min:s2v_max])/s
            b[s1h_min:s1h_max,s1v_min:s1v_max] = (b[s1h_min:s1h_max,s1v_min:s1v_max] + b[s2h_min:s2h_max,s2v_min:s2v_max])/s


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
    
    def flip_chunk(self,mode=0):

        xpos = random.randint(0,self.image.shape[0])
        ypos = random.randint(0,self.image.shape[1])
        
        xpos2 = random.randint(xpos,self.image.shape[0])
        ypos2 = random.randint(ypos,self.image.shape[1])
        #print(xpos)
        #print(ypos)
        self.image[xpos:xpos2,ypos:ypos2,:] = np.flip(self.image[xpos:xpos2,ypos:ypos2,:],mode)

    def rot_chunk(self,mode=0):
        xpos = random.randint(0,self.image.shape[0])
        ypos = random.randint(0,self.image.shape[1])
        s1 = random.randint(0,self.image.shape[0]-xpos)
        s2 = random.randint(0,self.image.shape[1]-ypos)
        size = min(s1,s2)
        for x in range(size):
            for y in range(size):
                self.image[xpos+x,ypos+y,:] = self.image[xpos-x,ypos+x,:]


    def modulo(self,mod_size,offset):
        temp = np.zeros(self.image.shape)
        for x in range(self.image.shape[0]):
            for y in range(self.image.shape[1]):
                temp[x,y,:]=self.image[(x)%mod_size[0]+offset[0],
                                       (y)%mod_size[1]+offset[1],
                                        :]


        self.image[:] = temp[:]   

                #self.image[x,y,:]=self.image[((x)%(y+1+offset[0])+offset[2])%self.image.shape[0],
                #                             ((y)%(x+1+offset[1])+offset[3])%self.image.shape[1],
                #                             :]
    def mod_kal(self,mod_size,offset,mode=0):
        temp = np.zeros(self.image.shape)
        
        if mode==0:
            for x in range(self.image.shape[0]):
                for y in range(self.image.shape[1]):
                    
                    if (x%(2*mod_size[0])>=mod_size[0]) and (y%(2*mod_size[1])>=mod_size[1]):
                        temp[x,y,:]=self.image[(mod_size[0]-x-1)%mod_size[0]+offset[0],
                                               (mod_size[1]-y-1)%mod_size[1]+offset[1],
                                               :]
                    elif (x%(2*mod_size[0])>=mod_size[0]):
                        temp[x,y,:]=self.image[(mod_size[0]-x-1)%mod_size[0]+offset[0],
                                               (y)%mod_size[1]+offset[1],
                                               :]
                    elif (y%(2*mod_size[1])>=mod_size[1]):
                        temp[x,y,:]=self.image[(x)%mod_size[0]+offset[0],
                                               (mod_size[1]-y-1)%mod_size[1]+offset[1],
                                               :]
                    else:
                        temp[x,y,:]=self.image[(x)%mod_size[0]+offset[0],
                                               (y)%mod_size[1]+offset[1],
                                                :]
                                                
        else:
            for x in range(self.image.shape[0]):
                for y in range(self.image.shape[1]):
                    
                    if (x%(2*mod_size[0])>=mod_size[0]) and (y%(2*mod_size[1])>=mod_size[1]):
                        if x%mod_size[0]>y%mod_size[1]:
                            temp[x,y,:]=self.image[(mod_size[0]-y-1)%mod_size[0]+offset[0],
                                                   (mod_size[1]-x-1)%mod_size[1]+offset[1],
                                                   :]
                        else:
                            temp[x,y,:]=self.image[(mod_size[0]-x-1)%mod_size[0]+offset[0],
                                                   (mod_size[1]-y-1)%mod_size[1]+offset[1],
                                                   :]
                    elif (x%(2*mod_size[0])>=mod_size[0]):

                        if x%mod_size[0]<y%mod_size[1]:

                            temp[x,y,:]=self.image[(mod_size[0]-y-1)%mod_size[0]+offset[0],
                                                   (x)%mod_size[1]+offset[1],
                                                   :]
                        else:
                            temp[x,y,:]=self.image[(mod_size[0]-x-1)%mod_size[0]+offset[0],
                                                   (y)%mod_size[1]+offset[1],
                                                   :]
                    elif (y%(2*mod_size[1])>=mod_size[1]):
                        if x%mod_size[0]>y%mod_size[1]:

                            temp[x,y,:]=self.image[(y)%mod_size[0]+offset[0],
                                                   (mod_size[1]-x)%mod_size[1]+offset[1],
                                                   :]
                        else:
                            temp[x,y,:]=self.image[(x)%mod_size[0]+offset[0],
                                                   (mod_size[1]-y)%mod_size[1]+offset[1],
                                                   :]
                    else:
                        if x%mod_size[0]>y%mod_size[1]:
                            temp[x,y,:]=self.image[(y)%mod_size[0]+offset[0],
                                                   (x)%mod_size[1]+offset[1],
                                                    :]
                        else:
                            temp[x,y,:]=self.image[(x)%mod_size[0]+offset[0],
                                                   (y)%mod_size[1]+offset[1],
                                                    :]

        self.image[:] = temp[:]   



    def polynom(self,coef,offset):
        temp = np.zeros(self.image.shape)
        xsize=self.image.shape[0]
        ysize=self.image.shape[1]

        for x in range(xsize):
            for y in range(ysize):
                temp[x,y,:]=self.image[int(coef[0]+(x+offset[0][0])*coef[1]+(x+offset[1][0])**2*coef[2]+(x+offset[2][0])**3*coef[3])%xsize,
                                       int(coef[0]+(y+offset[0][1])*coef[1]+(y+offset[1][1])**2*coef[2]+(y+offset[2][1])**3*coef[3])%ysize,:]
        self.image[:] = temp[:]


    def kron(self,other):
        #Kronecker product of 2 images - best to use low res images as resolution of result will
        #be multiples
        r1,g1,b1 = self._split()
        r2,g2,b2 = other._split()
        r = np.kron(r1,r2/255.0).astype("int")
        g = np.kron(g1,g2/255.0).astype("int")
        b = np.kron(b1,b2/255.0).astype("int")
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
    def svd(self,mode="u",alpha = 0.5):
        r,g,b = self._split()
        u_r,s_r,v_r = LA.svd(r,full_matrices=False)
        u_g,s_g,v_g = LA.svd(g,full_matrices=False)
        u_b,s_b,v_b = LA.svd(b,full_matrices=False)


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

        if mode=="n":
            mask1 = np.random.randint(2,size=s_r.shape)
            mask2 = np.random.randint(2,size=v_r.shape)
            mask3 = np.random.randint(2,size=u_r.shape)
            #print(s_r)
            #s_r*=mask1
            #s_g*=mask1
            #s_b*=mask1
            ds_r = np.random.permutation(np.random.permutation(s_r*mask1).T)
            ds_g = np.random.permutation(np.random.permutation(s_g*mask1).T)
            ds_b = np.random.permutation(np.random.permutation(s_b*mask1).T)

            s_r = s_r*(1-mask1)+ds_r
            s_g = s_g*(1-mask1)+ds_g
            s_b = s_b*(1-mask1)+ds_b
            #v_r*=mask2
            #v_g*=mask2
            #v_b*=mask2

            #u_r*=mask3
            #u_g*=mask3
            #u_b*=mask3
            #u_r = np.random.permutation(u_r)
            #u_g = np.random.permutation(u_g)
            #u_b = np.random.permutation(u_b)
            
            r = np.absolute(np.dot(u_r,np.dot(s_r,v_r))).astype(int)
            g = np.absolute(np.dot(u_g,np.dot(s_g,v_g))).astype(int)
            b = np.absolute(np.dot(u_b,np.dot(s_b,v_b))).astype(int)            

        self._comb(r,g,b)

#----------------------------------------------------------------


#--- Methods to analyse images ----------------------------------
    def col_spectrum(self):
        #Histogram of brightnesses at each color
        r,g,b = self._split()
        fig,ax=plt.subplots(3,1,sharex=True)
        ax[0].hist(r.flatten(),bins=255,color="red")
        ax[1].hist(g.flatten(),bins=255,color="green")
        ax[2].hist(b.flatten(),bins=255,color="blue")
        plt.show()

    def mean_col(self):
        #Returns the mean colour of image
        r,g,b = self._split()
        col = [int(np.mean(r)),int(np.mean(g)),int(np.mean(b))]
        return col

    def mode_col(self):
        #Returns the most often appearing colour
        r,g,b = self._split()
        col = [int(stats.mode(r.flatten())[0]),
               int(stats.mode(g.flatten())[0]),
               int(stats.mode(b.flatten())[0])]
        return col

    
    def hist_2d(self,cols="rg"):
        h = np.zeros((256,256,3),dtype=int)
        count = np.zeros((256,256),dtype=int)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if 'r' in cols:
                    cx = self.image[i,j,0]
                    if 'g' in cols:
                        cy = self.image[i,j,1]
                    elif 'b' in cols:
                        cy = self.image[i,j,2]
                elif 'g' in cols:
                    cx = self.image[i,j,1]
                    if 'b' in cols:
                        cy = self.image[i,j,2]

                if np.all(h[cx,cy]):
                    h[cx,cy] += self.image[i,j]
                    h[cx,cy] = h[cx,cy]//2
                else:
                    h[cx,cy] = self.image[i,j]
                count[cx,cy]+=1
        self.image=h
        plt.matshow(count)
        plt.show()
        #plt.hist2d(self.image[:,:,0],self.image[:,:,1],bins=255)
        #plt.show()


    def size(self):
        s = self.image.shape
        return np.array([s[0],s[1]])