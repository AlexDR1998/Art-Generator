import numpy as np
import sys
from numpy import linalg as LA
from scipy import linalg as LA
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
from scipy import signal
from scipy import stats
from scipy.optimize import minimize
import random
import math
import time


from image_manipulator import Image 


class ArtGenerator(Image):


	"""

	Higher level subclass of Image - uses lower level methods in sequence to produce
	unusual or abstract versions of images
	"""

	def abstract1(self,f):
    	
	    #Abstract nonsense
	    #f between 1 and 10 works best

	    self.rgb_partition("rgb",[random.randint(0,1),random.randint(0,1),random.randint(0,1)],random.randint(0,20))
	    self.fft_filter(f,"h")
	    self.perm_chunk()
	    self.fft_filter(f*2,"l")
	    self.edge()
	    self.smooth(2)
	    self.edge()
	    self.norm() 
	    self.show()


	def mangle(self,am,show=True):
	    
	    #Shuffles image beyond recognition
	    for x in range(am):   
	        self.rgb_partition("rgb",[random.randint(0,1),random.randint(0,1),random.randint(0,1)],random.randint(0,x))
	        #if x%2==0:
	    for x in range(am):
	        self.perm_chunk("b")
	    if show: 
	    	self.show()

	def abstract3(self,am):
		#only works for square images
		self.pixelate(8)
		for x in range(4):
			self.rgb_partition("rgb",[random.randint(0,1),random.randint(0,1),random.randint(0,1)],x)
			self.svd("u")
		self.pixelate(1/(8.0))
		#self.svd("u")

		self.show()

	def glitch(self,am):
		axes = np.array([random.randint(0,1),random.randint(0,1),random.randint(0,1)])
		self.interleave([random.randint(0,am),random.randint(0,am),random.randint(0,am)],"v")
		for x in range(am):
			self.roll_chunk(x)
		self.rgb_offset([random.randint(0,am),random.randint(0,am),random.randint(0,am)],1-axes)
		self.rgb_partition("rgb",axes,[random.randint(0,am),random.randint(0,am),random.randint(0,am)])
		self.interleave([random.randint(0,am),random.randint(0,am),random.randint(0,am)],"h")
		#self.fft_filter(100,"l")
		self.show()

	def glitch2(self,am):
	    
	    #tamer glitching

	    l_im = Image()
	    u_im = Image()
	    self.thresh_split(u_im,l_im,128)
	    #for x in range(am//10):
	        
	    #    u_im.perm_chunk("b")
	    #    l_im.perm_chunk("b")
	    u_im.interleave([random.randint(0,am),random.randint(0,am),random.randint(0,am)],"h")
	    l_im.interleave([random.randint(0,am),random.randint(0,am),random.randint(0,am)],"v")
	    u_im.fft_filter(100,"l")
	    l_im.fft_filter(10,"h")
	    self.add(u_im,l_im)
	    self.smooth(3)
	    self.show()

	def multi_voro(self,th,am1,am2):
		l_im = ArtGenerator()
		u_im = ArtGenerator()
		self.thresh_split(u_im,l_im,th,2,[210,180,150])
		l_im.voronoi(am1)
		u_im.voronoi(am2)
		l_im.show()
		u_im.show()
		self.add(u_im,l_im)
		self.show()

	def glitch3(self,f,am,mode=0):
		axes = np.array([random.randint(0,1),random.randint(0,1),random.randint(0,1)])
		
		m = np.array([[1.0,0,0],
				  [0,1.0,0],
				  [0,0,1.0]])
		m+=(np.random.random((3,3))-0.5)


		sub_im1 = ArtGenerator()
		sub_im2 = ArtGenerator()
		self.thresh_split(sub_im1,sub_im2,f)
		if mode==0:
			sub_im1.rgb_partition("rgb",axes,[random.randint(0,am),random.randint(0,am),random.randint(0,am)])
			#sub_im1.col_mat(m)
			#sub_im1.show()
			sub_im2.norm()
			sub_im2.key_black(sub_im1,10,1)
			#sub_im2.show()
		else:
			sub_im2.rgb_partition("rgb",axes,[random.randint(0,am),random.randint(0,am),random.randint(0,am)])
			#sub_im2.col_mat(m)
			#sub_im2.show()
			sub_im1.norm()
			sub_im1.key_black(sub_im2,10,1)
			#sub_im1.show()
		self.add(sub_im1,sub_im2)
		self.show()


	def chop(self,am,mode="hv"):
		#self.pixelate(8)
		if mode=="hv":
			for x in range(1,am):
				r = np.random.randint(0,2*(am-x))-am+x
				self.roll_chunk(r,"h")
				r = np.random.randint(0,2*x)-x
				self.roll_chunk(r,"v")
		elif mode=="vh":
			for x in range(1,am):
				r = np.random.randint(0,2*(am-x))-am+x
				self.roll_chunk(r,"v")
				r = np.random.randint(0,2*x)-x
				self.roll_chunk(r,"h")
		elif mode=="h":
			for x in range(1,am):
				r = np.random.randint(0,2*x)-x
				self.roll_chunk(r,"h")
		else:
			for x in range(1,am):
				r = np.random.randint(0,2*x)-x
				self.roll_chunk(r,"v")


		self.show()



	def twist(self,am):
	    #Rotates image and slices it in horrible ways
	    for x in range(am):
	        self.rotate(360//am)
	        self.perm_chunk("h")
	        self.rgb_offset([random.randint(0,am),random.randint(0,am),random.randint(0,am)],[0,0,0])
	    self.show()


	

	def abstract2(self,am):
	    self.fold(3)
	    self.smooth(5)
	    self.thresh(100)
	    self.fft_perm_chunk("b",n=am)
	    self.fft_offset([random.randint(0,am),random.randint(0,am),random.randint(0,am)],[random.randint(0,1),random.randint(0,1),random.randint(0,1)])
	    self.bright(4)	    
	    self.show()


	def bright_pixel(self,am1,am2,th):
		axes = np.array([random.randint(0,1),random.randint(0,1),random.randint(0,1)])


		sub_im1 = ArtGenerator()
		sub_im2 = ArtGenerator()

		#self.copy_to(sub_im1)
		#self.copy_to(sub_im2)
		res = gcd(self.image.shape[1],self.image.shape[0])
		#res = 1024
		self.thresh_split(sub_im1,sub_im2,th)
		
		

		print(res)
		sub_im1.pixelate(res/float(am1))
		sub_im1.pixelate(float(am1)/res)

		sub_im2.pixelate(res/float(am2))
		sub_im2.pixelate(float(am2)/res)


		#self.add(sub_im1,sub_im2)
		#sub_im1.rgb_partition("rgb",axes,[random.randint(0,am1),random.randint(0,am1),random.randint(0,am1)])
		#sub_im1.fft_filter(am1,"l")
		sub_im1.key(sub_im2,th)
		#sub_im1.show()
		#sub_im2.show()
		self.copy_from(sub_im1)
		self.show()

	def rgb_pixel(self,am=[1,2,4]):
		#r_im = self.image[:,:,0]
		#g_im = self.image[:,:,1]
		#b_im = self.image[:,:,2]
		res = gcd(self.image.shape[1],self.image.shape[0])
		r = ArtGenerator()
		g = ArtGenerator()
		b = ArtGenerator()
		self.copy_to(r)
		self.copy_to(g)
		self.copy_to(b)

		r.monochrome("r")
		g.monochrome("g")
		b.monochrome("b")
		
		r.pixelate(res/float(am[0]))
		r.pixelate(float(am[0])/res)

		g.pixelate(res/float(am[1]))
		g.pixelate(float(am[1])/res)

		b.pixelate(res/float(am[2]))
		b.pixelate(float(am[2])/res)
		
		

		self.add(r,g)
		self.add(self,b)
		self.show()
		#r.show()
		#g.show()
		#b.show()

		#self._comb(r,g,b)
		#self.show()
		#plt.matshow(r_im)
		#plt.show()


	def mangle_parallel(self,am):
		r = ArtGenerator()
		g = ArtGenerator()
		b = ArtGenerator()
		self.copy_to(r)
		self.copy_to(g)
		self.copy_to(b)
		cols = np.array([random.randint(0,1),random.randint(0,1),random.randint(0,1)])
		if cols[0]>0.5:
			r.col_inv()
		if cols[1]>0.5:
			g.col_inv()
		if cols[2]>0.5:
			b.col_inv()
		
		r.monochrome("r")
		g.monochrome("g")
		b.monochrome("b")

		r.mangle(am[0])#,False)
		g.mangle(am[1])#,False)
		b.mangle(am[2])#,False)

		self.add(r,g)
		self.add(self,b)
		self.show()

	def multi_chop(self,am,modes=["hv","hv","hv"]):
		r = ArtGenerator()
		g = ArtGenerator()
		b = ArtGenerator()
		self.copy_to(r)
		self.copy_to(g)
		self.copy_to(b)
		cols = np.array([random.randint(0,1),random.randint(0,1),random.randint(0,1)])

		if cols[0]>0.5:
			r.col_inv()
		if cols[1]>0.5:
			g.col_inv()
		if cols[2]>0.5:
			b.col_inv()
		
		r.monochrome("r")
		g.monochrome("g")
		b.monochrome("b")

		r.chop(am[0],modes[0])
		g.chop(am[1],modes[1])
		b.chop(am[2],modes[2])

		self.add(r,g)
		self.add(self,b)
		self.show()


	def kscope1(self,am):
		
		i=random.randint(0,3)
		j=random.randint(0,1)
		for x in range(am):
			#self.mirror(random.randint(0,3),random.randint(0,1))
			self.mirror(i,j)
			self.rotate(360/am)
			#self.show()
		#self.mirror(0,random.randint(0,1))
		#self.mirror(1,random.randint(0,1))
		#self.rotate(30)
		#self.mirror(2,random.randint(0,1))
		#self.mirror(3,random.randint(0,1))
		self.show()


	def voronoi_key_other(self,other,key,t,res=16,col=[200,200,200],density=500,mode=0):
		key.pixelate(res)
		key.voronoi_monte_carlo2(density,col,10)
		key.pixelate(1/float(res))
		key.show()
		self.key3(other,key,t,mode)
		self.show()
		self.save()

	def voronoi_key_self(self,t,res=16,col=[200,200,200],density=500,mode=0):
		key = ArtGenerator()
		key.copy_from(self)
		key.pixelate(res)
		key.voronoi_monte_carlo2(density,col,3)
		key.pixelate(1/float(res))
		key.show()
		self.col_inv()
		self.key(key,t,mode)
		self.show()

		self.save()


	def smear(self,l=2,g=0,mode="vh"):

		for i in range(l):
			self.streak(mode,g)
		self.show()
		self.save()


	def conv_noise(self,am,sm=4,sh=1):
		for a in range(am):
			self.smooth(sm)
			for i in range(sh):
				self.sharpen()
		self.show()
		self.save()

	def bandpass(self,freq,width,gain,mode=0):
		if mode==0:
			self.fft_filter(freq-width,"h")
			self.fft_filter(freq+width,"l")
		else:
			self.fft_filter(freq+width,"l")
			self.fft_filter(freq-width,"h")
		self.bright(gain)
		#self.show()
		#self.save()

	def notch(self,freq,width,gains):
		h = ArtGenerator()
		self.copy_to(h)
		h.fft_filter(freq+width,"h")
		self.fft_filter(freq-width,"l")
		h.bright(gains[1])
		self.bright(gains[0])
		self.add(self,h)
		#self.bright(gain)
		#self.show()
		
	def filterbank(self,fs,width,gains):
		l = ArtGenerator()
		m = ArtGenerator()
		h = ArtGenerator()
		self.copy_to(l)
		self.copy_to(m)
		self.copy_to(h)

		l.bandpass(fs[0],width,gains[0])
		m.bandpass(fs[1],width,gains[1])
		h.bandpass(fs[2],width,gains[2])
		self.add(l,m)
		self.add(self,h)
		self.show()
		self.save()


	def filter_pixel(self,px,fr,w,br,c,col=0.1):
		self.pixelate(px)
		m = (np.random.random((3,3))-0.5)*col+np.eye(3)
		self.voronoi_block(200/px,2)
		self.pixelate(1.0/px)
		#self.filterbank([1,4,10],0.001,[2,4,8])
		self.col_mat(m)
		self.notch(fr,w,br)
		self.contrast(c)
	
		self.show()


	def spongle(self):
		col = (self.mean_col())
		m=(np.random.random((3,3))-0.5)*3

		self.col_mat(m)
		self.show()
		self.voronoi_monte_carlo2(1000,col,5,2,5)
	
		self.show()
		self.save()


	def spectral(self,f=9,mode="n"):
		self.smooth(f)
		
		self.svd(mode)
		
		self.smooth(f)
		self.show()
		self.save()

	def feedback(self,r_am,z_am,iters):
		im = ArtGenerator()
		dims = self.size()
		m=np.eye(3)+(np.random.random((3,3))-0.5)*0.2

		for i in range(iters):
			self.copy_to(im)	
			#im.show()
			im.zoom(z_am)
			im.rotate(r_am)
			im.crop((dims*(z_am-1)).astype(int)//2,dims)
			#im.col_mat(m)
			
			#im.show()
			#self.key(im,20)
			self.mean(im)
		self.show()

def gcd(x,y):
	if x==y:
		return x
	while(y): 
		x,y = y,x%y 
	return x 









def main():
	im1 = ArtGenerator()
	im2 = ArtGenerator()
	im_key = ArtGenerator()
	#--- im1.load("path_to_image_file",x_center,y_center,size)

	#while True:
	#	im1.load(0)
	#	#im2.load(4)
	#	im1.show()
	#	im1.kscope1(24)
	#	im1.save()
	

	im1.load(n=0)
	
	#im1.hist_2d("rb")
	im1.show()
	im1.feedback(0,1.1,30)
	#im1.smooth()
	#im1.sort("r",0)
	#im1.sort("g",1)
	
	
	
	
	
	
	#im1.image = X.reshape(ii)
	#im1.show()
	#im1.mod_kal([300,300],[300,300],0)
	#im1.kscope1(12)
	#im1.save()
	#im1.show()
	#im1.filter_pixel(2,7,3,[0.8,2],1,0.5)
	#im1.col_inv()
	#im1.mangle(5)
	#im1.col_inv()
	#im1.pixelate(4)
	#im1.voronoi(1000,3,1)
	#im_key.load(n=1)
	#im1.show()
	#im1.polynom([0,1,-0.0009,0],[[280,350],[200,200],[0,0]])
	#im1.polynom([1,1,-0.001,0],[[400,200],[0,500],[400,200]])
	#im1.mangle(6)
	#im1.fft()
	#im1.voronoi_block(20)
	#im2.mangle(7)
	#im1.show()
	#im1.fft_perm_chunk("h",5)
	#im1.fft_symm(1,1)
	#for i in range(2):
	#	im1.streak("vh",40)
	#im1.pixelate(4)
	#im1.glitch3(128,20)
	

	
	#im2.show()
	#im1.mean(im2)
	
	
	
	

	#im1.bandpass(5,0.01,10)
	#im1.interleave([10,10,10],"h")
	#im1.black_and_white([1,1,1])
	#im1.show()
	#im1.col_inv()
	#im1.show()
	#im1.fft_filter(1,"l")

	#im1.fft()
	#im1.show()
	#im1.conv_noise(10,16,2)
	#im1.voronoi_block(80,5)
	#im1.smooth(5)
	#im1.save()
	#im1.smear(4,0)
	#im1.glitch3(100,200,0)
	#im1.mirror(1,0,1)
	#im1.fft_filter(7,"h")
	#im1.fft_filter(7.1,"l")
	#im1.norm()
	#im1.bright(10)
	#im1.show()
	#im1.save()
	#im1.save()
	#im1.mangle(100)
	
	#im1.show()
	#im1.show()
	#im1.save()
	#im2.key_col_2(im1,10,[240,200,150])
	#im2.show()
	#im1.fft_rgb_partition("rgb",[0,1,0],[10,10,10])
	#mat = np.zeros((5,5))
	#mat[0,0]=1
	#mat[-1,-1]=-1
	#mat[0,-1]=1
	#mat[-1,0]=1
	#im1.lu()
	#im1.conv_mat(mat)
	#im1.mirror(2,1)
	#im1.show()
	#im1.show()
	#im1.save()
	#im1.kscope1(24)
	#for x in range(10):
	#	im1.sum_chunk("b")
	#im1.chop(100)
	#im1.show()
	#im2.show()
	#im1.voronoi_key_self(40,1,[180,120,120],500,0)
	#im1.show()
	#im1.multi_chop([30,30,30])
	#im2.show()
	#im1.thresh_colour([254,254,254],10,1,[0,0,0])
	#im1.voronoi_bi(1000,2,[0.36,0.3],0.7,[2,2])
	#im1.voronoi_bi(1000,2,[300,300],1,[20,20])
	
	#im1.col_inv()

	#im1.edge()
	#im1.mangle(8)
	
	#im1.thresh_colour([210,160,150],50,0,[0,0,0])
	#im1.multi_voro(500,1000,100)
	#im1.voronoi(300)
	#im1.mangle(5)
	#for x in range(10):
		#im1.streak("b")
	#im1.show()
	#im1.edge("h")
	#im1.show()
	#im1.bright(2)
	
	#im2.thresh(120,1)
	#for x in range(10):
	#	im1.flip_chunk(random.randint(0,1))
	#im1.rot_chunk()
	#im1.thresh_colour([210,160,150],80,0,[50,0,0])
	#im1.show()
	#im1.rgb_partition('g',[0,0,0],10)
	#im1.flip([1,1,1],"r")
	#im1.col_inv()
	#im1.key_black(im2,50,1)
	#im1.mangle(4)
	#im1.fft_offset([30,20,40],[1,0,1])
	#im1.fft_rgb_partition("rgb",[0,1,0],[10,10,10])
	#im1.kscope1(2)
	#im1.show()
	#im2.bright(2)
	#im2.col_inv()
	#im2.show()
	#im2.rgb_offset([100,50,40],[1,0,0])
	#im1.add(im1,im2)
	#im1.show()
	#im1.save()
	#im2.show()
	#im1.mangle_parallel([5,5,5])
	#im1.save()
	#im1.glitch(20)
	#im1.mangle(6)
	#im2.smooth(100)
	#im1.black_and_white('r')
	#im1.show()
	#im2.key(im1,50)
	#im2.mul(im1)
	#im2.show()

	#--- im1.image_editing_method(parameters)
	

	#im1.thresh(253,3)
	#im1.flip(0)
	#im1.show()
	#im1.pixelate(100)
	#im1.pixelate(0.01)
	#im1.mangle(3)
	#im1.mirror(1)
	#im1.mirror(1,0)
	#im1.mirror(1,1)
	#im1.apply_parallel([1,5,10],im1.mangle,im1.mangle,im1.mangle)
	#im1.show()
	#im1.col_inv()
	#im1.mangle(7)
	#im1.chop(100,"v")
	#im1.col_inv()
	
	#im1.edge()
	#im1.black_and_white("r")
	#im1.show()
	#im1.chop(50)
	#im1.rgb_pixel([20,30,50])
	#im1.black_and_white("g")
	#im1.mangle_parallel([6,5,6])
	#im1.multi_chop([100,100,50],["v","v","v"])
	#im1.show()
	#im1.bright_pixel(400,40,100)
	#im1.glitch2(100)
	

	#im1.show()
	#im1.glitch2(128)
	#im1.abstract3(7)
	#im1.chop(50)


main()
