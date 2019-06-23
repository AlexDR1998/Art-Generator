import numpy as np
import sys
from numpy import linalg as LA
from scipy import linalg as LA
import matplotlib.pyplot as plt
from scipy import misc
from scipy import ndimage
from scipy import signal
import random
import math

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

	def glitch3(self,f):
		
		sub_im1 = ArtGenerator()
		sub_im2 = ArtGenerator()
		self.thresh_split(sub_im1,sub_im2,f)
		
		sub_im1.show()
		sub_im2.show()


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


def gcd(x,y):
	if x==y:
		return x
	while(y): 
		x,y = y,x%y 
	return x 


def main():
	im1 = ArtGenerator()
    
	#--- im1.load("path_to_image_file",x_center,y_center,size)

	im1.load()

	#im1.show()

	#--- im1.image_editing_method(parameters)
	
	#for i in range(10):
	#im1.pixelate(64)
	#im1.pixelate(1.0/64.0)
	#im1.abstract3(200,20,160)
	#im1.fft_filter(20,"l")
	#im1.svd("u")
	

	#im1.abstract2(20)
	#for x in range(9):
	#	im1.col_rot()
	#	im1.mangle(4)
	
	#im1.col_rot()
	#im1.mangle(3)
	#im1.col_rot()
	#im1.mangle(3)

	#im1.apply_parallel([1,5,10],im1.mangle,im1.mangle,im1.mangle)
	#im1.show()
	#im1.col_inv()
	#im1.mangle(7)
	#im1.chop(100,"v")
	#im1.col_inv()
	
	#im1.edge()
	#im1.show()
	#im1.monochrome("r")
	im1.mangle_parallel([6,5,6])
	#im1.rgb_pixel([1,3,5])
	#im1.multi_chop([100,100,50],["hv","hv","hv"])
	#im1.show()
	#im1.bright_pixel(400,40,100)
	#im1.glitch2(100)
	

	#im1.show()
	#im1.glitch2(128)
	#im1.abstract3(7)
	#im1.chop(50)
	im1.save()


main()
