from ij import IJ, ImagePlus, ImageStack, CompositeImage
import os
from ij.io import FileSaver
from ij.plugin import ContrastEnhancer
from ij.plugin.frame import RoiManager
from ij.plugin.filter import ParticleAnalyzer as PA, GaussianBlur
from ij.measure import Measurements as ms, ResultsTable
import shutil
import math
import ij.gui.Plot as Plot
import ij.gui.PlotWindow as PlotWindow
import array
from java.awt import Color


def linspace(x1,x2,N):
	diff = x2-x1
	step = diff/float(N-1)
	lin = []
	for i in range(int(N)):
		lin.append(x1+step*i)
	return lin
def mean(x):
	m = sum(x)/float(len(x))
	return m
def avgPath(X,Y):
	NumberNewPoints = 5
	xPath = []
	yPath = []
	
	for i in range(len(X)-1):	
	    xvals = linspace(X[i], X[i+1], NumberNewPoints+2)
	    yvals = linspace(Y[i], Y[i+1], NumberNewPoints+2)
	    if i != 0:
	        xvals = xvals[1:-1]
	        yvals = yvals[1:-1]
	    for j in range(len(xvals)):
		    xPath.append(xvals[j])    
		    yPath.append(yvals[j])
	
	windowSize = NumberNewPoints*3 	
	xPathSmooth = []
	yPathSmooth = []
	for i in range(math.trunc(windowSize/2),int(len(xPath)-math.ceil(windowSize/2))):
	    xPathSmooth.append( mean (xPath [int(i-math.trunc(windowSize/2)):int(i+math.ceil(windowSize/2))] ) )
	    yPathSmooth.append( mean (yPath [int(i-math.trunc(windowSize/2)):int(i+math.ceil(windowSize/2))] ) )	
	
	#uno el primer y ultimo punto con el avg path asi me queda una avg path completo

	xvals1 = linspace(X[0], xPathSmooth[0], math.ceil(windowSize/2)+2)
	yvals1 = linspace(Y[0], yPathSmooth[0], math.ceil(windowSize/2)+2)
	xvalsEnd = linspace(xPathSmooth[-1],X[-1], math.ceil(windowSize/2)+2)
	yvalsEnd = linspace(yPathSmooth[-1],Y[-1], math.ceil(windowSize/2)+2)	
	for i in range(len(xvals1)-1):
		xPathSmooth = [xvals1[-i-2]]+xPathSmooth+[xvalsEnd[i+1]]
		yPathSmooth = [yvals1[-i-2]]+yPathSmooth+[yvalsEnd[i+1]]

	return xPathSmooth, yPathSmooth

def VSL(X,Y,T):
	dist = math.sqrt((X[-1]-X[0])**2+(Y[-1]-Y[0])**2)
	time = T[-1]-T[0]
	vsl = dist/time
	return vsl

def VCL(X,Y,T):
	vel = []
	for i in range(len(X)-1):
		dist = math.sqrt((X[i+1]-X[i])**2+(Y[i+1]-Y[i])**2)
		time = T[i+1]-T[i]
		vel.append(dist/time)
	vcl = mean(vel)
	return vcl


def VAP(X,Y,avgPathX,avgPathY,T):
	vel = []
	minIndexOld = 0		
	for j in range(1,len(X)):
		minDist = float('Inf')
		for i in range (len(avgPathX)):
			dist = math.sqrt((X[j]-avgPathX[i])**2+(Y[j]-avgPathY[i])**2)
			if dist < minDist:
				minIndex = i
				minDist = dist
		dist = 0
		if minIndex>=minIndexOld:
			for i in range(minIndexOld,minIndex):
				dist = dist + math.sqrt((avgPathX[i+1]-avgPathX[i])**2+(avgPathY[i+1]-avgPathY[i])**2) 
		else:
			for i in range(minIndex,minIndexOld):
				dist = dist - math.sqrt((avgPathX[i+1]-avgPathX[i])**2+(avgPathY[i+1]-avgPathY[i])**2) 		
		minIndexOld = minIndex
		time = T[j]-T[j-1]
		vel.append(dist/time)
	vap = mean(vel)
	return vap


def ALH(X,Y,avgPathX,avgPathY):
	alh = []		
	for j in range(len(X)):
		minDist = float('Inf')
		for i in range (len(avgPathX)):
			dist = math.sqrt((X[j]-avgPathX[i])**2+(Y[j]-avgPathY[i])**2)
			if dist < minDist:
				minDist = dist
		alh.append(minDist)
	return mean(alh)

def LIN(X,Y,T):
	lin = VSL(X,Y,T)/VCL(X,Y,T)
	return lin

def WOB(X,Y,avgPathX,avgPathY,T):
	wob = VAP(X,Y,avgPathX,avgPathY,T)/VCL(X,Y,T)
	return wob

def STR(X,Y,avgPathX,avgPathY,T):
	stra = VSL(X,Y,T)/VAP(X,Y,avgPathX,avgPathY,T)
	return stra

def BCF(X,Y,avgPathX,avgPathY,T):
	bcf = []
	for j in range(1,len(X)):
		minDist = float('Inf')
		for i in range (len(avgPathX)):
			dist = math.sqrt((X[j]-avgPathX[i])**2+(Y[j]-avgPathY[i])**2)
			if dist < minDist:
				minIndexNew = i
				minDist = dist
		if j>1:
			Ax = avgPathX[minIndexOld]
			Ay = avgPathY[minIndexOld]
			Bx = avgPathX[minIndexNew]
			By = avgPathY[minIndexNew]
			discNew = (Bx - Ax) * (Y[j] - Ay) - (By - Ay) * (X[j] - Ax)
			discOld = (Bx - Ax) * (Y[j-1] - Ay) - (By - Ay) * (X[j-1] - Ax)
			if discOld*discNew<0:
				bcf.append(1)
			else:
				bcf.append(0)
		minIndexOld = minIndexNew
	return mean(bcf)

def MAD(X,Y):
	mad = []
	for i in range(1,len(X)-1):
		if (X[i]-X[i-1])!= 0:
			pend1 = (Y[i]-Y[i-1])/(X[i]-X[i-1])
			angle1 = math.atan(pend1)
			if (pend1 < 0 and X[i]<X[i-1]) or (pend1 > 0 and X[i]<X[i-1]):
				angle1 = angle1+math.pi
			else:
				angle1 = 2*math.pi+angle1
		elif Y[i]>Y[i-1]:
			angle1 = math.pi/2
		else: 
			angle1 = -math.pi/2
		if (X[i+1]-X[i])!= 0:
			pend2 = (Y[i+1]-Y[i])/(X[i+1]-X[i])			
			angle2 = math.atan(pend2)
			if (pend2 < 0 and X[i+1]<X[i]) or (pend2 > 0 and X[i+1]<X[i]):
				angle2 = angle2+ math.pi
			else:
				angle2 = 2*math.pi+angle2
		elif Y[i+1]>Y[i]:
			angle2 = math.pi/2
		else: 
			angle2 = -math.pi/2
		mad.append(abs(angle1-angle2))
	return mean(mad)




def get_carac(fileName):

	POS_TRACK_ID = 3
	POS_X = 5
	POS_Y = 6
	POS_T = 8

	X = []
	Y = []
	T = []
	TRACK_ID = []
	with open(fileName,'r') as file:
		lines = file.read().split("\n")
		terminos = lines[1].split(",")
		trackId = int(terminos[POS_TRACK_ID])
		x = []
		y = []
		t = []
		i = 1
		while True:
			if lines[i]:
				terminos = lines[i].split(",")
				if trackId != int(terminos[POS_TRACK_ID]):
					TRACK_ID.append(trackId)
					trackId = int(terminos[POS_TRACK_ID])
					#print(trackId)
					X.append(x)
					Y.append(y)
					T.append(t)
					x = []
					y = []
					t = []
				elif t == [] or float(terminos[POS_T]) != t[-1]:
					x.append(float(terminos[POS_X]))
					y.append(float(terminos[POS_Y]))
					t.append(float(terminos[POS_T]))
					i = i+1
				else:
					i = i + 1
			else:
				X.append(x)
				Y.append(y)
				T.append(t)
				break
	
	file.close()
	print(len(X))
	#print(X)
	allX = X
	allY = Y
	allT = T
	
	CARAC_WHO = []
	CARAC_FOUR = []
	CARAC_SIX = []
	
	
	for i in range(len(allX)):
	
		X = allX[i]
		Y = allY[i]
		T = allT[i]
		#print('sabe',X)
		avgPathX, avgPathY = avgPath(X,Y)
	
	
		vcl = VCL(X,Y,T)
		vsl = VSL(X,Y,T)
		vap = VAP(X,Y,avgPathX,avgPathY,T)
		alh = ALH(X,Y,avgPathX,avgPathY)
		lin = LIN(X,Y,T)
		wob = WOB(X,Y,avgPathX,avgPathY,T)
		stra = STR(X,Y,avgPathX,avgPathY,T)
		bcf = BCF(X,Y,avgPathX,avgPathY,T)
		mad = MAD(X,Y)
	
		carac_six = [vcl,vsl,vap,lin,bcf,mad]
	
		CARAC_SIX.append(carac_six)
		#print(vcl,vsl,vap,alh,lin,wob,stra,bcf,mad)
		carac_who = [vcl,vsl,vap,alh,lin,wob,stra,bcf,mad]
	
		CARAC_WHO.append(carac_who)
	
		carac_four = [vap,lin,alh,bcf]
	
		CARAC_FOUR.append(carac_four)
	
	
	save_filename = os.path.splitext(fileName)[0]
	
	with open(save_filename+'_features_six.arff','w') as file:
		file.write('@relation \'sperm\' \n')
		file.write('@attribute VCL real \n')
		file.write('@attribute VSL real \n')
		file.write('@attribute VAP real \n')
		file.write('@attribute LIN real \n')
		file.write('@attribute BCF real \n')
		file.write('@attribute MAD real \n')
		file.write('@data \n \n')
		for carac in CARAC_SIX:
			#print(carac)
			for i in range(len(carac)):
				file.write(str(carac[i]))
				if i != len(carac)-1:
					file.write(',')
			file.write('\n')
	
		file.write('\n')
	
	file.close()
	
	with open(save_filename+'_features_four.arff','w') as file:
		file.write('@relation \'sperm\' \n')
		file.write('@attribute VAP real \n')
		file.write('@attribute LIN real \n')
		file.write('@attribute ALH real \n')
		file.write('@attribute BCF real \n')
		file.write('@data \n')
		for carac in CARAC_FOUR:
			#print(carac)
			for i in range(len(carac)):
				file.write(str(carac[i]))
				if i != len(carac)-1:
					file.write(',')
			file.write('\n')
	
		file.write('\n')
	
	file.close()
	
	with open(save_filename+'_features_features_who.arff','w') as file:
		file.write('@relation \'sperm\' \n')
		file.write('@attribute VCL real \n')
		file.write('@attribute VSL real \n')
		file.write('@attribute VAP real \n')
		file.write('@attribute ALH real \n')
		file.write('@attribute LIN real \n')
		file.write('@attribute WOB real \n')
		file.write('@attribute STRAIGHTNESS real \n')
		file.write('@attribute BCF real \n')
		file.write('@attribute MAD real \n')
		file.write('@data \n')
		for carac in CARAC_WHO:
			#print(carac)
			for i in range(len(carac)):
				file.write(str(carac[i]))
				if i != len(carac)-1:
					file.write(',')
			file.write('\n')
	
		file.write('\n')
	file.close()

	with open(save_filename + '_features_who.csv', 'w') as file:
		for carac in CARAC_WHO:
			# print(carac)
			for i in range(len(carac)):
				file.write(str(carac[i]))
				if i != len(carac) - 1:
					file.write(',')
			file.write('\n')

		file.write('\n')

	file.close()


#main


filename = IJ.getFilePath("Select a csv File, spots output of trackmate")
get_carac(filename)

