from numpy import *
import random
import math
def findNeighbours(point , neighbour):#point is a list with x,y
	if point[0]!=0:
		neighbour.append([point[0]-1,point[1]])
	if point[0]!=127:
		neighbour.append([point[0]+1,point[1]])
	if point[1]!=0:
		neighbour.append([point[0],point[1]-1])
	if point[1]!=127:
		neighbour.append([point[0],point[1]+1])

def distance(point1 , point2):
	return math.sqrt(pow((point1[0]-point2[0]),2)+pow((point1[1]-point2[1]),2))
	
def readPoints(file):
	picture = open(file)
	picture = map(int,picture.read().split(','))

	points=[];
	for i in picture:
		#print(i)
		points.append(i)
	
	return array(points).reshape(128,128)
	

def outputFile(fileName , points):
	outputy=[]
	points=points.reshape(1,16384)
	for i in range(16384):
		outputy.append(points[0,i])
		
	pre=str(outputy)
	pre=pre.replace("[","")
	pre=pre.replace("]","")+"\n"

	f=open(fileName,"w")
	f.write(pre)
	f.close()
	
def deleteOutPoints(points , originFile):
	seed = [];
	seed.append([0,0])
	points[seed[0][0],seed[0][1]]=0
	while(len(seed)):
		neighbour=[]
		findNeighbours(seed[0],neighbour)
		for i in neighbour:
			if points[i[0],i[1]]==1:
				points[i[0],i[1]]=0
				seed.append(i)
		seed.pop(0)
	originPoints = readPoints(originFile)
	for x in range(128):
		for y in range(128):
			if originPoints[x][y]==1:
				points[x][y]=0

def cluster(k , points):
	centers=[]
	for i in range(k):
		centers.append([random.randint(0,127),random.randint(0,127)])
	clusterPoints=[]
	for x in range(128):
		for y in range(128):
			if points[x,y]==1:
				clusterPoints.append([x,y])
	lable=zeros([128,128])
	#culculate all the distance and lable points
	isEnd=False
	while(not (isEnd)):#determine whether this time cluster is end
		isEnd=True
		for point in clusterPoints:#this iteration is to lable the points
			x = point[0]
			y = point[1]		
			disMin=None
			for centerIndex in range(len(centers)):				
				dis = distance(centers[centerIndex],[x,y])
				if disMin==None:
					disMin = dis
					lable[x][y]=centerIndex						
				elif disMin> dis:
					disMin=dis
					lable[x][y]=centerIndex
		#then culculate the new centers		
		for centerIndex in range(len(centers)):
			xlabel=0
			ylabel=0
			pointNumber=0
			for point in clusterPoints:#this iteration is to lable the points
				x = point[0]
				y = point[1]
				if lable[x][y]==centerIndex:
					xlabel+=x
					ylabel+=y
					pointNumber+=1
			if pointNumber==0:#reRandom
				centers[centerIndex]=[random.randint(0,127),random.randint(0,127)]
				isEnd=False
				continue
			nextCenterx=xlabel/pointNumber
			nextCentery=ylabel/pointNumber
			if (nextCenterx != centers[centerIndex][0]) or nextCentery != centers[centerIndex][1]:
				isEnd=False
			centers[centerIndex][0]=nextCenterx
			centers[centerIndex][1]=nextCentery
	#print('centers',centers)
	return centers,lable

def slihouetteCul(k , points , lable):
	pointNumber=0
	sAdd=0
	clusterPoints=[]
	for x in range(128):
		for y in range(128):
			if points[x,y]==1:
				clusterPoints.append([x,y])
	for pointi in clusterPoints:
		x = pointi[0]
		y = pointi[1]
		pointNumber+=1
		distances=[]
		numbers=[]
		a=0
		b=None
		for i in range(k):
			distances.append(0.0)
			numbers.append(0.0)
		for pointj in clusterPoints:
			x_ = pointj[0]
			y_ = pointj[1]
			distances[int(lable[x_][y_])]+=distance([x,y],[x_,y_])
			numbers[int(lable[x_][y_])]+=1
		if numbers[int(lable[x][y])]==1:
			a=distances[int(lable[x][y])]
		else:
			a=distances[int(lable[x][y])]/(numbers[int(lable[x][y])]-1)			
		bList=[]
		for i in range(k):
			if numbers[i]==0:
				bList.append(0)
				continue
			bList.append(distances[i]/numbers[i])
		for i in range(k):				
			if i !=lable[x][y]:
				if b==None:
					b=bList[i]
				else:
					if bList[i] < b:
						b = bList[i]
		s=(b-a)/max(a,b)
		sAdd+=s
	s=sAdd/pointNumber
	#print(sAdd,pointNumber,s)
	return s

def multiCluster(kNumbers , points , clusteTimes):#return a list of cluster centers
	allCenters=[]
	allLables=[]
	slihouette=[]
	for k in range(kNumbers[0],kNumbers[1]):
		print(k)
		centers = None
		lable = None
		slihouette_ = None
		for i in range(clusteTimes):
			centers_,lable_=cluster(k,points)
			slihouette_temp = slihouetteCul(k , points , lable_)
			#print(k,slihouette_temp,centers_)
			if slihouette_==None:
				slihouette_ = slihouette_temp
				centers = centers_
				lable = lable_
			if slihouette_ < slihouette_temp:
				slihouette_ = slihouette_temp
				centers = centers_
				lable = lable_
		allCenters.append(centers)
		allLables.append(lable)
		slihouette.append(slihouette_)
		#for x in range(128):
			#	for y in range(128):
				#	if(allLables[0][x][y]!=0):
				#		print(allLables[0][x][y])
		

	#then culculate the silhoutte conefficient
	maxSlihouette = 0
	index = None
	temp = 0
	for i in slihouette:
		if i > maxSlihouette:
			maxSlihouette = i
			index = temp
		temp += 1
	#print(index)
	
	return (kNumbers[0]+index),allCenters[index],allLables[index]
	
	
def potentialEnergy(points , k , centers , lable):
	clusterPoints = []
	for x in range(128):
		for y in range(128):
			if points[x][y]==1:
				clusterPoints.append([x,y])
	potentials = []
	for i in range(k):
		potentials.append(0)
	for center in range(k):
		for point in clusterPoints:
			dis = distance(centers[center],point)
			if dis<=15:
				potentials[center] += 1/pow(dis,1)
			#print(dis)
	return potentials
	
def filler(points , centers , potentials):#input is origin image and deal with it
	for x in range(128):
		for y in range(128):
			if points[x][y]==0:
				for i in range(len(centers)):
					if distance([x,y],centers[i]) <= (potentials[i]/8):
						points[x][y]=1
	
if __name__=='__main__':
	points = readPoints('cluster.txt')
	
	deleteOutPoints(points , 'outPoints.txt')
	
	k,centers,lable = multiCluster([4,5],points,1)
	print(k , centers , slihouetteCul(k , points , lable))
	#print(points[1][1])
	#outputFile('lable.txt', points)
	potentials = potentialEnergy(points , k , centers , lable)
	print(potentials)
	originPoints = readPoints('outPoints.txt')
	filler(originPoints , centers , potentials)
	outputFile('lable.txt', originPoints)
	
	#for k in range(10,20):
	#clusters=[]
	#for i in range(k):
	#	clusters.append([random.randint(0,127),random.randint(0,127)])
	
	#culculate all the distance
	#disMin=0

		

