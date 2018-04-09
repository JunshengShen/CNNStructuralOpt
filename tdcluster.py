from numpy import *
import random
import math
def findNeighbours(point,neighbour):#point is a list with x,y
	if point[0]!=0:
		neighbour.append([point[0]-1,point[1],point[2]])
	if point[0]!=127:
		neighbour.append([point[0]+1,point[1],point[2]])
	if point[1]!=0:
		neighbour.append([point[0],point[1]-1,point[2]])
	if point[1]!=127:
		neighbour.append([point[0],point[1]+1,point[2]])
	if point[2]!=0:
		neighbour.append([point[0],point[1],point[2]-1])
	if point[2]!=127:
		neighbour.append([point[0],point[1],point[2]+1])

def findTwoDNeighbours(point , neighbour):#point is a list with x,y
	if point[0]!=0:
		neighbour.append([point[0]-1,point[1]])
	if point[0]!=127:
		neighbour.append([point[0]+1,point[1]])
	if point[1]!=0:
		neighbour.append([point[0],point[1]-1])
	if point[1]!=127:
		neighbour.append([point[0],point[1]+1])

def distance(point1 , point2):
	return math.sqrt(pow((point1[0]-point2[0]),2)+pow((point1[1]-point2[1]),2)+pow((point1[2]-point2[2]),2))
	
def readPoints(dir):
	tdpoints=[]
	for i in range(128):
		picture = open(dir+str(i)+'.txt')
		points = list(map(int,picture.read().split(',')))
		tdpoints.append(array(points).reshape(128,128))
		
	
	return tdpoints
	

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
	
def deleteOutPoints(points , originPoints):
	#3d seed fill
	#seed=[]
	#seed.append([127,127,127])
	#points[seed[0][0]][seed[0][1]][seed[0][2]] = 0
	#count=0
	#for i in range(10):
	#	for y in range(128):
	#		for z in range(128):
	#			points[i][y][z]=0
	#while (len(seed)):
	#	neighbour = []
	#	findNeighbours(seed[0], neighbour)
	#	for i in neighbour:
	#		if points[i[0]][i[1]][i[2]] == 1:
	#			points[i[0]][i[1]][i[2]] = 0
	#			count+=1
	#			seed.append(i)
	#	seed.pop(0)
	#print('count=',count)
	for layer in points:
		seed = [];
		seed.append([0, 0])
		layer[seed[0][0], seed[0][1]] = 0
		while (len(seed)):
			neighbour = []
			findTwoDNeighbours(seed[0], neighbour)
			for i in neighbour:
				if layer[i[0], i[1]] == 1:
					layer[i[0], i[1]] = 0
					seed.append(i)
			seed.pop(0)

def cluster(k , points):
	centers=[]
	for i in range(k):
		centers.append([random.randint(0,127),random.randint(0,127),random.randint(0,127)])
	clusterPoints=[]
	for x in range(128):
		for y in range(128):
			for z in range(128):
				if points[x][y][z]==1:
					clusterPoints.append([x,y,z])
	lable=zeros([128,128,128])
	#culculate all the distance and lable points
	isEnd=False
	while(not (isEnd)):#determine whether this time cluster is end
		isEnd=True
		for point in clusterPoints:#this iteration is to lable the points
			x = point[0]
			y = point[1]
			z = point[2]
			disMin=None
			for centerIndex in range(len(centers)):				
				dis = distance(centers[centerIndex],[x,y,z])
				if disMin==None:
					disMin = dis
					lable[x][y][z]=centerIndex						
				elif disMin> dis:
					disMin=dis
					lable[x][y][z]=centerIndex
		print(1)
		#then culculate the new centers
		#deleteCenters=[]
		for centerIndex in range(len(centers)):
			xlabel=0
			ylabel=0
			zlabel=0
			pointNumber=0
			for point in clusterPoints:#this iteration is to lable the points
				x = point[0]
				y = point[1]
				z = point[2]
				if lable[x][y][z]==centerIndex:
					xlabel+=x
					ylabel+=y
					zlabel+=z
					pointNumber+=1
			if pointNumber==0:#reRandom
				centers[centerIndex]=[random.randint(0,127),random.randint(0,127),random.randint(0,127)]
				isEnd=False
				continue
				#deleteCenters.append(centerIndex)
				#continue
			nextCenterx=xlabel/pointNumber
			nextCentery=ylabel/pointNumber
			nextCenterz=zlabel/pointNumber
			if (nextCenterx != centers[centerIndex][0]) or nextCentery != centers[centerIndex][1] or  nextCenterz != centers[centerIndex][2]:
				isEnd=False
			centers[centerIndex][0]=nextCenterx
			centers[centerIndex][1]=nextCentery
			centers[centerIndex][2]=nextCenterz

	#print('centers',centers)
	return centers,lable

def slihouetteCul(k , points , lable):
	pointNumber=0
	sAdd=0
	clusterPoints=[]
	for x in range(128):
		for y in range(128):
			for z in range(128):
				if points[x][y][z]==1:
					clusterPoints.append([x,y,z])
	for pointi in random.sample(clusterPoints,50):
		x = pointi[0]
		y = pointi[1]
		z = pointi[2]
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
			z_ = pointj[2]
			distances[int(lable[x_][y_][z_])]+=distance([x,y,z],[x_,y_,z_])
			numbers[int(lable[x_][y_][z_])]+=1
		if numbers[int(lable[x][y][z])]==1:
			a=distances[int(lable[x][y][z])]
		else:
			a=distances[int(lable[x][y][z])]/(numbers[int(lable[x][y][z])]-1)
		bList=[]
		for i in range(k):
			if numbers[i]==0:
				bList.append(0)
				continue
			bList.append(distances[i]/numbers[i])
		for i in range(k):				
			if i !=lable[x][y][z]:
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

		centers = None
		lable = None
		slihouette_ = None
		for i in range(clusteTimes):
			print(k)



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
			for z in range(128):			
				if points[x][y][z]==1:
					clusterPoints.append([x,y,z])
	potentials = []
	for i in range(k):
		potentials.append(0)
	for center in range(k):
		for point in clusterPoints:
			dis = distance(centers[center],point)
			if dis<1:
				potentials[center] +=1
				continue
			elif dis<=15:
				potentials[center] += 1/pow(dis,1)
			#print(dis)
	return potentials
	
def filler(points , centers , potentials):#input is origin image and deal with it
	for x in range(128):
		for y in range(128):
			for z in range(128):
				if points[x][y][z]==0:
					for i in range(len(centers)):
						if distance([x,y,z],centers[i]) <= (potentials[i]/100):
							points[x][y][z]=1

def squareFiller(points , centers , potentials):#input is origin image and deal with it
	for x in range(128):
		for y in range(128):
			for z in range(128):
				if points[x][y][z]==0:
					for i in range(len(centers)):
						if abs(x-centers[i][0]) <= (potentials[i]/100) and abs(y-centers[i][1]) <= (potentials[i]/100) and abs(z-centers[i][2]) <= (potentials[i]/100):
							#print(x,y,z,centers[i][0],centers[i][1],centers[i][2],potentials[i]/200)
							points[x][y][z]=1


def clusterAndSave(clusterNumberMin,clusterNumberMax,clusterTimes):
	points = readPoints('result/a')
	originPoints = readPoints('result/b')
	
	
	deleteOutPoints(points,originPoints)

	count=0
	for x in range(128):
		for y in range(128):
			for z in range(128):
				if points[x][y][z]==1:
					count+=1
	print(count)
	k,centers,lable = multiCluster([clusterNumberMin,clusterNumberMax] , points , clusterTimes)
	potentials = potentialEnergy(points , k , centers , lable)

	originPointsSquare = originPoints
	filler(originPoints, centers, potentials)
	for i in range(128):
		outputFile('filled/lable'+str(i)+'.txt', originPoints[i])

	squareFiller(originPointsSquare , centers , potentials)
	print(k)

	for i in range(128):
		outputFile('filledsquare/lable'+str(i)+'.txt', originPointsSquare[i])
	
	#print('lable.txt', points[1])
	#count=0
	#for x in range(128):
	#	for y in range(128):
	#		for z in range(128):
	#			if points[x][y][z]==1:
	#				count+=1
	
	#print(count)
		
if __name__=='__main__':
	clusterAndSave(31,32,1)
