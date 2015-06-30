library(ggplot2)

setwd("~/PhD/FLSRGROUP_PCA/")

#Resources:
#Nice tutorial  - http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
#Nice visualistion and simple explanation https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/

#Why use it?
#Find patterns in high dimensional space
#Reduce the dataset to the components that explain the most variation
#Identify number of variables in a system

#Principles
#High correlation = redundancy
#Most imporant variables are those which explain the most variation
####################
#PCA projects data along the directions where the data varies the most

#Simple 2D example
#Have some expression values of genes by some measure
Gene1 <- c(-3,1,1,-3,0,-1,-1,0,-1,-1,3,4,5,-2,1,2,-2,-1,1,-2,1,-3,4,-6,1,-3,-4,3,3,-5,0,3,0,-3,1,-2,-1,0,-3,3,-4,-4,-7,-5,-2,-2,-1,1,1,2,0,0,2,-2,4,2,1,2,2,7,0,3,2,5,2,6,0,4,0,-2,-1,2,0,-1,-2,-4,-1)
Gene2 <- c(-4,-3,4,-5,-1,-1,-2,2,1,0,3,2,3,-4,2,-1,2,-1,4,-2,6,-2,-1,-2,-1,-1,-3,5,2,-1,3,3,1,-3,1,3,-3,2,-2,4,-4,-6,-4,-7,0,-3,1,-2,0,2,-5,2,-2,-1,4,1,1,0,1,5,1,0,1,1,0,2,0,7,-2,3,-1,-2,-3,0,0,0,0)

#plot the data
df <- data.frame(cbind(Gene1, Gene2))
g <- ggplot(data = df, mapping = aes(x = Gene1, y = Gene2))
g <- g + geom_point() 
g

#calculate pca # ignore how for time being
pca<-eigen(cov(df))

# calculate slopes as ratios
#new x axis
pca$slopes[1] <- pca$vectors[1,1]/pca$vectors[2,1] 
#new y axis
pca$slopes[2] <- pca$vectors[1,1]/pca$vectors[1,2]
g <- g + geom_abline(intercept = 0, slope = pca$slopes[1], colour = "green")
g <- g + geom_abline(intercept = 0, slope = pca$slopes[2], colour = "red")
g
#####################
#PCA is performed by computing eigenvectors of the covariance matrix
#What is an eigenvalue?

#mutiply matrix by a vector
a=matrix(c(2,2,3,1),2,2)
b=matrix(c(1,3),2,1)
c=matrix(c(3,2),2,1)

a %*% b
a %*% c

#Note that 
identical(a %*% c, 4 * c)

#Here c is a vector (eginvector) that has magnitude 4 (eginvalue)

#Multiplication of the matrix a can be thought of as a transformation of a vector
#egienvectors are perpendicular to each other regardless of the number of dimensions
#An n*n matrix had n eigenvectors
#PCA projects the data along the directions where the data varies the most.
#These directions are determined by the eigenvectors of the covariance matrix corresponding to the largest eigenvalues.
#The magnitude of the eigenvalues corresponds to the variance of the data along the eigenvector directions

###################
#Manual PCA

#load the iris dataset
data(iris)

##Let's perform PCA manually
iris.data<-as.matrix(iris[,-5])

#create the covariance matrix to find dimensions that are similar
#Note the function centres the data for us
iris.cov<-cov(iris.data)

#find the eigenvectors and eigenvalues
iris.pca<-eigen(iris.cov)

#eigenvectors
iris.pca$vectors

#eigenvalues
iris.pca$values

#get the principal components
x<-iris.data %*% iris.pca$vectors


#Plot the PCs
plotData<-cbind(iris,x[,1],x[,2],x[,3])
colnames(plotData)[6:8]<-c("PC1","PC2","PC3")

#plot PC1 and PC2 with Species as the colour
g <- ggplot(data = plotData, mapping = aes(x = PC1, y = PC2))+geom_point(aes(color=Species,size=3))+scale_size_identity(guide=FALSE)
g

#The eigenvalues give the variance explained by each PC
vars<-iris.pca$values/sum(iris.pca$values)*100

#R has a built in base function
pca<-prcomp(iris.data,center=T,scale=F)

#eigenvectors
pca$rotation

#The multiplication step is done for us
head(pca$x)

#Gives stdev rather than variance so need to square
vars2<-pca$sdev^2/sum(pca$sdev^2)*100


#####################
#Data reconstruction

#Let's try reconstructing visual data from pca to demonstrate dimension reduction (compression)

#load the data (im.train.sample)
#Source https://www.kaggle.com/c/facial-keypoints-detection
load(file="faceImages.RData")

#Each column is a grey scale image of a face (96*96 pixels = 9216 rows)
#plot a face!
im <- matrix(data=rev(im.train.sample[20,]), nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))


pca<-prcomp(im.train.sample,center=T,scale=T)

#How many PCs are important?
summary(pca)
screeplot(pca,npcs=30,type="line")

#how little data do we need to reconstuct the faces?
plotFaces<-function(pca,num_pc,faceID){

data <- pca$x[,1:num_pc] %*% t(pca$rotation[,1:num_pc])
# unscale and uncenter the data
data <- scale(data, center = FALSE , scale=1/pca$scale)
data <- scale(data, center = -1 * pca$center, scale=FALSE)

# plot your original image and reconstructed image
par(mfcol=c(1,2), mar=c(1,1,2,1))
im <- matrix(data=rev(im.train.sample[faceID,]), nrow=96, ncol=96)
image(1:96, 1:96, im, col=gray((0:255)/255))

rst <- matrix(data=rev(data[faceID,]), nrow=96, ncol=96)
image(1:96, 1:96, rst, col=gray((0:255)/255))
}

#all PCs should reconstruct the original data
plotFaces(pca,100,23)
#what does the first PC look like?
plotFaces(pca,1,23)
#The scree plot suggests 20 should be sufficient
plotFaces(pca,20,23)
}
#which pixels contribute the most to PC1?

#get the top 500 most influential pixels in PC1
pc1Loadings<-pca$rotation[,1]
names(pc1Loadings)<-1:length(pc1Loadings)
pc1Loadings<-sort(abs(pc1Loadings),decreasing=T)[1:1000]

#add pixels on a  background
face<-matrix(rep(0,9216),96,96)
face[as.numeric(names(pc1Loadings))]<-255
image(1:96, 1:96, face, col=gray((0:255)/255))


#############################################
#Another example - RNA-Seq data exploration

#ENCODE dataset - different paired tissues sequenced from mouse and human
#http://www.ncbi.nlm.nih.gov/pubmed/25413365
#Aimed to ask how similar the tissue between the species are
#i.e do the samples group by species or sample

#Recent refutation study
#http://f1000r.es/5ez

#For the mouse genes data the data been mapped to human genes by orthologs.
#load the FPKM values and the sample info from the refutation study
Stanford_datasets <- read.delim("~/PhD/Stanford_datasets.txt", header=F)
Stanford_datasets_fpkmMat <- read.delim("~/PhD/Stanford_datasets_fpkmMat.txt", header=F)

#log transform and get rid of the zero variance rows
data<-as.matrix(log2(Stanford_datasets_fpkmMat+1))
data<-data[apply(data,1,var)!=0,]

#perform PCA
pca<-prcomp(t(data),center=T,scale=T)

#add to the dataset info
plotData<-cbind(Stanford_datasets,pca$x[,1],pca$x[,2],pca$x[,3])
colnames(plotData)<-c("Tissue_Species","Batch","Species","Tissue","PC1","PC2","PC3")

#plot PC1 and PC2 with Species as the colour
g <- ggplot(data = plotData, mapping = aes(x = PC1, y = PC2))+geom_point(aes(color=Species,size=5))+scale_size_identity(guide=FALSE)
g

#plot PC1 and PC2 with Species as the colour, Batch as point shape
g <- ggplot(data = plotData, mapping = aes(x = PC1, y = PC2))+geom_point(aes(color=Species,size=5,shape=Batch))+scale_size_identity(guide=FALSE)
g

