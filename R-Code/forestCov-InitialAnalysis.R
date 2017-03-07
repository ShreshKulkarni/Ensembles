library(corrplot)
library(ggplot2)
library(caret)

forestCovDS <- read.csv("~/MSPA/Kaggle/Datasets/covtype.csv")
summary(forestCovDS)

#Correlation Matrix for numerical attributes
corMat <- cor(forestCovDS[,1:10])
corrplot(corMat,method = "square",type="upper")

#Combining the binary qualitative attributes for better visualization
forestCovCom <- forestCovDS
forestCovCom$SoilType <- forestCovCom$Soil_Type1

for(name in colnames(forestCovDS)[grep(glob2rx("Soil_Type*"),colnames(forestCovDS))]) {
  forestCovCom[which(forestCovCom[,name] == 1),"SoilType"]<- strsplit(name,"e")[[1]][2]
  forestCovCom[,name] <- NULL
}

forestCovCom$WildernessArea <- forestCovCom$Wilderness_Area1
for(name in colnames(forestCovDS)[grep(glob2rx("Wilderness_Area*"),colnames(forestCovDS))]) {
  forestCovCom[which(forestCovCom[,name] == 1),"WildernessArea"]<- strsplit(name,"a")[[1]][2]
  forestCovCom[,name] <- NULL
}

rm(forestCovDS)
forestCovCom$SoilType <- factor(forestCovCom$SoilType)
forestCovCom$WildernessArea <- factor(forestCovCom$WildernessArea)
forestCovCom$Cover_Type <- factor(forestCovCom$Cover_Type)


#Cover Type per wilderness area
ggplot(data=as.data.frame(table(forestCovCom$Cover_Type,forestCovCom$WildernessArea)),
       aes(x=reorder(Var1,Freq),y=Freq,fill=Var2)) + geom_bar(stat="identity")+
  facet_grid(.~Var2)+labs(x="Cover Type", y= "Frequency",title="Distribution of Cover Type per Wilderness Area")+
  scale_fill_discrete(name="Wilderness Area", labels=c("1 - Rawah","2 - Neota","3 - Comanche Peak","4 - Cache la Poudre"))

#Cover type per soil type distribution
ggplot(data=as.data.frame(table(forestCovCom$Cover_Type,forestCovCom$SoilType)),
       aes(x=reorder(Var2,Freq),y=log10(Freq),fill=Var1)) + 
  geom_bar(stat="identity") +coord_flip()+
  labs(x="Soil Type", y= "Log10(Frequency)",title="Distribution of Soil Type per Cover Type")+
  scale_fill_discrete(name="Cover Type")

ggplot(data=forestCovCom,aes(Cover_Type)) +geom_bar()

par(mfrow=c(2,3))
plot(forestCovCom$Cover_Type,forestCovCom$Elevation, xlab = "Cover Type",ylab="Elevation")
plot(forestCovCom$Cover_Type,forestCovCom$Horizontal_Distance_To_Roadways, xlab = "Cover Type",
     ylab="Horz. Dist. to Roadways")
plot(forestCovCom$Cover_Type,forestCovCom$Horizontal_Distance_To_Hydrology, xlab = "Cover Type",
     ylab="Horz. Dist. to Hydrology")
plot(forestCovCom$Cover_Type,forestCovCom$Hillshade_9am, xlab = "Cover Type",
     ylab="Hillshade - 9AM")
plot(forestCovCom$Cover_Type,forestCovCom$Hillshade_3pm, xlab = "Cover Type",
     ylab="HillShade - 3PM")
plot(forestCovCom$Cover_Type,forestCovDS$Horizontal_Distance_To_Fire_Points, xlab = "Cover Type",
     ylab="Horz. Dist. to Fire Pts")


featurePlot(forestCovCom, plot="pairs")


par(mfrow=c(2,2))
ggplot(data=forestCovDS,aes(Cover_Type)) + geom_point(aes(y=Elevation,col=Cover_Type))
ggplot(data=forestCovDS,aes(Cover_Type)) + geom_point(aes(y=Aspect,col=Cover_Type))