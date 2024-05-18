library(here)
library(IFC)

here::i_am("IdeasConv.R")

# Path to DAF ----
wd<-file.path(getwd())
data<-"/Ideas/file.daf"

file_D<-paste(wd,data, paste="")
file_D<-gsub(" ", "", file_D)

#ExtractFromDAF ----
daf<- ExtractFromDAF(fileName = file_D)

#ExportToFCS----
FCS<-ExportToFCS(daf,write_to = paste("FCS/file",".FCS",sep = "",collapse = NULL))
