#!/bin/bash



javaJar="AlloyLogGenerator.jar"
normDeclName="xraynorm"
devDeclName="xraydev"
declEnd=".decl"
outStart="out_"
outEnd=".xes"

minTraceLength=15
maxTraceLength=20

normCountPer=150
devCountPer=150

normDeclCount=7
devDeclCount=1

# Generate positive cases

for i in `seq 1 $normDeclCount`;
do
	fullFilename="$normDeclName$i$declEnd"
	logFilename="$outStart$normDeclName$i$outEnd"
	yes | java --add-modules java.xml.bind -jar $javaJar $minTraceLength $maxTraceLength $normCountPer $fullFilename $logFilename -eld 
	#echo $fullFilename
	echo "Done with $logFilename"
done

# Generate negative cases
for i in `seq 1 $devDeclCount`;
do
	fullFilename="$devDeclName$i$declEnd"
	logFilename="$outStart$devDeclName$i$outEnd"
	yes | java --add-modules java.xml.bind -jar $javaJar $minTraceLength $maxTraceLength $devCountPer $fullFilename $logFilename -eld
	echo "Done with $logFilename"
done



#java --add-modules java.xml.bind -jar AlloyLogGenerator.jar 15 20 1000 xray.decl out.xes -eld -shuffle 2 -vacuity
