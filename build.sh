javac -d build neuralNet.java 
cd build
jar cvfm "Character Recognition.jar" manifest.txt *.class 
cd .. 
mv "build/Character Recognition.jar" "Character Recognition.jar"
java -jar "Character Recognition.jar"
