JFLAGS = -g -cp .:. -d .
JC  = javac

.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	ClientRunnable.java \
	Main.java
default: classes

classes: $(CLASSES:.java=.class)
 
clean:
	$(RM) *.class