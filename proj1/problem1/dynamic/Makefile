JFLAGS = -g -cp .:. -d .
JC  = javac

.SUFFIXES: .java .class
.java.class:
	$(JC) $(JFLAGS) $*.java

CLASSES = \
	pc_dynamic.java
default: classes

classes: $(CLASSES:.java=.class)
 
clean:
	$(RM) *.class