# This file is part of libDAI - http://www.libdai.org/
#
# libDAI is licensed under the terms of the GNU General Public License version
# 2, or (at your option) any later version. libDAI is distributed without any
# warranty. See the file COPYING for more details.
#
# Copyright (C) 2006-2010  Joris Mooij  [joris dot mooij at libdai dot org]
# Copyright (C) 2006-2007  Radboud University Nijmegen, The Netherlands


# Load the platform independent build configuration file
include Makefile.ALL

# Load the local configuration from Makefile.conf
include Makefile.conf

# Set version and date
DAI_VERSION="git HEAD"
DAI_DATE="April 1, 2010 - or later"

# Directories of libDAI sources
# Location libDAI headers
INC=include/dai
# Location of libDAI source files
SRC=src
# Destination directory of libDAI library
LIB=lib

# Set final compiler flags
ifdef DEBUG
  CCFLAGS:=$(CCFLAGS) $(CCDEBUGFLAGS)
else
  CCFLAGS:=$(CCFLAGS) $(CCNODEBUGFLAGS)
endif

# Define build targets
TARGETS:=tests utils lib examples
ifneq ($(OS),WINDOWS)
  TARGETS:=$(TARGETS) unittests
endif
TARGETS:=$(TARGETS) testregression testem
ifdef WITH_DOC
  TARGETS:=$(TARGETS) doc
endif
ifdef WITH_MATLAB
  TARGETS:=$(TARGETS) matlabs
endif

# Define conditional build targets
NAMES:=bipgraph graph varset daialg alldai clustergraph factor factorgraph properties regiongraph util weightedgraph exceptions exactinf evidence emalg
ifdef WITH_BP
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_BP
  NAMES:=$(NAMES) bp
endif
ifdef WITH_FBP
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_FBP
  NAMES:=$(NAMES) fbp
endif
ifdef WITH_TRWBP
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_TRWBP
  NAMES:=$(NAMES) trwbp
endif
ifdef WITH_MF
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_MF
  NAMES:=$(NAMES) mf
endif
ifdef WITH_HAK
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_HAK
  NAMES:=$(NAMES) hak
endif
ifdef WITH_LC
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_LC
  NAMES:=$(NAMES) lc
endif
ifdef WITH_TREEEP
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_TREEEP
  NAMES:=$(NAMES) treeep
endif
ifdef WITH_JTREE
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_JTREE
  NAMES:=$(NAMES) jtree
endif
ifdef WITH_MR
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_MR
  NAMES:=$(NAMES) mr
endif
ifdef WITH_GIBBS
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_GIBBS
  NAMES:=$(NAMES) gibbs
endif
ifdef WITH_CBP
  WITHFLAGS:=$(WITHFLAGS) -DDAI_WITH_CBP
  NAMES:=$(NAMES) bbp cbp bp_dual
endif
ifdef DAI_SPARSE
  WITHFLAGS:=$(WITHFLAGS) -DDAI_SPARSE=$(DAI_SPARSE)
endif

# Define standard libDAI header dependencies, source file names and object file names
HEADERS=$(foreach name,bipgraph graph spvector spvector_map index var factor factorsp varset smallset fo prob probsp daialg properties alldai enum exceptions util,$(INC)/$(name).h)
SOURCES:=$(foreach name,$(NAMES),$(SRC)/$(name).cpp)
OBJECTS:=$(foreach name,$(NAMES),$(name)$(OE))

# Setup final command for C++ compiler
ifneq ($(OS),WINDOWS)
  CC:=$(CC) $(CCINC) $(CCFLAGS) $(WITHFLAGS) $(CCLIB)
else
  CC:=$(CC) $(CCINC) $(CCFLAGS) $(WITHFLAGS)
  LIBS:=$(LIBS) $(CCLIB)
endif

# Setup final command for MEX
ifdef NEW_MATLAB
  MEXFLAGS:=$(MEXFLAGS) -largeArrayDims
else
  MEXFLAGS:=$(MEXFLAGS) -DSMALLMEM
endif
MEX:=$(MEX) $(MEXINC) $(MEXFLAGS) $(WITHFLAGS) $(MEXLIB)


# META TARGETS
###############

all : $(TARGETS)

examples : examples/example$(EE) examples/example_bipgraph$(EE) examples/example_varset$(EE) examples/example_permute$(EE) examples/example_sprinkler$(EE) examples/example_sprinkler_gibbs$(EE) examples/example_sprinkler_em$(EE)

matlabs : matlab/dai$(ME) matlab/dai_readfg$(ME) matlab/dai_writefg$(ME) matlab/dai_potstrength$(ME)

unittests : tests/unit/var$(EE) tests/unit/smallset$(EE) tests/unit/varset$(EE) tests/unit/graph$(EE) tests/unit/bipgraph$(EE) tests/unit/weightedgraph$(EE) tests/unit/enum$(EE) tests/unit/enum$(EE) tests/unit/util$(EE) tests/unit/properties$(EE) tests/unit/index$(EE) tests/unit/prob$(EE) tests/unit/factor$(EE)
	echo Running unit tests...
	tests/unit/var$(EE)
	tests/unit/smallset$(EE)
	tests/unit/varset$(EE)
	tests/unit/graph$(EE)
	tests/unit/bipgraph$(EE)
	tests/unit/weightedgraph$(EE)
	tests/unit/enum$(EE)
	tests/unit/util$(EE)
	tests/unit/properties$(EE)
	tests/unit/index$(EE)
	tests/unit/prob$(EE)
	tests/unit/factor$(EE)

tests : tests/testdai$(EE) tests/testem/testem$(EE) tests/testbbp$(EE) $(unittests)

utils : utils/createfg$(EE) utils/fg2dot$(EE) utils/fginfo$(EE)

lib: $(LIB)/libdai$(LE)


# OBJECTS
##########

%$(OE) : $(SRC)/%.cpp $(INC)/%.h $(HEADERS)
	$(CC) -c $<

bbp$(OE) : $(SRC)/bbp.cpp $(INC)/bbp.h $(INC)/bp_dual.h $(HEADERS)
	$(CC) -c $<

cbp$(OE) : $(SRC)/cbp.cpp $(INC)/cbp.h $(INC)/bbp.h $(INC)/bp_dual.h $(HEADERS)
	$(CC) -c $<

hak$(OE) : $(SRC)/hak.cpp $(INC)/hak.h $(HEADERS) $(INC)/regiongraph.h
	$(CC) -c $<

jtree$(OE) : $(SRC)/jtree.cpp $(INC)/jtree.h $(HEADERS) $(INC)/weightedgraph.h $(INC)/clustergraph.h $(INC)/regiongraph.h
	$(CC) -c $<

treeep$(OE) : $(SRC)/treeep.cpp $(INC)/treeep.h $(HEADERS) $(INC)/weightedgraph.h $(INC)/clustergraph.h $(INC)/regiongraph.h $(INC)/jtree.h
	$(CC) -c $<

emalg$(OE) : $(SRC)/emalg.cpp $(INC)/emalg.h $(INC)/evidence.h $(HEADERS)
	$(CC) -c $<


# EXAMPLES
###########

examples/%$(EE) : examples/%.cpp $(HEADERS) $(LIB)/libdai$(LE)
	$(CC) $(CCO)$@ $< $(LIBS)


# UNIT TESTS
#############

tests/unit/%$(EE) : tests/unit/%.cpp $(HEADERS) $(LIB)/libdai$(LE)
	$(CC) $(CCO)$@ $< $(LIBS) $(BOOSTLIBS_UTF)


# TESTS
########

tests/testdai$(EE) : tests/testdai.cpp $(HEADERS) $(LIB)/libdai$(LE)
	$(CC) $(CCO)$@ $< $(LIBS) $(BOOSTLIBS_PO)
tests/testem/testem$(EE) : tests/testem/testem.cpp $(HEADERS) $(LIB)/libdai$(LE)
	$(CC) $(CCO)$@ $< $(LIBS) $(BOOSTLIBS_PO)
ifdef WITH_CBP
tests/testbbp$(EE) : tests/testbbp.cpp $(HEADERS) $(LIB)/libdai$(LE)
	$(CC) $(CCO)$@ $< $(LIBS)
endif


# MATLAB INTERFACE
###################

matlab/dai$(ME) : $(SRC)/matlab/dai.cpp $(HEADERS) $(SOURCES) $(SRC)/matlab/matlab.cpp
	$(MEX) -output $@ $< $(SRC)/matlab/matlab.cpp $(SOURCES)

matlab/dai_readfg$(ME) : $(SRC)/matlab/dai_readfg.cpp $(HEADERS) $(SRC)/matlab/matlab.cpp $(SRC)/factorgraph.cpp $(SRC)/exceptions.cpp $(SRC)/bipgraph.cpp
	$(MEX) -output $@ $< $(SRC)/matlab/matlab.cpp $(SRC)/factorgraph.cpp $(SRC)/exceptions.cpp $(SRC)/bipgraph.cpp

matlab/dai_writefg$(ME) : $(SRC)/matlab/dai_writefg.cpp $(HEADERS) $(SRC)/matlab/matlab.cpp $(SRC)/factorgraph.cpp $(SRC)/exceptions.cpp $(SRC)/bipgraph.cpp
	$(MEX) -output $@ $< $(SRC)/matlab/matlab.cpp $(SRC)/factorgraph.cpp $(SRC)/exceptions.cpp $(SRC)/bipgraph.cpp

matlab/dai_potstrength$(ME) : $(SRC)/matlab/dai_potstrength.cpp $(HEADERS) $(SRC)/matlab/matlab.cpp $(SRC)/exceptions.cpp
	$(MEX) -output $@ $< $(SRC)/matlab/matlab.cpp $(SRC)/exceptions.cpp


# UTILS
########

utils/createfg$(EE) : utils/createfg.cpp $(HEADERS) $(LIB)/libdai$(LE)
	$(CC) $(CCO)$@ $< $(LIBS) $(BOOSTLIBS_PO)

utils/fg2dot$(EE) : utils/fg2dot.cpp $(HEADERS) $(LIB)/libdai$(LE)
	$(CC) $(CCO)$@ $< $(LIBS)

utils/fginfo$(EE) : utils/fginfo.cpp $(HEADERS) $(LIB)/libdai$(LE)
	$(CC) $(CCO)$@ $< $(LIBS)


# LIBRARY
##########

ifneq ($(OS),WINDOWS)
$(LIB)/libdai$(LE) : $(OBJECTS)
	-mkdir -p lib
	ar rcus $(LIB)/libdai$(LE) $(OBJECTS)
else
$(LIB)/libdai$(LE) : $(OBJECTS)
	-mkdir lib
	lib /out:$(LIB)/libdai$(LE) $(OBJECTS)
endif


# REGRESSION TESTS
###################

testregression : tests/testdai$(EE)
	@echo Starting regression test...this can take a minute or so!
ifneq ($(OS),WINDOWS)
	cd tests && ./testregression && cd ..
else
	cd tests && testregression.bat && cd ..
endif

testem : tests/testem/testem$(EE)
	@echo Starting EM tests
ifneq ($(OS),WINDOWS)
	cd tests/testem && ./runtests && cd ../..
else
	cd tests\testem && runtests && cd ..\..
endif


# DOCUMENTATION
################

doc : $(INC)/*.h $(SRC)/*.cpp examples/*.cpp doxygen.conf
	doxygen doxygen.conf

README : doc scripts/makeREADME
	DAI_VERSION=$(DAI_VERSION) DAI_DATE=$(DAI_DATE) scripts/makeREADME

TAGS :
	etags src/*.cpp include/dai/*.h tests/*.cpp utils/*.cpp
	ctags src/*.cpp include/dai/*.h tests/*.cpp utils/*.cpp


# CLEAN
########

ifneq ($(OS),WINDOWS)
.PHONY : clean
clean :
	-rm $(OBJECTS)
	-rm matlab/*$(ME)
	-rm examples/example$(EE) examples/example_bipgraph$(EE) examples/example_varset$(EE) examples/example_permute$(EE) examples/example_sprinkler$(EE) examples/example_sprinkler_gibbs$(EE) examples/example_sprinkler_em$(EE)
	-rm tests/testdai$(EE) tests/testem/testem$(EE) tests/testbbp$(EE)
	-rm tests/unit/var$(EE) tests/unit/smallset$(EE) tests/unit/varset$(EE) tests/unit/graph$(EE) tests/unit/bipgraph$(EE) tests/unit/weightedgraph$(EE) tests/unit/enum$(EE) tests/unit/util$(EE) tests/unit/properties$(EE) tests/unit/index$(EE) tests/unit/prob$(EE) tests/unit/factor$(EE)
	-rm utils/fg2dot$(EE) utils/createfg$(EE) utils/fginfo$(EE)
	-rm -R doc
	-rm -R lib
else
.PHONY : clean
clean :
	-del *.obj
	-del *.ilk
	-del *.pdb
	-del matlab\*$(ME)
	-del examples\*$(EE)
	-del examples\*$(EE).manifest
	-del examples\*.ilk
	-del examples\*.pdb
	-del tests\testdai$(EE)
	-del tests\testbbp$(EE)
	-del tests\testdai$(EE).manifest
	-del tests\testbbp$(EE).manifest
	-del tests\testem\testem$(EE)
	-del tests\testem\testem$(EE).manifest
	-del tests\*.pdb
	-del tests\*.ilk
	-del tests\testem\*.pdb
	-del tests\testem\*.ilk
	-del utils\*$(EE)
	-del utils\*$(EE).manifest
	-del utils\*.pdb
	-del utils\*.ilk
	-del tests\unit\*.ilk
	-del tests\unit\*.pdb
	-del tests\unit\var$(EE)
	-del tests\unit\smallset$(EE)
	-del tests\unit\varset$(EE)
	-del tests\unit\graph$(EE)
	-del tests\unit\bipgraph$(EE)
	-del tests\unit\weightedgraph$(EE)
	-del tests\unit\enum$(EE)
	-del tests\unit\util$(EE)
	-del tests\unit\properties$(EE)
	-del tests\unit\index$(EE)
	-del tests\unit\prob$(EE)
	-del tests\unit\factor$(EE)
	-del $(LIB)\libdai$(LE)
	-rmdir lib
endif
