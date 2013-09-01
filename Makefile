HEADERS = chess.h switches.h MoveGeneratorBitboard.h perft_bb.h 
OBJECTS = randoms.o GlobalVars.o Magics.o UciInterface.o util.o perft.obj

default: perft_gpu

%.o: %.cpp $(HEADERS)
	g++ -c $< -o $@ -msse4.2 -Ofast

%.obj: %.cu $(HEADERS)
	nvcc -dc $< -o $@ -arch=sm_35 -O3 -Xcompiler -O3 

perft_gpu: $(OBJECTS)
	nvcc $(OBJECTS) -o $@ -arch=sm_35 -lcudadevrt -O3 -Xcompiler -O3
	-rm -f $(OBJECTS)

clean:
	-rm -f $(OBJECTS)
	-rm -f perft_gpu
	
