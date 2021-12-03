
all:
	make -C encoder
	make -C decoder
	g++ centrify.cpp -o centrify

clean:
	make -C encoder clean
	make -C decoder clean
	rm -f centrify
