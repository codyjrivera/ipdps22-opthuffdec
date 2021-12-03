#include <iostream>
#include <fstream>
#include <cstdlib>

int main(int argc, char** argv) {
    using namespace std;
    ifstream inf(argv[1], ios_base::in | ios_base::binary);
    ofstream outf(argv[2], ios_base::out | ios_base::binary);
    int center = atoi(argv[3]);

    unsigned short code;
    unsigned char ccode;
    while (!inf.eof()) {
        inf.read((char*) &code, sizeof(unsigned short));

        int c = ((int) code) - ((center / 2) - 128);

        if (c > 0 && c <= 255) ccode = (unsigned char) c;
        else ccode = 0;
        
        outf.write((char*) &ccode, sizeof(unsigned char));
    }
    
    return 0;
}
