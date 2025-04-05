// Possible change: send filtered data instead. see /**/
// @@@
// To compile:
// replace the part after "-I" with the location of the folder containing the eemagine directory
// g++ -o stream_data.out stream_data.cc -ldl -pthread -std=c++11 -I /home/ncel/eego-SDK-1.3.19.40453/
// At runtime:
// args: (filename), IP (optional), PORT (optional)

#define EEGO_SDK_BIND_DYNAMIC // How to bind
#include <eemagine/sdk/factory.h> // SDK header
#include <eemagine/sdk/wrapper.cc> // Wrapper code to be compiled.

#include <iostream>
#include <thread>
#include <atomic>
#include <signal.h>
#include <fstream>
//#include <ifstream> // may not need fstream if using ifstream.
#include <string.h> // for memcpy, memmove. <string> might work also
#include <ctime>
#include <sys/types.h> // for mkdir
#include <sys/stat.h> // for mkdir
//#include <unistd.h> // for mkdir

#include <cmath>

// Networking libraries
#include <unistd.h>
#include <netinet/in.h>
#include <netinet/tcp.h> // for disabling nagle buffering
#include <sys/socket.h>
#include <arpa/inet.h>

#define IP_default "127.0.0.1"
#define PORT_default 7779

#define Fs 1000

#define nChannels 66 // don't change this. EEGO amplifier specific.
#define nElectrodes 64 //TODO: get this value from input instead of defining it here

using namespace eemagine::sdk;

std::atomic<bool> endStream(false);

void handleInterrupt(int signum) {
    std::cout << std::endl << "Caught signal with code " << signum << ", ending stream" << std::endl;
    endStream = true;
}

double dot(double *a, double *b, int length, int stridea=1, int strideb=1, bool flipb=false){
    double sum = 0.0;
    for(int i=0; i < length; i++) {
        if(!flipb)
            sum += a[i*stridea]*b[i*strideb];
        else
            sum += a[i*stridea]*b[(length-i-1)*strideb];
    }
    return sum;
}

int main(int argc, char **argv) {
    char* IP;
    int PORT;
    if(argc > 1) {
        IP = argv[1];
        if(argc > 2) {
            PORT = std::stoi(argv[2]);
        }
    } else {
        IP = (char*)IP_default;
        PORT = PORT_default;
    }
    //std::cout << argv[1] << '\n';

    signal(SIGINT, handleInterrupt); // allows the interrupt handler to capture CTRL-C events
    
    // Setup save files (x and y)
    std::time_t starttime = std::time(0);
    char timebuffer[100] = {0};
    struct tm starttime_local;
    // localtime_s(&starttime_local, &starttime); // may need to change to localtime_r for linux compatibility.
    localtime_r(&starttime, &starttime_local); // possibly this.
    // std::strftime(timebuffer, sizeof(timebuffer), "%Y-%m-%d_%H-%M-%S", std::localtime(&starttime)); // seems to work on the EEG machine though.
    std::strftime(timebuffer, sizeof(timebuffer), "%Y-%m-%d_%H-%M-%S", &starttime_local);
    std::string savefolder = "./data/";
    // make a new directory
    // see https://stackoverflow.com/questions/7430248/creating-a-new-directory-in-c
    struct stat st = {0};
    if (stat(savefolder.c_str(), &st) == -1) {
        mkdir(savefolder.c_str(), 0777); // check permissions, may want 0777
    }
    
    std::string filenameX = savefolder + "expX" + std::to_string(nElectrodes + 1) + "_";
    for (int i = 0; timebuffer[i] != '\0'; i++) {
        filenameX += timebuffer[i];
    }
    filenameX += ".bin";
    
    std::ofstream xFile;
    xFile.open (filenameX);
    
    double maxval = 1.0; // may need a larger number (reducing precision). possibly up to 4.0
    double savescale = std::pow(2.0, 15)/maxval; // for int16_t saving.
    
    // Create socket
    int sockfd;
    if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        std::cout << "Socket creation failed" << std::endl;
        return -1;
    }
    
    // remove nagle buffering
    // see https://stackoverflow.com/questions/31997648/where-do-i-set-tcp-nodelay-in-this-c-tcp-client
    int yes = 1;
    int sockopt_result = setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char *) &yes, sizeof(int)); // 1: on, 0: off
    

    // Fill in server information
    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);
    servaddr.sin_addr.s_addr = inet_addr(IP);

    std::cout << "connecting" << std::endl;
    // Connect to the server
    if (connect(sockfd, (struct sockaddr *) &servaddr, sizeof(servaddr)) < 0) {
        std::cout << "Failed to connect to socket" << std::endl;
        close(sockfd);
        return -1;
    }
    std::cout << "after connecting" << std::endl;

	// @@@
	// Point to the appropriate shared library file (may end with eego-SDK-1.3.19.40453/linux/64bit/libeego-SDK.so )
    // Initialize eeg stream
    factory fact("/home/ncel/eego-SDK-1.3.19.40453/libeego-SDK.so");
    amplifier* amp = fact.getAmplifier(); // Get an amplifier
    
    stream* eegStream = amp->OpenEegStream(Fs, 1.0); // The sampling rate is the only argument needed
    // sampling rate options: 500, 512, 1000, 1024, 2000, 2048, 4000, 4096, 8000, 8192, 16000, 16384
    
    int nSamples = 0;
    double sampleNo = 0;
    while (!endStream.load()) {
        buffer buf = eegStream->getData(); // Retrieve data from stream
        double * dataptr = buf.data();
        
        // Process and save new data
        for(int b = 0; b < buf.getSampleCount(); b++) {
            // Check data sample continuity in 66th channel
            if(nSamples == 0) {
                sampleNo = dataptr[65];
            } else {
                if(dataptr[65] != sampleNo + 1) {
                    std::cerr << "Missing " << (int)(dataptr[65] - sampleNo - 1);
                    std::cerr << " samples between sample numbers ";
                    std::cerr << (int)sampleNo << " and " << (int)dataptr[65] << '\n';
                }
                sampleNo = dataptr[65];
            }
            
            ssize_t bytes_sent = send(sockfd, &dataptr[0], nChannels*sizeof(double), 0);
            if (bytes_sent <= 0) {
                std::cout << "Error: no bytes sent over socket (" << bytes_sent;
                std::cout << "). Ending stream." <<  std::endl;
                endStream = true;
                break; // the socket has likely been closed
            }
                        
            // save data into respective files. May want to compress before saving.
            // save raw electrode data (64 channels)
            char * saveptr = (char *)&dataptr[0];
            for(int k = 0; k < nElectrodes*sizeof(double); k++)
                xFile << saveptr[k];
            // save sample No.
            saveptr = (char *)&dataptr[65];
            for(int k = 0; k < sizeof(double); k++)
                xFile << saveptr[k];       
            
            dataptr += nChannels;
            nSamples++;
            if(nSamples % 1000 == 0)
                std::cout << nSamples << std::endl;
        }
        
        if (buf.getSampleCount() > 1000.0 * 0.010 * 2)
            std::cout << "Got many samples: " << buf.getSampleCount() << std::endl;
        
        // Need to sleep less than 1s otherwise data may be lost
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    std::cout << "Stream ended. Data saved in: " << std::endl;
    std::cout << filenameX << std::endl;
    std::cout << "Total samples: " << nSamples << std::endl;

    // release Resources
    delete eegStream;
    delete amp;
    close(sockfd);
    return 0;
}
