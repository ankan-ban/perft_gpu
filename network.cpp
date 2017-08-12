// network.cpp: network related routines

#include "chess.h"

#include <stdlib.h>
#include <thread>
#include <mutex>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <time.h> 
#include <chrono>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>
#include <ifaddrs.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <linux/if_link.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>


void printIpAddress(char *addrString)
{
    struct ifaddrs *ifaddr, *ifa;
    int family, s, n;
    char host[NI_MAXHOST];

    if (getifaddrs(&ifaddr) == -1) {
        perror("getifaddrs");
        exit(EXIT_FAILURE);
    }

    /* Walk through linked list, maintaining head pointer so we
        can free list later */

    for (ifa = ifaddr, n = 0; ifa != NULL; ifa = ifa->ifa_next, n++) 
    {
        if (ifa->ifa_addr == NULL)
            continue;

        family = ifa->ifa_addr->sa_family;

        if (family == AF_INET)
        {
            s = getnameinfo(ifa->ifa_addr, sizeof(struct sockaddr_in),
                    host, NI_MAXHOST,
                    NULL, 0, NI_NUMERICHOST);
            if (strstr(ifa->ifa_name, "e"))
                sprintf(addrString, "%s", host);                                           
        }
    }

    freeifaddrs(ifaddr);
}

volatile bool sendingCompleteTT = false;

// max 1 million entires can be in flight
#define MAX_QUEUE_LENGTH (1024*1024)

struct NetworkWorkItem
{
    HashKey128b hash;
    uint64 perft;
    uint64 padding;        // might want to store something here
};
CT_ASSERT(sizeof(CompleteHashEntry) == 32);

NetworkWorkItem WorkQueue[MAX_QUEUE_LENGTH];
volatile uint32 put = 0;                     // position where to put a new work item
volatile uint32 get = MAX_QUEUE_LENGTH - 1;  // position from where the last work item was picked up

std::mutex queueCS;

std::mutex clientCS;

// happens on main thread
// puts a completed work item in workQueue
void enqueueWorkItem(CompleteHashEntry *item)
{
    // wait for space to be available on queue
    if (put == get)
    {
        auto t_start = std::chrono::high_resolution_clock::now();        
        while (put == get) 
        {
            auto t_end = std::chrono::high_resolution_clock::now();
            uint32 waitTime = (uint32) std::chrono::duration<double>(t_end-t_start).count();
            if (waitTime % 10 == 5)
            {
                printf("\nenqueueWorkItem waited for > %d seconds\n", waitTime);
                fflush(stdout);
                sleep(1);
            }
        }
    }

    queueCS.lock();     // to protect against multiple worker threads generating work items
    WorkQueue[put].hash  = item->hash;
    WorkQueue[put].perft = item->perft;
    put = (put + 1) % MAX_QUEUE_LENGTH;
    queueCS.unlock();
}


char myAddress[16];
uint32 myPort;
char myUID[32]; // address_port

#define MAX_NODES 128
char nodeIPs[MAX_NODES][16];
uint32 nodePorts[MAX_NODES];
bool reachable[MAX_NODES];

#define BROADCAST_PORT      0x4dab
#define COMPLETE_TT_PORT    0x1dab

int numNodes = 0;

void markAllReachable()
{
    for (int i=0;i<MAX_NODES;i++)
        reachable[i] = true;
}

int readNodeFile(FILE *fp)
{
    int n = 0;
    char line[1024];
    while(fgets(line, sizeof(line), fp))
    {
        sscanf(line, "%s %u", nodeIPs[n], &nodePorts[n]);
        n++;
    }
    return n;
}

void writeNodeFile(FILE *fp)
{
    for (int i=0;i<numNodes;i++)
        fprintf(fp, "%s %u\n", nodeIPs[i], nodePorts[i]);
}



std::thread networkThread, broadcasterThread, completeTTServerThread;
volatile bool networkThreadKillRequest = false;
volatile bool broadcasterThreadKillRequest = false;

// called periodically to update list of active nodes
void updateNodesList()
{
    int lfd;
    while((lfd = open("net.lock", O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) ;
    close(lfd);
    
    FILE *fp = fopen("nodes.txt", "rb+");
    if (fp)
    {
        numNodes = readNodeFile(fp);
        fclose(fp);
    }

    remove("net.lock");
}


void completeTTUpdateFromNetwork(HashKey128b hash, uint64 perft);
void lockCompleteTT();
void unlockCompleteTT();

// accessors for  transferring entire completeTT
extern CompleteHashEntry *completeTT;

extern CompleteHashEntry *chainMemoryChunks[1024];
extern int nChunks;
extern uint64 chainIndex;
extern uint64 completeTTSize;
extern uint64 chainMemorySize;


void allocChainMemoryChunk();


// read write 1MB chunks
#define NETWORK_CHUNK_SIZE (1024*1024)
#define min(a,b) ((a)<(b) ? (a) : (b))

int writeDataNetwork(int connfd, void *data, uint64 size)
{
    uint64 n = 0;
    char *buf = (char*) data;
    uint64 remaining = size;
    
    while(remaining)
    {
        uint64 chunk = min(NETWORK_CHUNK_SIZE, remaining);
        n = write(connfd, buf, chunk);
        if (n <= 0)
        {
            FILE *fplog = fopen(myUID, "ab+");
            fprintf(fplog, "Error writing to network: %s\n", strerror(errno));        
            fclose(fplog);
            printf("Error writing to network: %s\n", strerror(errno));
            fflush(stdout);
            return -1;
        }
        buf +=n;
        remaining -= n;
    }
    return 0;
}

int readDataNetwork(int sockfd, void *data, uint64 size)
{
    uint64 n = 0;
    char *buf = (char*) data;
    uint64 remaining = size;    
    
    while(remaining)
    {
        uint64 chunk = min(NETWORK_CHUNK_SIZE, remaining);        
        n = read(sockfd, buf, chunk);
        if (n <= 0)
        {
            const char *err = (n == 0) ? "Read ZERO bytes" : strerror(errno);
            FILE *fplog = fopen(myUID, "ab+");
            fprintf(fplog, "Error reading from network: %s\n", err);        
            fclose(fplog);
            printf("Error reading from network: %s\n", strerror(errno));
            fflush(stdout);
            return (n == 0) ? -2 : -1;
        }        
        buf +=n;
        remaining -= n;
    }
    return 0;
}


// network communication protocol
// client connects to server and sends:
// COMMAND <params> <data>
// two types of commands are supported
//  1    numWorkItems item0 item1 item2 ...             // For sending work items to server
//  2                                                   // for full TT copy request
//   - upon recieving TT copy request (2), server will send a uint64 which is no of ChainEntries, 
//     followed by main hash table part, followed by data in chain part

void broadcaster_thread_body()
{
    int counter = 0;
    //printf("broadcaster thread begins\n");    

    while(!broadcasterThreadKillRequest)
    {
        // send queued up work items to other nodes
        int firstItem = (get+1) % MAX_QUEUE_LENGTH;
        while(firstItem == put) 
        {
            usleep(1000);
            if(broadcasterThreadKillRequest)
                break;
        }

        if(broadcasterThreadKillRequest)
            break;
        
        int lastItem = (put-1);
        if (lastItem == -1) lastItem = MAX_QUEUE_LENGTH - 1;

        int numItems = 1;
        for (int item = firstItem; item != lastItem; item = (item +1) % MAX_QUEUE_LENGTH)
            numItems++;


        bool unreachableDetected = false;
        for (int i = 0; i < numNodes; i++)
        {
            if(strcmp(myAddress, nodeIPs[i]))
            {
                int sockfd = 0;
                struct sockaddr_in serv_addr = {}; 
                sockfd = socket(AF_INET, SOCK_STREAM, 0);
                assert(sockfd >= 0);
                serv_addr.sin_family = AF_INET;
                serv_addr.sin_port = htons(nodePorts[i]); 
                inet_pton(AF_INET, nodeIPs[i], &serv_addr.sin_addr);

                //clientCS.lock();
                int result = connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
                if (result < 0)
                {
                    if (reachable[i])
                    {
                        FILE *fplog = fopen(myUID, "ab+");
                        fprintf(fplog, "broadcaster thread had issues connecting to %s:%u, connect returned: %s\n", nodeIPs[i], nodePorts[i], strerror(errno));        
                        fclose(fplog);
                    }
                    
                    // some error in connecting, mark un-reachable
                    reachable[i] = false;
                    unreachableDetected = true;
                }
                else
                {
                    reachable[i] = true;
                    uint32 command = 1;

                    int n = write(sockfd, &command, sizeof(uint32));
                    if (n<=0)
                    {
                        printf("\nerror writing command for broadcasting work items\n");
                        fflush(stdout);
                        //exit(0);
                        close(sockfd);
                        continue;
                    }
                    
                    n = write(sockfd, &numItems, sizeof(uint32));
                    if (n<=0)
                    {
                        printf("\nerror writing numItems for broadcasting work items\n");
                        fflush(stdout);
                        //exit(0);
                        close(sockfd);
                        continue;
                    }
                    bool caughtError = false;
                    for (int item = firstItem; item != lastItem; item = (item +1) % MAX_QUEUE_LENGTH)
                    {
                        n = write(sockfd, &WorkQueue[item], sizeof(NetworkWorkItem));
                        if (n<=0)
                        {
                            printf("\nerror writing work items when broadcasting work items, node: %s, error: %s\n", nodeIPs[i], strerror(errno));
                            fflush(stdout);
                            //exit(0);
                            close(sockfd);
                            caughtError = true;
                            break;
                        }
                    }
                    if (caughtError)
                        continue;

                    n = write(sockfd, &WorkQueue[lastItem], sizeof(NetworkWorkItem));       
                    if (n<=0)
                    {
                        printf("\nerror writing last work item when broadcasting work items, node: %s, error: %s\n", nodeIPs[i], strerror(errno));
                        fflush(stdout);
                        //exit(0);
                        close(sockfd);
                        continue;
                    }
                    
                    close(sockfd);
                }
                //clientCS.unlock();                
            }
        }

        // TODO: check if this is robust enough!
        //  6 Aug 2017: seems robust enough... (unless lot of job instances are scheduled together)
#if 1
        if (unreachableDetected)
        {
            int lfd;
            while((lfd = open("net.lock", O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) ;
            close(lfd);
            
            // remove unreachable nodes from list
            char oldIPs[MAX_NODES][16];
            uint32 oldPorts[MAX_NODES];
            memcpy(oldIPs, nodeIPs, sizeof(oldIPs));
            memcpy(oldPorts, nodePorts, sizeof(oldPorts));
            int newCount = 0;
            for (int i = 0; i < numNodes; i++)
                if (reachable[i])
                {
                    reachable[newCount] = true;
                    strcpy(nodeIPs[newCount], oldIPs[i]);
                    nodePorts[newCount++] = oldPorts[i];
                }
            numNodes = newCount;            

            // write updated list to file
            FILE *fp = fopen("nodes.txt", "wb+");
            writeNodeFile(fp);
            fclose(fp);

            remove("net.lock");
            markAllReachable();
        }
#endif
        get = lastItem;
        usleep(10000);  // wait for 10 ms
        counter++;
        if (counter % 100 == 0)
        {
            // update list of active nodes every second
            updateNodesList();
        }
    }

    //printf("broadcaster thread ends\n");
    broadcasterThreadKillRequest = false;    
}

#if 0
// the function below relies on the first two fields to be at same locations!
CT_ASSERT(sizeof(NetworkWorkItem) == sizeof(CompleteHashEntry));
#define STAGING_ITEMS (8*1024)

void completeTTServer(struct sockaddr_in client_addr)
{
    char clientip[32]; clientip[0] = 0;
    inet_ntop(AF_INET, &client_addr.sin_addr, clientip, sizeof(clientip));
    FILE *fplog = fopen(myUID, "ab+");
    fprintf(fplog, "Got complete TT request from: %s\n", clientip);
    fclose(fplog);

    struct sockaddr_in serv_addr = {}; 
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(myPort);
    inet_pton(AF_INET, clientip, &serv_addr.sin_addr);
    
    sleep(10);  // wait for 10 seconds before starting. Let the new node initialize and get ready for work!

    auto t_start = std::chrono::high_resolution_clock::now();

    uint32 numElements = completeTTSize / sizeof(CompleteHashEntry);
    for(int i=0; i<numElements;i+=STAGING_ITEMS)
    {
        int sockfd = 0;
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        assert(sockfd >= 0);
        
        //clientCS.lock();
        int result = connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
        if (result < 0)
        {
            FILE *fplog = fopen(myUID, "ab+");
            fprintf(fplog, "Error sending complete TT: %s\n", strerror(errno));
            fclose(fplog);
            return;
        }
        else
        {
            uint32 command = 1;
            uint32 numItems = STAGING_ITEMS;
            write(sockfd, &command, sizeof(uint32));
            write(sockfd, &numItems, sizeof(uint32));
            write(sockfd, &completeTT[i], sizeof(NetworkWorkItem)*STAGING_ITEMS);
            close(sockfd);
        }
        //clientCS.unlock();
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double transferTime = std::chrono::duration<double>(t_end-t_start).count();

    fplog = fopen(myUID, "ab+");
    fprintf(fplog, "Complete TT sending complete, time taken: %g seconds observed network bandwidth: %g MBps\n", 
                    transferTime, (completeTTSize/* + nChunks * chainMemorySize*/)/(1024*1024*transferTime));
    fclose(fplog);
}
#endif

#if 1
void completeTTServer()
{
    // open a socket and start listening for clients
    int listenfd = 0, connfd = 0;
    struct sockaddr_in serv_addr = {}; 
    listenfd = socket(AF_INET, SOCK_STREAM, 0);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(COMPLETE_TT_PORT);

    int result = bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)); 
    if (result < 0)
    {
        printf("some error at bind() call in completeTTServer: %d\n", result);
        exit(-1);
    }
    
    result = listen(listenfd, 32); 
    if (result < 0)
    {
        printf("some error at listen() call in completeTTServer: %d\n", result);
        exit(-1);
    }

    while(1)
    {
        struct sockaddr_in clientAddr = {};
        socklen_t addrLen = sizeof(clientAddr);

        FILE *fplog = fopen(myUID, "ab+");
        fprintf(fplog, "TT Server: Waiting for new connection...\n");
        fclose(fplog);

        connfd = accept(listenfd, (sockaddr*) &clientAddr, &addrLen); 

        if (connfd < 0)
        {
            printf("some error after accept() call\n");
            exit(-1);
        }

        // send ENTIRE complete TT to client
        char clientip[32]; clientip[0] = 0;
        inet_ntop(AF_INET, &clientAddr.sin_addr, clientip, sizeof(clientip));
        fplog = fopen(myUID, "ab+");
        fprintf(fplog, "Got complete TT request from: %s\n", clientip);
        fclose(fplog);
        
        // 1. take a lock on completeTT to make sure nobody updates it
        sendingCompleteTT = true;
        lockCompleteTT();

        // 2. write first the hash table part of the complete TT and then the chain part
        auto t_start = std::chrono::high_resolution_clock::now();
        int n = write(connfd, &nChunks, sizeof(int));
        if (n<=0)
        {
            printf("\nerror writing value of nChunks when sending complete TT\n");
            fflush(stdout);
            //exit(0);
            unlockCompleteTT();
            sendingCompleteTT = false;
            close(connfd);
            continue;
        }

        n = write(connfd, &chainIndex, sizeof(uint64));
        if (n<=0)
        {
            printf("\nerror writing value of chainIndex when sending complete TT\n");
            fflush(stdout);
            //exit(0);
            unlockCompleteTT();
            sendingCompleteTT = false;
            close(connfd);
            continue;
        }

        n = writeDataNetwork(connfd, completeTT, completeTTSize);
        if (n<0)
        {
            unlockCompleteTT();
            sendingCompleteTT = false;
            close(connfd);
            continue;
        }
        bool gotError = false;
        for (int i=0;i<nChunks;i++)
        {
            n = writeDataNetwork(connfd, chainMemoryChunks[i], chainMemorySize);
            if (n < 0)
            {
                unlockCompleteTT();
                sendingCompleteTT = false;
                close(connfd);
                gotError = true;
                break;
            }
        }
        if (gotError)
            continue;

        auto t_end = std::chrono::high_resolution_clock::now();
        double transferTime = std::chrono::duration<double>(t_end-t_start).count();

        unlockCompleteTT();
        sendingCompleteTT = false;

        fplog = fopen(myUID, "ab+");
        fprintf(fplog, "Complete TT send complete, time taken: %g seconds observed network bandwidth: %g MBps\n", 
                        transferTime, (completeTTSize + nChunks * chainMemorySize)/(1024*1024*transferTime));
        fclose(fplog);

        close(connfd);
    }

    close(listenfd);
}
#endif

// main network thread responsible for sharing work items across multiple nodes on network
// 1. adds own ip address to the file containing list of nodes
// 2. wait for clients to connect
//     - once connected, get new work items from other nodes and update in complete TT tables
void network_thread_body()
{
    // 1. Add own ip address to list of nodes
    //printIpAddress(myAddress);  // this doesn't work inside container :-/


    // open a socket and start listening for clients
    int listenfd = 0, connfd = 0;
    struct sockaddr_in serv_addr = {}; 
    listenfd = socket(AF_INET, SOCK_STREAM, 0);
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(myPort);

    FILE *fplog = fopen(myUID, "ab+");
    int result = bind(listenfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)); 
    fprintf(fplog, "bind returned: %d\n", result);
    result = listen(listenfd, 32); 
    fprintf(fplog, "listen returned: %d\n", result);
    fprintf(fplog, "num of nodes found: %d\n", numNodes);    
    fclose(fplog);


    // start broadcster thread
    broadcasterThread       = std::thread(broadcaster_thread_body);

    completeTTServerThread = std::thread(completeTTServer);

    // write updated list to file
    int lfd;
    while((lfd = open("net.lock", O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) ;
    close(lfd);
    FILE *fp = fopen("nodes.txt", "wb+");
    writeNodeFile(fp);
    fclose(fp);            
    remove("net.lock");

    while(!networkThreadKillRequest)
    {
        struct sockaddr_in clientAddr = {};
        socklen_t addrLen = sizeof(clientAddr);
        connfd = accept(listenfd, (sockaddr*) &clientAddr, &addrLen); 
        uint32 command = 0;
        read(connfd, &command, sizeof(uint32));

        if (command == 1)
        {
            // update work items in local TT from client node
            uint32 items = 0;
            read(connfd, &items, sizeof(uint32));
            for (int i=0;i<items;i++)
            {
                NetworkWorkItem workItem = {};
                read(connfd, &workItem, sizeof(NetworkWorkItem));
                if (!sendingCompleteTT)
                {
                    // avoid blocking the network
                    completeTTUpdateFromNetwork(workItem.hash, workItem.perft);
                }
            }
        }
        else if (command == 2)
        {
            // start complete TT server thread (that sends the entire TT piece by piece to the client who requested it)
            //completeTTServerThread  = std::thread(completeTTServer, clientAddr);
            
        }
        else if (command == 3)
        {
            // exit request
        }
        else if (command == 4)
        {
            // get my IP address!
            char clientip[32]; clientip[0] = 0;
            inet_ntop(AF_INET, &clientAddr.sin_addr, clientip, sizeof(clientip));
            int n = write(connfd, clientip, sizeof(clientip));
            if (n <= 0)
            {
                printf("\nerror writing client IP\n");
                fflush(stdout);
                //exit(0);
            }

            FILE *fplog = fopen(myUID, "ab+");
            fprintf(fplog, "Got ip request from: %s\n", clientip);
            fclose(fplog);
        }
        close(connfd);                    
    }
    close (listenfd);

    //printf("waiting for broadcaster thread to die\n");
    while(broadcasterThreadKillRequest) ;    
    broadcasterThread.join();

    networkThreadKillRequest = false;
}

#include <signal.h>
/* Catch Signal Handler functio */
void signal_callback_handler(int signum)
{

    printf("Caught signal: %d\n", signum);
}


void createNetworkThread()
{
    signal(SIGPIPE, signal_callback_handler);
    signal(SIGHUP,  signal_callback_handler);


    numNodes = 0;
    updateNodesList();

    if (numNodes > 0)
    {
        // get own IP address from an existing node
        int sockfd = 0;
        struct sockaddr_in serv_addr = {}; 
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        assert(sockfd >= 0);
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(nodePorts[0]); 
        inet_pton(AF_INET, nodeIPs[0], &serv_addr.sin_addr);

        int result = connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
        uint32 command = 4;
        int n = write(sockfd, &command, sizeof(uint32));
        if (n<=0)
        {
            printf("\nerror writing command for getting ip address\n");
            fflush(stdout);
            exit(0);
        }
        char buf[32];
        read(sockfd, buf, sizeof(buf));
        strcpy(myAddress, buf);
        close(sockfd);
    }
    else
    {
        // hack! get ip address from a file where it's manually entered!
        FILE *fp;
        while(!(fp = fopen("ip.txt", "rb+"))) 
            usleep(100000); // 100ms

        char line[1024];
        fgets(line, sizeof(line), fp);
        fclose(fp);
        remove("ip.txt");
        strncpy(myAddress, line, strlen(line)-1);
    }

    // add current node to the list
    reachable[numNodes] = true;
    strcpy(nodeIPs[numNodes], myAddress);
    myPort = BROADCAST_PORT;
    nodePorts[numNodes] = myPort;
    numNodes++;
    sprintf(myUID, "%s_%u", myAddress, myPort);

    
    if (numNodes > 1)
    {
        // ask server (first already running node) to send complete TT
        int sockfd = 0;
        struct sockaddr_in serv_addr = {}; 
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        assert(sockfd >= 0);
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(COMPLETE_TT_PORT); 
        inet_pton(AF_INET, nodeIPs[0], &serv_addr.sin_addr);

        int result = connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
        //uint32 command = 2;
        //write(sockfd, &command, sizeof(uint32));

        if (result < 0)
        {
            FILE *fplog = fopen(myUID, "ab+");
            fprintf(fplog, "newly started node had issues getting complete TT from %s:%u, connect returned: %s\n", nodeIPs[0], COMPLETE_TT_PORT, strerror(errno));        
            fclose(fplog);
        }
        else
        {
            // read first the hash table part of the complete TT and then the chain part
            auto t_start = std::chrono::high_resolution_clock::now();

            int incomingChunks = 0;
            read(sockfd, &incomingChunks, sizeof(int));

            uint64 incomingChainIndex = 0;
            read(sockfd, &incomingChainIndex, sizeof(uint64));
            
            FILE *fplog = fopen(myUID, "ab+");
            fprintf(fplog, "no of incoming chunks: %d, chainIndex: %llu\n", incomingChunks, incomingChainIndex);
            fclose(fplog);
            
            int n = readDataNetwork(sockfd, completeTT, completeTTSize);
            if (n < 0)
            {
                exit(0);
            }


            for (int i=nChunks;i<incomingChunks;i++)
            {
                allocChainMemoryChunk();
            }

            for (int i=0;i<incomingChunks;i++)
            {
                n = readDataNetwork(sockfd, chainMemoryChunks[i], chainMemorySize);
                if (n == -2)
                {
                    // HACK! server not willing to give any more chunks. Stop anyway :-/
                    incomingChunks = i;
                    incomingChainIndex = 0;
                    printf("\nServer didn't give me all chunks :-/\n");
                    break;
                }
                if (n < 0)
                {
                    exit(0);
                }
            }

            nChunks = incomingChunks;
            chainIndex = incomingChainIndex;            

            auto t_end = std::chrono::high_resolution_clock::now();
            double transferTime = std::chrono::duration<double>(t_end-t_start).count();

            fplog = fopen(myUID, "ab+");
            fprintf(fplog, "Complete TT recieve complete, time taken: %g seconds observed network bandwidth: %g MBps\n", 
                            transferTime, (completeTTSize + nChunks * chainMemorySize)/(1024*1024*transferTime));
            fclose(fplog);

            close(sockfd);
        }
    }

    networkThread = std::thread(network_thread_body);

    printf("\nnetwork setup done.\n");
    fflush(stdout);
}

void endNetworkThread()
{
    // kill broadcaster thread
    broadcasterThreadKillRequest = true;
    networkThreadKillRequest = true;

    // request own network server thread to exit
    {
        int sockfd = 0;
        struct sockaddr_in serv_addr = {}; 
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        assert(sockfd >= 0);
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(myPort); 
        inet_pton(AF_INET, myAddress, &serv_addr.sin_addr);

        int result = connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
        uint32 command = 3;
        write(sockfd, &command, sizeof(uint32));
        close(sockfd);
    }

    //printf("waiting for main network thread to die\n");
    while(networkThreadKillRequest) ;
    networkThread.join();
}





//  code that didn't work or no longer used!



#if 0
    uint32 existingPort = 0x1dab;
    for (int i=0;i<numNodes;i++)
    {
        reachable[i] = true;
        if (nodePorts[i] > existingPort)
            existingPort = nodePorts[i];
    }
#endif


#if 0
    //set master socket to allow multiple connections , 
    int opt = 1;
    if( setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, (char *)&opt, 
          sizeof(opt)) < 0 )  
    {  
        printf("coundn't set socket option for server socket\n");
        exit(EXIT_FAILURE);
    } 
#endif

    // get entire complete TT from the first node
#if 0
    if (numNodes > 1)
    {
        FILE *fplog = fopen(myUID, "ab+");
        fprintf(fplog, "Getting complete TT from %s:%u\n", nodeIPs[0], nodePorts[0]);
        fclose(fplog);

        int sockfd = 0;
        struct sockaddr_in serv_addr = {}; 
        sockfd = socket(AF_INET, SOCK_STREAM, 0);
        assert(sockfd >= 0);
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(COMPLETE_TT_PORT); 
        inet_pton(AF_INET, nodeIPs[0], &serv_addr.sin_addr);

        int result = connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
        if (result < 0)
        {
            FILE *fplog = fopen(myUID, "ab+");
            fprintf(fplog, "newly started node had issues getting complete TT from %s:%u, connect returned: %s\n", nodeIPs[0], COMPLETE_TT_PORT, strerror(errno));        
            fclose(fplog);
        }
        else
        {
            // 1. take a lock on completeTT to make sure nobody updates it
            lockCompleteTT();

            // 2. read first the hash table part of the complete TT and then the chain part
            auto t_start = std::chrono::high_resolution_clock::now();

            int incomingChunks = 0;
            read(sockfd, &incomingChunks, sizeof(int));

            fplog = fopen(myUID, "ab+");
            fprintf(fplog, "no of incoming chunks: %d\n", incomingChunks);
            fclose(fplog);
            
            readDataNetwork(sockfd, completeTT, completeTTSize);


            for (int i=nChunks;i<incomingChunks;i++)
            {
                allocChainMemoryChunk();
            }

            for (int i=0;i<incomingChunks;i++)
            {
                readDataNetwork(sockfd, chainMemoryChunks[i], chainMemorySize);
            }
            unlockCompleteTT();

            auto t_end = std::chrono::high_resolution_clock::now();
            double transferTime = std::chrono::duration<double>(t_end-t_start).count();

            fplog = fopen(myUID, "ab+");
            fprintf(fplog, "Complete TT recieve complete, time taken: %g seconds observed network bandwidth: %g MBps\n", 
                            transferTime, (completeTTSize + nChunks * chainMemorySize)/(1024*1024*transferTime));
            fclose(fplog);

            close(sockfd);
        }
    }
#endif
