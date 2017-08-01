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

// happens on main thread
// puts a completed work item in workQueue
void enqueueWorkItem(CompleteHashEntry *item)
{
    // wait for space to be available on queue
    while (put == get) ;

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

int numNodes = 0;

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



std::thread networkThread, broadcasterThread;
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
extern CompleteHashEntry *chainMemory;
extern uint64 completeTTSize;
extern uint64 chainMemorySize;


void writeDataNetwork(int connfd, void *data, uint64 size)
{
    uint64 n = 0;
    char *buf = (char*) data;
    uint64 remaining = size;
    
    while(remaining)
    {
        n = write(connfd, buf, remaining);
        buf +=n;
        remaining -= n;
    }
}

void readDataNetwork(int sockfd, void *data, uint64 size)
{
    uint64 n = 0;
    char *buf = (char*) data;
    uint64 remaining = size;    
    
    while(remaining)
    {
        n = read(sockfd, buf, remaining);
        buf +=n;
        remaining -= n;
    }
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
            if(strcmp(myAddress, nodeIPs[i]) || nodePorts[i] != myPort)
            {
                int sockfd = 0;
                struct sockaddr_in serv_addr = {}; 
                sockfd = socket(AF_INET, SOCK_STREAM, 0);
                assert(sockfd >= 0);
                serv_addr.sin_family = AF_INET;
                serv_addr.sin_port = htons(nodePorts[i]); 
                inet_pton(AF_INET, nodeIPs[i], &serv_addr.sin_addr);

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
                    write(sockfd, &command, sizeof(uint32));
                    write(sockfd, &numItems, sizeof(uint32));
                    for (int item = firstItem; item != lastItem; item = (item +1) % MAX_QUEUE_LENGTH)
                    {
                        write(sockfd, &WorkQueue[item], sizeof(NetworkWorkItem));
                    }
                    write(sockfd, &WorkQueue[lastItem], sizeof(NetworkWorkItem));       
                    close(sockfd);
                }
            }
        }

        // TODO: check if this is robust enough!
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
        }

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

bool sendingCompleteTT = false;

void completeTTServer(int connfd)
{
    // send ENTIRE complete TT to client

    FILE *fplog = fopen(myUID, "ab+");
    fprintf(fplog, "Server thread got command to send complete TT\n");
    fclose(fplog);
    

    // 1. take a lock on completeTT to make sure nobody updates it
    sendingCompleteTT = true;
    lockCompleteTT();

    // 2. write first the hash table part of the complete TT and then the chain part
    auto t_start = std::chrono::high_resolution_clock::now();
    writeDataNetwork(connfd, completeTT, completeTTSize);
    writeDataNetwork(connfd, chainMemory, chainMemorySize);
    auto t_end = std::chrono::high_resolution_clock::now();
    double transferTime = std::chrono::duration<double>(t_end-t_start).count();
    unlockCompleteTT();
    sendingCompleteTT = false;

    fplog = fopen(myUID, "ab+");
    fprintf(fplog, "Complete TT send complete, time taken: %g seconds observed network bandwidth: %g MBps\n", 
                    transferTime, (completeTTSize+chainMemorySize)/(1024*1024*transferTime));
    fclose(fplog);
    close(connfd);
}

// main network thread responsible for sharing work items across multiple nodes on network
// 1. adds own ip address to the file containing list of nodes
// 2. wait for clients to connect
//     - once connected, get new work items from other nodes and update in complete TT tables
void network_thread_body()
{
    // 1. Add own ip address to list of nodes
    printIpAddress(myAddress);

    // hack! get ip address from a file where it's manually entered!
    {
        FILE *fp;
        while(!(fp = fopen("ip.txt", "rb+"))) 
            usleep(100000); // 100ms

        char line[1024];
        fgets(line, sizeof(line), fp);
        //sscanf(myAddress, "%s", line);
        fclose(fp);
        remove("ip.txt");
        strncpy(myAddress, line, strlen(line)-1);
    }

    int lfd;
    while((lfd = open("net.lock", O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) ;
    close(lfd);
    
    FILE *fp = fopen("nodes.txt", "rb+");
    if (fp)
    {
        numNodes = readNodeFile(fp);
        fclose(fp);
    }

    uint32 existingPort = 0x1dab;
    for (int i=0;i<numNodes;i++)
    {
        reachable[i] = true;
        if (nodePorts[i] > existingPort)
            existingPort = nodePorts[i];
    }


    // add current node to the list
    reachable[numNodes] = true;
    strcpy(nodeIPs[numNodes], myAddress);
    myPort = 0x4dab ; //existingPort+1;
    nodePorts[numNodes] = myPort;
    numNodes++;

    sprintf(myUID, "%s_%u", myAddress, myPort);


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

    // write updated list to file
    fp = fopen("nodes.txt", "wb+");
    writeNodeFile(fp);
    fclose(fp);            
    remove("net.lock");

    fprintf(fplog, "num of nodes found: %d\n", numNodes);    
    fclose(fplog);

    // get entire complete TT from the first node
#if 1    
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
        serv_addr.sin_port = htons(nodePorts[0]); 
        inet_pton(AF_INET, nodeIPs[0], &serv_addr.sin_addr);

        int result = connect(sockfd, (struct sockaddr *)&serv_addr, sizeof(serv_addr));
        if (result < 0)
        {
            FILE *fplog = fopen(myUID, "ab+");
            fprintf(fplog, "newly started node had issues getting complete TT from %s:%u, connect returned: %s\n", nodeIPs[0], nodePorts[0], strerror(errno));        
            fclose(fplog);
        }
        else
        {
            uint32 command = 2;
            write(sockfd, &command, sizeof(uint32));

            // 1. take a lock on completeTT to make sure nobody updates it
            lockCompleteTT();

            // 2. read first the hash table part of the complete TT and then the chain part
            auto t_start = std::chrono::high_resolution_clock::now();
            readDataNetwork(sockfd, completeTT, completeTTSize);
            readDataNetwork(sockfd, chainMemory, chainMemorySize);
            auto t_end = std::chrono::high_resolution_clock::now();
            double transferTime = std::chrono::duration<double>(t_end-t_start).count();
            unlockCompleteTT();

            FILE *fplog = fopen(myUID, "ab+");
            fprintf(fplog, "Complete TT recieve complete, time taken: %g seconds observed network bandwidth: %g MBps\n", 
                            transferTime, (completeTTSize+chainMemorySize)/(1024*1024*transferTime));
            fclose(fplog);

            close(sockfd);
        }
    }
#endif

    // start broadcster thread
    broadcasterThread = std::thread(broadcaster_thread_body);

    //printf("server thread starts\n");

    while(!networkThreadKillRequest)
    {
        connfd = accept(listenfd, (struct sockaddr *) NULL, NULL); 
        uint32 command = 0;
        read(connfd, &command, sizeof(uint32));

        //FILE *fplog = fopen(myUID, "ab+");
        //fprintf(fplog, "server thread got command: %d\n", command);        
        //fclose(fplog);

        if (command == 1)
        {
            // update work items in local TT from client node
            uint32 items = 0;
            read(connfd, &items, sizeof(uint32));
            for (int i=0;i<items;i++)
            {
                NetworkWorkItem workItem = {};
                read(connfd, &workItem, sizeof(NetworkWorkItem));
                if (!sendingCompleteTT) // avoid locking down the network
                    completeTTUpdateFromNetwork(workItem.hash, workItem.perft);
            }
            close(connfd);            
        }
        else if (command == 2)
        {
            // send complete TT on a seperate thread to make sure other requests are not blocked
            std::thread(completeTTServer, connfd);
        }
        else if (command == 3)
        {
            // exit request
            close(connfd);            
        }
    }
    close (listenfd);

    //printf("waiting for broadcaster thread to die\n");
    while(broadcasterThreadKillRequest) ;    
    broadcasterThread.join();

    networkThreadKillRequest = false;
}




void createNetworkThread()
{
    networkThread = std::thread(network_thread_body);
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
