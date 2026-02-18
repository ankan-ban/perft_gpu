// launcher utils


// read set of positions and occurence counts from a file, compute perfts and sum them up
#define PERFT_RECORDS_MODE 0

// the set of positions are in a text file (with FEN and occurence counts)
// default is binary file
#define PERFT_RECORD_TEXT_MODE 0

// can't make this bigger than 6/7, as the _simple kernel (breadth first search) gets called directly
// breadth first search uses lot of memory and can can't hold bigger tree 
#define GPU_LAUNCH_DEPTH 6

// launch all boards at last level in a single kernel launch 
// enable this with GPU_LAUNCH_DEPTH 6
//  - allows better use of CPU side hash, but maybe slightly less GPU utilization
#define SINGLE_LAUNCH_FOR_LAST_LEVEL 1

// print divided perft values (subtotals) after reaching this depth
#define DIVIDED_PERFT_DEPTH 10

// use multiple CPU threads to split the tree at greater depth 
#define PARALLEL_THREAD_GPU_SPLIT 1
// depth at which work is split among multiple GPUs
#define MIN_SPLIT_DEPTH 9

// launch one level of work serially on GPU
#define ENABLE_GPU_SERIAL_LEVEL 0

// use a hash table to store *all* positions at depth 7/8, etc
#define USE_COMPLETE_TT_AT_LAST_CPU_LEVEL 1

// store a level (depth-4) of positions in a persistent disk hash
// useful for save/restore of work AND also 
// for parallel perft over network with multiple nodes
#define ENABLE_DISK_HASH 1
// min problem size to make use of DISK_HASH
#define DISK_HASH_MIN_DEPTH 11
#define DISK_HASH_LEVEL 2
#if ENABLE_DISK_HASH == 1
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>
// ~4096 entries for the main hash table part
#define DISK_TT_BITS 12
#endif

// use complete hash for all levels
#define USE_COMPLETE_HASH_ALL_LEVELS 1

// the perf calculation is running over a network with multiple systems (nodes)
// each node broadcasts work computed by itself to all other nodes to avoid duplication
// the work items recieved from other nodes are added to current node's complete TT
// a new node can also request the *entire* completeTT to be sent from one of the existing nodes
#define MULTI_NODE_NETWORK_MODE 1
uint64 numItemsFromPeers = 0;


#define MEASURE_GPU_ACTIVE_TIME 1

// size of transposition tables for each depth
// depth1 transposition table has special purpose -> to find duplicates for 'deep' levels during BFS
// 20 bits ->  1 million entries  (16 MB)
// 25 bits -> 32  million entries (512 MB)
// 26 bits -> 64  million ...     (1 GB)
// 27 bits -> 128 million ...     (2 GB)
// 28 bits -> 256 million ...     (4 GB)
//           depth->        0      1      2      3       4       5       6       7       8       9      10      11      12      13      14      15         

const bool  shallow[] = {true,  true,  true,   true,   true,  false,  false,  false,  false,  false,  false,  false,  false,  false,  false,  false};

#if 1
// settings for P100 (16 GB card) + 32 GB+ sysmem
const uint32 ttBits[] = {0,       24,    25,     27,     28,     28,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0};
const bool   sysmem[] = {true, false, false,  false,   false,  true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
const int  sharedHashBits = 25;

// 1 billion entries for the main hash table part
#define COMPLETE_TT_BITS 30
//#define COMPLETE_TT_BITS 27

// 128 million entries (for each chunk of chained part)
#define COMPLETE_HASH_CHAIN_ALLOC_SIZE 128*1024*1024

#elif 0
// settings for 8 GB card (GTX 1080) + just 4 GB sysmem
const uint32 ttBits[] = { 0,       23,    25,     26,     26,     26,     26,     27,     26,      0,      0,      0,      0,      0,      0,      0 };
const bool   sysmem[] = { true, false, false,  false,  false,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true };
const int  sharedHashBits = 26;

// 16 million entries for the main hash table part
#define COMPLETE_TT_BITS 24

// 128 million entries (for chained part)
#define COMPLETE_HASH_CHAIN_ALLOC_SIZE 128*1024*1024

#elif 0
// settings for home PC (GTX 970: 4 GB card + 8 GB sysmem)
const uint32 ttBits[] = {0,       20,    25,     26,     26,     25,     26,     25,     25,      0,      0,      0,      0,      0,      0,      0};
const bool   sysmem[] = {true, false, false,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
const int  sharedHashBits = 25;

// 16 million entries for the main hash table part
#define COMPLETE_TT_BITS 24

// 128 million entries (for chained part)
#define COMPLETE_HASH_CHAIN_ALLOC_SIZE 128*1024*1024

#else
// settings for laptop (2 GB card + 16 GB sysmem)
const uint32 ttBits[] = {0,       22,     25,     27,     27,     26,     25,      0,      0,      0,      0,      0,      0,      0,      0,      0};
const bool   sysmem[] = {true, false,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
const int  sharedHashBits = 20;

// 16 million entries for the main hash table part
#define COMPLETE_TT_BITS 24

// 128 million entries (for chained part)
#define COMPLETE_HASH_CHAIN_ALLOC_SIZE 128*1024*1024

#endif
const bool sharedsysmem = true;


#define COMPLETE_TT_INDEX_BITS GET_TT_INDEX_BITS(COMPLETE_TT_BITS)

// launcher routines


double gpuTime = 0.0;
std::mutex timerCS;


CompleteHashEntry *completeTT = NULL;
CompleteHashEntry *chainMemory = NULL;
volatile uint64 chainIndex = 0;
volatile uint64 completeTTSize = 0;
volatile uint64 chainMemorySize = 0;

int numGPUs = 0;

#if USE_TRANSPOSITION_TABLE == 1

// TODO: avoid these global vars?
TTInfo128b TransTables128b[MAX_GPUs];
HexaBitBoardPosition *gpuBoard[MAX_GPUs];
uint64 *gpu_perft[MAX_GPUs];
HashKey128b *gpuHashes[MAX_GPUs];

// to avoid allocating sysmem tables multiple times!
// HACKY! - TODO: get rid of this and have the function allocate memory for all GPUs itself
bool sysmemTablesAllocated = false;

void allocAndClearMem(void **devPointer, void **hostPointer, size_t size, bool sysmem, int depth)
{
#if USE_COMPLETE_TT_AT_LAST_CPU_LEVEL == 1
    if (depth == GPU_LAUNCH_DEPTH)
    {
        // no need of this level's hash table
        *devPointer = NULL;
        *hostPointer = NULL;
        return;
    }
#endif
// still need these hash tables for smaller perfts that go to GPU directly!
#if 0 //USE_COMPLETE_HASH_ALL_LEVELS == 1
    // no need of any regular hash tables on CPU side
    if (depth < GPU_LAUNCH_DEPTH)
    {
        // no need of this level's hash table
        *devPointer = NULL;
        *hostPointer = NULL;
        return;
    }
#endif

    cudaError_t res;
    void *temp = NULL;
    *devPointer = NULL;

    if (sysmem)
    {
        if (depth >= GPU_LAUNCH_DEPTH)
        {
            if (sysmemTablesAllocated)
            {
                temp = TransTables128b[0].cpuTable[depth];
            }
            else
            {
                // plain system memory
                temp = malloc(size);
                if (!temp)
                {
                    printf("\nFailed allocating pure sysmem for transposition table!\n");
                    exit(0);
                }
            }
        }
        else
        {
            // try allocating in system memory
            if (sysmemTablesAllocated)
            {
                temp = TransTables128b[0].cpuTable[depth];
                *devPointer = TransTables128b[0].hashTable[depth];
            }
            else
            {
                res = cudaHostAlloc(&temp, size, cudaHostAllocMapped | /*cudaHostAllocWriteCombined |*/ cudaHostAllocPortable);
                if (res != cudaSuccess)
                {
                    printf("\nFailed to allocate sysmem transposition table for depth %d of %llu bytes, with error: %s\n", depth, size, cudaGetErrorString(res));
                    exit(0);
                }
                
                res = cudaHostGetDevicePointer(devPointer, temp, 0);
                if (res != S_OK)
                {
                    printf("\nFailed to get GPU mapping for sysmem hash table, with error: %s\n", cudaGetErrorString(res));
                    exit(0);
                }
            }
        }
    }
    else
    {
        res = cudaMalloc(devPointer, size);
        if (res != cudaSuccess)
        {
            printf("\nFailed to allocate GPU transposition table of %llu bytes, with error: %s\n", size, cudaGetErrorString(res));
            exit(0);
        }
    }
    *hostPointer = temp;
    if (devPointer)
    {
        hugeMemset(devPointer, size);
    }
    else
    {
        assert(*hostPointer);
        memset(*hostPointer, 0, size);
    }
}

CompleteHashEntry *chainMemoryChunks[1024];
volatile int nChunks = 0;
void allocChainMemoryChunk()
{
    chainMemorySize = COMPLETE_HASH_CHAIN_ALLOC_SIZE * sizeof(CompleteHashEntry);
    chainMemory = (CompleteHashEntry *)malloc(chainMemorySize);
    if (!chainMemory)
    {
            printf("\nFailed allocating chainMemory chunk %d!\n", nChunks);
            exit(0);
    }
    memset(chainMemory, 0, chainMemorySize);
    chainMemoryChunks[nChunks++] = chainMemory;
    chainIndex = 0;    
}

void allocCompleteTT()
{
#if USE_COMPLETE_TT_AT_LAST_CPU_LEVEL == 1
    if (completeTT == NULL)
    {
        completeTTSize = GET_TT_SIZE_FROM_BITS(COMPLETE_TT_BITS) * sizeof(CompleteHashEntry);
        completeTT = (CompleteHashEntry *)malloc(completeTTSize);
        if (!completeTT)
        {
                printf("\nFailed allocating completeTT!\n");
                exit(0);
        }
        memset(completeTT, 0, completeTTSize);

        memset(chainMemoryChunks, 0, sizeof(chainMemoryChunks));
        allocChainMemoryChunk();
    }
#endif
}

void freeCompleteTT()
{
    if (completeTT)
        free(completeTT);

    for (int i=0;i<nChunks;i++)
        free(chainMemoryChunks[i]);
}

void setupHashTables128b(TTInfo128b &tt)
{
    // allocate the shared hash table
    void *sharedTable, *sharedTableCPU;
    allocAndClearMem(&sharedTable, &sharedTableCPU, GET_TT_SIZE_FROM_BITS(sharedHashBits) * sizeof(HashEntryPerft128b), sharedsysmem, 9);

    memset(&tt, 0, sizeof(tt));
    for (int i = 1; i < MAX_PERFT_DEPTH; i++)
    {
        tt.shallowHash[i] = shallow[i];
        uint32 bits = ttBits[i];
        if (bits)
        {
            allocAndClearMem(&tt.hashTable[i], &tt.cpuTable[i],
                GET_TT_SIZE_FROM_BITS(bits) * (shallow[i] ? sizeof(HashKey128b) : sizeof(HashEntryPerft128b)), sysmem[i], i);
        }
        else
        {
            tt.hashTable[i] = sharedTable;
            tt.cpuTable[i] = sharedTableCPU;
            bits  = sharedHashBits;
        }
        tt.indexBits[i] = GET_TT_INDEX_BITS(bits);
        tt.hashBits[i] = GET_TT_HASH_BITS(bits);
    }

    sysmemTablesAllocated = true;
}

void freeHashTables()
{
    bool sharedDeleted = false;
    for (int g = 0; g < numGPUs; g++)
    {
        cudaSetDevice(g);
        for (int i = 1; i < MAX_PERFT_DEPTH; i++)
        {
            uint32 bits = ttBits[i];
            if (bits == 0)
            {
                if (!sharedDeleted)
                {
                    // delete the shared sysmem hash table
                    free(TransTables128b[g].cpuTable[i]);
                    sharedDeleted = true;
                }
            }
            else if(sysmem[i])
            {
                if (g == 0)
                {
                    if (i >= GPU_LAUNCH_DEPTH)
                    {
                        if (TransTables128b[g].cpuTable[i])
                            free(TransTables128b[g].cpuTable[i]);
                    }
                    else
                    {
                        cudaFree(TransTables128b[g].hashTable[i]);
                    }
                }
            }
            else
            {
                cudaFree(TransTables128b[g].hashTable[i]);
            }
        }
    }
    cudaSetDevice(0);
    memset(TransTables128b, 0, sizeof(TransTables128b));
}

// quick and dirty move-list sorting routine
// only purpose is to get all quiet moves at the start and hope for better hash table usage
void sortMoves(CMove *moves, int nMoves)
{
    int nq = 0;
    CMove sortedMoves[MAX_MOVES];

    for (int i = 0; i < nMoves; i++)
    {
        if (moves[i].getFlags() == CM_FLAG_QUIET_MOVE)
        {
            sortedMoves[nq++] = moves[i];
        }
    }

    int s = 0, u = 0;
    for (int i = 0; i < nMoves; i++)
    {
        if (moves[i].getFlags() == CM_FLAG_QUIET_MOVE)
        {
            sortedMoves[s++] = moves[i];
        }
        else
        {
            sortedMoves[nq + (u++)] = moves[i];
        }
    }

    memcpy(moves, sortedMoves, sizeof(CMove)* nMoves);
}

void randomizeMoves(CMove *moves, int nMoves)
{
    for (int i=0;i<nMoves;i++)
    {
        int j = std::rand() % nMoves;
        CMove otherMove = moves[j];
        moves[j] = moves[i];
        moves[i] = otherMove;
    }
}

InfInt perft_bb_cpu_launcher(HexaBitBoardPosition *pos, uint32 depth, char *dispPrefix);

thread_local int activeGpu = 0;

enum eThreadStatus
{
    THREAD_IDLE = 0,
    WORK_SUBMITTED = 1,
    THREAD_TERMINATE_REQUEST = 2,
    THREAD_TERMINATED = 3,
    THREAD_CREATED = 4
};

// 2 way communication between main thread and worker threads
volatile eThreadStatus threadStatus[MAX_GPUs];

// main thread -> worker threads
volatile HexaBitBoardPosition *posForThread[MAX_GPUs];
volatile char *dispStringForThread[MAX_GPUs];


// worker threads -> main thread
volatile InfInt *perftForThread[MAX_GPUs];

std::mutex criticalSection;
std::mutex diskCS;

void worker_thread_start(uint32 depth, uint32 gpuId)
{
    cudaSetDevice(gpuId);
    activeGpu = gpuId;

    threadStatus[gpuId] = THREAD_IDLE;

    // wait for work
    while (1)
    {
        if (threadStatus[gpuId] == THREAD_TERMINATE_REQUEST)
        {
            break;
        }
        else if (threadStatus[gpuId] == THREAD_IDLE)
        {
            continue;
        }
        else if (threadStatus[gpuId] == WORK_SUBMITTED)
        {
            InfInt perftVal = perft_bb_cpu_launcher((HexaBitBoardPosition *)posForThread[gpuId], depth, (char*)dispStringForThread[gpuId]);

            if (depth >= DIVIDED_PERFT_DEPTH)
            {
                criticalSection.lock();
                //printf("%s   %20llu\n", dispStringForThread[gpuId], perftVal);
                printf("%s   %20s\n", dispStringForThread[gpuId], perftVal.toString().c_str());

                fflush(stdout);
                criticalSection.unlock();
            }

            *((InfInt *)perftForThread[gpuId]) = perftVal;
            threadStatus[gpuId] = THREAD_IDLE;
        }
    }

    threadStatus[gpuId] = THREAD_TERMINATED;
}

// launch work on multiple threads (each associated with a single GPU), 
// wait for enough parallel work is done, and only then wait for the threads to finish
InfInt perft_multi_threaded_gpu_launcher(HexaBitBoardPosition *pos, uint32 depth, char *dispPrefix)
{
    CMove genMoves[MAX_MOVES];
    HexaBitBoardPosition childPos;
    HexaBitBoardPosition childBoards[MAX_MOVES];
    char childStrings[MAX_MOVES][128];
    InfInt perftResults[MAX_MOVES];

    int nMoves = generateMoves(pos, pos->chance, genMoves);

// doesn't help :-/
//#if ENABLE_DISK_HASH == 1
//    randomizeMoves(genMoves, nMoves);
//#else    
    sortMoves(genMoves, nMoves);
//#endif    

    std::thread threads[MAX_GPUs];
    // Launch a thread for each GPU
    for (int i = 0; i < numGPUs; ++i)
    {
        // create the thread
        threadStatus[i] = THREAD_CREATED;
        threads[i] = std::thread(worker_thread_start, depth - 1, i);

        // wait for the thread to get initialized
        while (threadStatus[i] != THREAD_IDLE);
    }

    for (int i = 0; i < nMoves; i++)
    {

        char moveString[10];
        Utils::getCompactMoveString(genMoves[i], moveString);
        strcpy(childStrings[i], dispPrefix);
        strcat(childStrings[i], moveString);

        childPos = *pos;
        uint64 fakeHash = 0;

        if (pos->chance == WHITE)
            MoveGeneratorBitboard::makeMove<WHITE, false>(&childPos, fakeHash, genMoves[i]);
        else
            MoveGeneratorBitboard::makeMove<BLACK, false>(&childPos, fakeHash, genMoves[i]);

        childBoards[i] = childPos;

        // find an idle worker thread to submit work
        int chosenThread = -1;
        while (chosenThread == -1)
        {
            for (int t = 0; t < numGPUs; t++)
                if (threadStatus[t] == THREAD_IDLE)
                {
                    chosenThread = t;
                    break;
                }
        }

        // submit work on the worker thread
        posForThread[chosenThread] = &childBoards[i];
        dispStringForThread[chosenThread] = childStrings[i];
        perftForThread[chosenThread] = &perftResults[i];
        threadStatus[chosenThread] = WORK_SUBMITTED;
    }

    // Ankan - TODO: if there is any thread idle, give it same position as some other active thread
    //        it will at least do something useful.
    while(1)
    {
        bool allIdle = true;
        for (int t = 0; t < numGPUs; t++)
        {
            if (threadStatus[t] == THREAD_IDLE)
            {
                for (int t2 = 0; t2 < numGPUs; t2++)
                if (threadStatus[t2] != THREAD_IDLE)
                {
                    posForThread[t] = posForThread[t2];
                    dispStringForThread[t] = dispStringForThread[t2];
                    InfInt temp;
                    perftForThread[t] = &temp;
                    threadStatus[t] = WORK_SUBMITTED;
                    allIdle = false;
                }
            }
        }
        if(allIdle)
        break;
    }

    // wait for all threads to terminate
    for (int t = 0; t < numGPUs; t++)
    {
        while (threadStatus[t] != THREAD_IDLE);
        threadStatus[t] = THREAD_TERMINATE_REQUEST;
        while (threadStatus[t] != THREAD_TERMINATED);
        threads[t].join();
    }
    

    InfInt count = 0;
    for (int i = 0; i < nMoves; i++)
    {
        count += perftResults[i];
    }
    return count;
}

// TODO: what happens to pending writes? No CS protection there!
//   - lockless XOR trick should give (some?) protection
void lockCompleteTT()
{
    criticalSection.lock();
}

void unlockCompleteTT()
{
    criticalSection.unlock();    
}

// returns non-null entryPtr if not found
uint64 completeTTProbe(HashKey128b hash, int depth, CompleteHashEntry **pEntryPtr, bool finalHash = false)
{
    if (!finalHash)
    {
        hash ^= (ZOB_KEY_128(depth) * depth);
    }

    uint64 ttVal = 0;   // value from transposition table in case of hash hit
    *pEntryPtr = NULL;    // new entry to update in case of hash miss

    CompleteHashEntry *entry;

    criticalSection.lock();
    entry = &completeTT[hash.lowPart & COMPLETE_TT_INDEX_BITS];
    while (1)
    {
        if (entry->hash == HashKey128b(0,0))
        {
            // blank record

            // mark in-use (so that other parallel thread doesn't overwrite it)
            // with this approach there is a (very!) small chance that duplicate entries might get added for the same position.
            //  ... but it should always be correct.
            entry->hash = HashKey128b(ALLSET,ALLSET);
            entry->nextIndex = ~0;
            entry->nextTT = ~0;

            *pEntryPtr = entry;

            //printf("g  %d: %llu ", depth, hash.lowPart);  // Ankan - for testing   

            break;
        }
        
        // XOR with perft value to extract hash part
        HashKey128b entryHash = entry->hash;
        entryHash.highPart ^= entry->perft;
        entryHash.lowPart  ^= entry->perft;
        
        if (entryHash == hash)
        {
            // hash hit
            ttVal = entry->perft;
            break;
        }

        if (entry->nextIndex == ~0)
        {
            chainIndex++;
            if (chainIndex > COMPLETE_HASH_CHAIN_ALLOC_SIZE)
            {
                allocChainMemoryChunk();
            }
            entry->nextIndex = chainIndex;
            entry->nextTT = (nChunks - 1);
        }
        entry = &(chainMemoryChunks[entry->nextTT][entry->nextIndex]);
    }
    criticalSection.unlock();

    return ttVal;
}


void enqueueWorkItem(CompleteHashEntry *item);

void completeTTStore(CompleteHashEntry *entryPtr, HashKey128b hash, int depth, uint64 perft)
{
    hash ^= (ZOB_KEY_128(depth) * depth);

    // XOR trick to prevent random bit flip errors (and also half read network entries)
    hash.highPart ^= perft;
    hash.lowPart  ^= perft;    

    criticalSection.lock(); // Ankan - remove this likely not needed (very low probablity)
    entryPtr->hash = hash;
    entryPtr->perft = perft;
    criticalSection.unlock();

#if MULTI_NODE_NETWORK_MODE == 1
    enqueueWorkItem(entryPtr);
#endif
}

void completeTTUpdateFromNetwork(HashKey128b hash, uint64 perft)
{
    CompleteHashEntry *pEntry;

    HashKey128b actualHash = hash;
    actualHash.highPart ^= perft;
    actualHash.lowPart  ^= perft;

    completeTTProbe(actualHash, 0, &pEntry, true);
    if (pEntry)
    {
        pEntry->hash = hash;
        pEntry->perft = perft;
        numItemsFromPeers++;
    }
}

#if ENABLE_DISK_HASH == 1
uint64 diskTTProbe(HashKey128b posHash128b, int depth, DiskHashEntry* pDiskEntry, uint64 *pDiskHashIndex)
{
    DiskHashEntry diskEntry = {};
    uint64 diskEntryIndex = 0;

    // check disk hash
    uint64 ttVal = ALLSET;
    int fd = open("perfthash.dat", O_RDWR);

    uint64 index = posHash128b.lowPart & GET_TT_INDEX_BITS(DISK_TT_BITS);


    diskCS.lock(); // protect against parallel access in same process!

    // protect against parallel access across nodes of cluster
    // this doesn't seem to work (causes cluster to hang!)
    // flock(fd, LOCK_EX);
    // use a lockfile instead
    int lfd;
    while((lfd = open(".lock", O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH)) == -1) ;
    close(lfd);

    lseek(fd, index*sizeof(DiskHashEntry), SEEK_SET);
    read(fd, &diskEntry, sizeof(DiskHashEntry));
    while(1)
    {
        if(diskEntry.hash == HashKey128b(0,0))
        {
            // blank record

            // mark in-use (so that other parallel thread doesn't overwrite it)
            diskEntry.hash = HashKey128b(ALLSET,ALLSET);
            diskEntry.next = ~0;
            diskEntry.depth = depth;

            diskEntryIndex = index;
            lseek(fd, index*sizeof(DiskHashEntry), SEEK_SET);
            write(fd, &diskEntry, sizeof(DiskHashEntry));                
            break;                
        }
        if (diskEntry.hash == posHash128b && diskEntry.depth == depth)
        {
            // hash hit
            ttVal = diskEntry.perft;
            break;
        }
        if (diskEntry.next == ~0)
        {
            // add new entry to chain
            uint64 nextIndex = lseek(fd, 0, SEEK_END) / sizeof(DiskHashEntry);
            diskEntry.next = nextIndex;
            lseek(fd, index*sizeof(DiskHashEntry), SEEK_SET);
            write(fd, &diskEntry, sizeof(DiskHashEntry));

            DiskHashEntry blankRecord = {};
            lseek(fd, nextIndex * sizeof(DiskHashEntry), SEEK_SET);
            write(fd, &blankRecord, sizeof(DiskHashEntry));
        }

        // go to the next entry in chain
        index = diskEntry.next;
        lseek(fd, index*sizeof(DiskHashEntry), SEEK_SET);
        read(fd, &diskEntry, sizeof(DiskHashEntry));
    }
    
    //flock(fd, LOCK_UN);
    close(fd);
    remove(".lock");
    diskCS.unlock();

    *pDiskEntry = diskEntry;
    *pDiskHashIndex = diskEntryIndex;
    return ttVal;
}


void diskTTStore(DiskHashEntry *diskEntry, uint64 diskEntryIndex, HashKey128b posHash128b, int depth, uint64 perft)
{
    diskEntry->hash = posHash128b;
    diskEntry->perft = perft;
    diskEntry->depth = depth;
    diskEntry->next = ~0;    // possible bug here! should read from the disk and check

    diskCS.lock();        // avoid parallel file access from same process
    int fd = open("perfthash.dat", O_RDWR);
    lseek(fd, diskEntryIndex * sizeof(DiskHashEntry), SEEK_SET);
    write(fd, diskEntry, sizeof(DiskHashEntry));
    close(fd);
    diskCS.unlock();        
}
#endif

int diskHashDepth = 0;  // set to search depth - 4

int splitDepth = MIN_SPLIT_DEPTH;
uint32 maxMemoryUsage = 0;

int numRegularLaunches = 0;
int numRetryLaunches = 0;

// launch last two levels
// attemps to overlap CPU and GPU time
// - cpu time to check hash table
// - gpu time for perft calculation
#if 0
uint64 perft_bb_second_last_level_launcher(HexaBitBoardPosition *pos, uint32 depth)
{
    HashKey128b hash = MoveGeneratorBitboard::computeZobristKey128b(pos);

    // first level children
    CMove firstLevelMoves[MAX_MOVES];
    uint64 firstLevelPerfts[MAX_MOVES];
    int firstLevelNewBoards = 0;
    CompleteHashEntry *firstLevelHashEntries[MAX_MOVES];

    uint8 color = pos->chance;
    int nMoves = generateMoves(pos, color, firstLevelMoves);
    sortMoves(moves, nMoves);

    uint64 perft = 0;

    for(j=0;j<nMoves;j++)
    {
        HexaBitBoardPosition curPos = *pos;
        HashKey128b curHash = makeMoveAndUpdateHash(&curPos, hash, firstLevelMoves[j], color);

        // check in hash table
        CompleteHashEntry *entryPtr;

        uint64 ttVal = completeTTProbe(newHash, depth+1, &entryPtr);
        if (entryPtr == NULL)
        {
            perft += ttVal;
            continue;
        }
        firstLevelHashEntries[firstLevelNewBoards++] = entryPtr;
        
        uint64 count = 0;

        // second level moves and boards
        HexaBitBoardPosition childBoards[MAX_MOVES];
        CMove                moves[MAX_MOVES];
        HashKey128b          hashes[MAX_MOVES];
        uint64               perfts[MAX_MOVES];

        uint8 color = pos->chance;
        int nChildMoves = generateMoves(curPos, color, moves);
        sortMoves(moves, nChildMoves);

        int nNewBoards = 0;
        CompleteHashEntry *newEntryPointer[MAX_MOVES];        

        for (int i = 0; i < nMoves; i++)
        {
            childBoards[nNewBoards] = *curPos;
            HashKey128b newHash = makeMoveAndUpdateHash(&childBoards[nNewBoards], curHash, moves[i], color);

            CompleteHashEntry *entryPtr;
            uint64 ttVal = completeTTProbe(newHash, depth + 2, &entryPtr);
            if (entryPtr == NULL)
            {
                count += ttVal;
                continue;
            }
            newEntryPointer[nNewBoards] = entryPtr;
            hashes[nNewBoards] = newHash;
            nNewBoards++;
        }
           
        numRegularLaunches += nNewBoards;


#if MEASURE_GPU_ACTIVE_TIME == 1
        START_TIMER
#endif

        // copy host->device in one go
        cudaMemcpy(gpuBoard[activeGpu], childBoards, sizeof(HexaBitBoardPosition)*nNewBoards, cudaMemcpyHostToDevice);
        cudaMemset(gpu_perft[activeGpu], 0, sizeof(uint64)*nNewBoards);
        cudaMemcpy(gpuHashes[activeGpu], hashes, sizeof(HashKey128b)*nNewBoards, cudaMemcpyHostToDevice);

        int batchSize = nNewBoards;
        memset(perfts, 0xFF, sizeof(uint64)*nNewBoards);
        while(1)
        {
            bool done = true;

            for (int i = 0; i < nNewBoards;)
            {
                if (perfts[i] == ALLSET)
                {
                    int count = ((nNewBoards - i) > batchSize) ? batchSize : nNewBoards - i;

                    // skip the ones already computed
                    while (perfts[i + count - 1] != ALLSET) count--;

                    if (batchSize != nNewBoards)
                        numRetryLaunches++;

                    perft_bb_gpu_simple_hash << <1, 1 >> > (count, &gpuBoard[activeGpu][i], &gpuHashes[activeGpu][i], &gpu_perft[activeGpu][i], depth - 1, preAllocatedBufferHost[activeGpu],
                                                            TransTables128b[activeGpu], true);

                    cudaError_t err = cudaMemcpy(&perfts[i], &gpu_perft[activeGpu][i], sizeof(uint64) * count, cudaMemcpyDeviceToHost);

                    if (perfts[i] == ALLSET)
                    {
                        done = false;
                        for (int j = 0; j < count; j++)
                            perfts[i + j] = ALLSET;

                        cudaMemset(&gpu_perft[activeGpu][i], 0, sizeof(uint64) * count);
                    }
                    i += count;
                }
                else
                {
                    i++;
                }
            }
            if (done)
                break;

            // some bug here??!!!
            if (batchSize == 1)
            {
                printf("\nCan't even fit a single launch! Exiting\n");
                exit(0);
            }
            batchSize = (batchSize + 3) / 4;
        }

        // update memory usage estimation
        uint32 currentMemUsage = 0;
        cudaError_t s = cudaMemcpyFromSymbol(&currentMemUsage, maxMemoryUsed, sizeof(int), 0, cudaMemcpyDeviceToHost);
        if (currentMemUsage > maxMemoryUsage)
        {
            maxMemoryUsage = currentMemUsage;
        }


        // collect perft results and update hash table
        for (int i = 0; i < nNewBoards; i++)
        {
            if (perfts[i] == ALLSET)
            {
                printf("\nUnexpected ERROR? Exiting!!\n");
                exit(0);
            }

            count += perfts[i];

            HashKey128b posHash128b = hashes[i];

                completeTTStore(newEntryPointer[i], hashes[i], depth+2, perfts[i]);
        }

        return count;
        
                
        
    }
}
#endif

// launch all boards of the last level without waiting for previous work to finish
// tiny bit improvement in GPU utilization
uint64 perft_bb_last_level_launcher(HexaBitBoardPosition *pos, uint32 depth)
{
    HashKey128b hash = MoveGeneratorBitboard::computeZobristKey128b(pos);

    HexaBitBoardPosition childBoards[MAX_MOVES];
    CMove moves[MAX_MOVES];
    HashKey128b hashes[MAX_MOVES];
    uint64 perfts[MAX_MOVES];

    // generate moves for the current board and call the breadth first routine for all child boards
    uint8 color = pos->chance;
    int nMoves = generateMoves(pos, color, moves);
    sortMoves(moves, nMoves);
 
    HashEntryPerft128b *hashTable = (HashEntryPerft128b *)TransTables128b[0].cpuTable[depth - 1];
    uint64 indexBits = TransTables128b[0].indexBits[depth - 1];
    uint64 hashBits = TransTables128b[0].hashBits[depth - 1];
#if USE_COMPLETE_TT_AT_LAST_CPU_LEVEL == 1
    CompleteHashEntry *newEntryPointer[MAX_MOVES];
#endif

    int nNewBoards = 0;
    uint64 count = 0;
    for (int i = 0; i < nMoves; i++)
    {
        childBoards[nNewBoards] = *pos;
        HashKey128b newHash = makeMoveAndUpdateHash(&childBoards[nNewBoards], hash, moves[i], color);

        // check in hash table
#if USE_COMPLETE_TT_AT_LAST_CPU_LEVEL == 1
        if (depth == GPU_LAUNCH_DEPTH + 1)
        {
            CompleteHashEntry *entryPtr;

            uint64 ttVal = completeTTProbe(newHash, GPU_LAUNCH_DEPTH, &entryPtr);
            if (entryPtr == NULL)
            {
                count += ttVal;
                continue;
            }
            newEntryPointer[nNewBoards] = entryPtr;
        }
        else
#endif
        {
            HashEntryPerft128b entry;
            entry = hashTable[newHash.lowPart & indexBits];
            // extract data from the entry using XORs (hash part is stored XOR'ed with data for lockless hashing scheme)
            entry.hashKey.highPart ^= entry.perftVal;
            entry.hashKey.lowPart ^= entry.perftVal;

            if ((entry.hashKey.highPart == newHash.highPart) && ((entry.hashKey.lowPart & hashBits) == (newHash.lowPart & hashBits))
                && (entry.depth == (depth - 1)))
            {
                // hash hit
                count += entry.perftVal;
                continue;
            }
        }
        hashes[nNewBoards] = newHash;
        nNewBoards++;
    }

    // count no. of launches
    if (depth == GPU_LAUNCH_DEPTH + 1)
        numRegularLaunches += nNewBoards;
    else
        numRetryLaunches += nNewBoards;

    if (depth < GPU_LAUNCH_DEPTH)
    {
        printf("Can't even meet depth - 1 ??\n");
    }

#if MEASURE_GPU_ACTIVE_TIME == 1
    START_TIMER
#endif

    // copy host->device in one go
    cudaMemcpy(gpuBoard[activeGpu], childBoards, sizeof(HexaBitBoardPosition)*nNewBoards, cudaMemcpyHostToDevice);
    cudaMemset(gpu_perft[activeGpu], 0, sizeof(uint64)*nNewBoards);
    cudaMemcpy(gpuHashes[activeGpu], hashes, sizeof(HashKey128b)*nNewBoards, cudaMemcpyHostToDevice);

#if SINGLE_LAUNCH_FOR_LAST_LEVEL == 1
    int batchSize = nNewBoards;
    memset(perfts, 0xFF, sizeof(uint64)*nNewBoards);
    while(1)
    {
        bool done = true;

        for (int i = 0; i < nNewBoards;)
        {
            if (perfts[i] == ALLSET)
            {
                int count = ((nNewBoards - i) > batchSize) ? batchSize : nNewBoards - i;

                // skip the ones already computed
                while (perfts[i + count - 1] != ALLSET) count--;

                if (batchSize != nNewBoards)
                    numRetryLaunches++;

                perft_bb_gpu_simple_hash << <1, 1 >> > (count, &gpuBoard[activeGpu][i], &gpuHashes[activeGpu][i], &gpu_perft[activeGpu][i], depth - 1, preAllocatedBufferHost[activeGpu],
                                                        TransTables128b[activeGpu], true);

                cudaError_t err = cudaMemcpy(&perfts[i], &gpu_perft[activeGpu][i], sizeof(uint64) * count, cudaMemcpyDeviceToHost);

                if (perfts[i] == ALLSET)
                {
                    done = false;
                    for (int j = 0; j < count; j++)
                        perfts[i + j] = ALLSET;

                    cudaMemset(&gpu_perft[activeGpu][i], 0, sizeof(uint64) * count);
                }
                i += count;
            }
            else
            {
                i++;
            }
        }
        if (done)
            break;

        // some bug here??!!!
        if (batchSize == 1)
        {
            printf("\nCan't even fit a single launch! Exiting\n");
            exit(0);
        }
        batchSize = (batchSize + 3) / 4;
    }
#else
    // hope that these will get scheduled on GPU in tightly packed manner without much overhead
    for (int i = 0; i < nNewBoards; i++)
    {
        perft_bb_gpu_simple_hash <<<1, 1 >>> (1, &gpuBoard[activeGpu][i], &gpuHashes[activeGpu][i], &gpu_perft[activeGpu][i], depth - 1, preAllocatedBufferHost[activeGpu],
                                              TransTables128b[activeGpu], true);
    }

    // copy device-> host in one go
    cudaError_t err = cudaMemcpy(perfts, gpu_perft[activeGpu], sizeof(uint64) * nNewBoards, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("\nGot error: %s\n", cudaGetErrorString(err));
    }
#endif

#if MEASURE_GPU_ACTIVE_TIME == 1
    STOP_TIMER
    timerCS.lock();
    gpuTime += gTime;
    timerCS.unlock();
#endif

    // update memory usage estimation
    uint32 currentMemUsage = 0;
    cudaError_t s = cudaMemcpyFromSymbol(&currentMemUsage, maxMemoryUsed, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (currentMemUsage > maxMemoryUsage)
    {
        maxMemoryUsage = currentMemUsage;
    }


    // collect perft results and update hash table
    for (int i = 0; i < nNewBoards; i++)
    {
        if (perfts[i] == ALLSET)
        {
#if SINGLE_LAUNCH_FOR_LAST_LEVEL == 1
            printf("\nUnexpected ERROR? Exiting!!\n");
            exit(0);
#endif
            // OOM error!
            // try with lower depth
            perfts[i] = perft_bb_last_level_launcher(&childBoards[i], depth - 1);
        }

        count += perfts[i];

        HashKey128b posHash128b = hashes[i];

#if USE_COMPLETE_TT_AT_LAST_CPU_LEVEL == 1
        if (depth == GPU_LAUNCH_DEPTH + 1)
        {
            completeTTStore(newEntryPointer[i], hashes[i], GPU_LAUNCH_DEPTH, perfts[i]);
        }
        else
#endif
        {
            HashEntryPerft128b oldEntry;
            oldEntry = hashTable[posHash128b.lowPart & indexBits];

            // replace only if old entry was shallower (or of same depth)
            if (hashTable && oldEntry.depth <= (depth - 1))
            {
                HashEntryPerft128b newEntry;
                newEntry.perftVal = perfts[i];
                newEntry.hashKey.highPart = posHash128b.highPart;
                newEntry.hashKey.lowPart = (posHash128b.lowPart & hashBits);
                newEntry.depth = (depth - 1);

                // XOR hash part with data part for lockless hashing
                newEntry.hashKey.lowPart ^= newEntry.perftVal;
                newEntry.hashKey.highPart ^= newEntry.perftVal;

                hashTable[posHash128b.lowPart & indexBits] = newEntry;
            }
        }
    }

    return count;
}

InfInt perft_bb_cpu_launcher(HexaBitBoardPosition *pos, uint32 depth, char *dispPrefix)
{
    HexaBitBoardPosition newPositions[MAX_MOVES];
    CMove genMoves[MAX_MOVES];
    char  dispString[128];

    HashKey128b posHash128b;
    posHash128b = MoveGeneratorBitboard::computeZobristKey128b(pos);

    // check hash table
#if USE_COMPLETE_HASH_ALL_LEVELS == 1
    CompleteHashEntry *completeTTEntryPtr = NULL;
    uint64 ttVal = completeTTProbe(posHash128b, depth, &completeTTEntryPtr);
    if (completeTTEntryPtr == NULL)
    {
        return ttVal;
    }
#else    
    HashEntryPerft128b *hashTable = (HashEntryPerft128b *) TransTables128b[0].cpuTable[depth];
    uint64 indexBits = TransTables128b[0].indexBits[depth];
    uint64 hashBits = TransTables128b[0].hashBits[depth];
    HashEntryPerft128b entry;

    if (hashTable)
    {
        entry = hashTable[posHash128b.lowPart & indexBits];
        // extract data from the entry using XORs (hash part is stored XOR'ed with data for lockless hashing scheme)
        entry.hashKey.highPart ^= entry.perftVal;
        entry.hashKey.lowPart ^= entry.perftVal;

        if ((entry.hashKey.highPart == posHash128b.highPart) && ((entry.hashKey.lowPart & hashBits) == (posHash128b.lowPart & hashBits))
            && (entry.depth == depth))
        {
            // hash hit
            return entry.perftVal;
        }
    }
#endif

#if ENABLE_DISK_HASH == 7

    // new entry to update in case of hash miss
    uint64 diskEntryIndex = 0; 
    DiskHashEntry diskEntry = {};

    if (depth == diskHashDepth)
    {
        uint64 ttVal = diskTTProbe(posHash128b, depth, &diskEntry, &diskEntryIndex);
        if (ttVal != ALLSET)
        {
#if USE_COMPLETE_HASH_ALL_LEVELS == 1
            completeTTStore(completeTTEntryPtr, posHash128b, depth, ttVal);
#else     
            // store in local in-memory hash table for faster access next time
            // replace only if old entry was shallower (or of same depth)
            if (hashTable && (entry.depth <= depth))
            {
                HashEntryPerft128b newEntry;
                newEntry.perftVal = ttVal;
                newEntry.hashKey.highPart = posHash128b.highPart;
                newEntry.hashKey.lowPart = (posHash128b.lowPart & hashBits);
                newEntry.depth = depth;

                // XOR hash part with data part for lockless hashing
                newEntry.hashKey.lowPart ^= newEntry.perftVal;
                newEntry.hashKey.highPart ^= newEntry.perftVal;

                hashTable[posHash128b.lowPart & indexBits] = newEntry;
            }
#endif            
            return ttVal;
        }
    }
#endif    

    uint32 nMoves = 0;
    InfInt count = 0;

    if (depth == GPU_LAUNCH_DEPTH+1)
    {
        count = perft_bb_last_level_launcher(pos, depth);
    }
    else if (depth <= GPU_LAUNCH_DEPTH)
    {
        // launch GPU perft routine
        uint64 res;
        {
            cudaMemcpy(gpuBoard[activeGpu], pos, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
            cudaMemset(gpu_perft[activeGpu], 0, sizeof(uint64));
            cudaMemcpy(gpuHashes[activeGpu], &posHash128b, sizeof(HashKey128b), cudaMemcpyHostToDevice);
            // gpu_perft is a single 64 bit integer which is updated using atomic adds by leave nodes

#if ENABLE_GPU_SERIAL_LEVEL == 1
            if (depth >= 6)
            {
                perft_bb_gpu_launcher_hash<<<1, 1 >>> (gpuBoard[activeGpu], posHash128b, gpu_perft[activeGpu], depth, preAllocatedBufferHost[activeGpu],
                                                       TransTables128b[activeGpu]);
            }
            else
#endif
            {
                perft_bb_gpu_simple_hash << <1, 1 >> > (1, gpuBoard[activeGpu], gpuHashes[activeGpu], gpu_perft[activeGpu], depth, preAllocatedBufferHost[activeGpu],
                    TransTables128b[activeGpu], true);

                // update memory usage estimation
                int currentMemUsage = 0;
                cudaError_t s = cudaMemcpyFromSymbol(&currentMemUsage, maxMemoryUsed, sizeof(int), 0, cudaMemcpyDeviceToHost);
                if (currentMemUsage > maxMemoryUsage)
                {
                    maxMemoryUsage = currentMemUsage;
                }
            }

            cudaError_t err = cudaMemcpy(&res, gpu_perft[activeGpu], sizeof(uint64), cudaMemcpyDeviceToHost);

            if (res == ALLSET)
            {
                //printf("\nOOM occured. BAD! Exiting\n");
                //exit(0);
                res = perft_bb_last_level_launcher(pos, depth);

            }
        }

        count = res;
    }
    else
#if PARALLEL_THREAD_GPU_SPLIT == 1
    if (depth == splitDepth)
    {
        count = perft_multi_threaded_gpu_launcher(pos, depth, dispPrefix);
    }
    else
#endif
    {
        nMoves = generateMoves(pos, pos->chance, genMoves);
#if ENABLE_DISK_HASH == 1
        if (depth > diskHashDepth)
        {
            randomizeMoves(genMoves, nMoves);
        }
        else
#endif   
        {     
            sortMoves(genMoves, nMoves);
        }

        for (uint32 i = 0; i < nMoves; i++)
        {
            newPositions[i] = *pos;
            uint64 fakeHash = 0;

            if (pos->chance == WHITE)
                MoveGeneratorBitboard::makeMove<WHITE, false>(&newPositions[i], fakeHash, genMoves[i]);
            else
                MoveGeneratorBitboard::makeMove<BLACK, false>(&newPositions[i], fakeHash, genMoves[i]);

            char moveString[10];
            Utils::getCompactMoveString(genMoves[i], moveString);
            strcpy(dispString, dispPrefix);
            strcat(dispString, moveString);
            InfInt childPerft = perft_bb_cpu_launcher(&newPositions[i], depth - 1, dispString);

            if (depth > DIVIDED_PERFT_DEPTH)
            {
                criticalSection.lock();
                //printf("%s   %20llu\n", dispString, childPerft);
                printf("%s   %20s\n", dispString, childPerft.toString().c_str());
                fflush(stdout);
                criticalSection.unlock();
            }
            count += childPerft;
        }

    }

    // store in hash table
#if USE_COMPLETE_HASH_ALL_LEVELS == 1
    if (count < InfInt(ALLSET))
    {
        completeTTStore(completeTTEntryPtr, posHash128b, depth, count.toUnsignedLongLong());
    }
#else        
    // replace only if old entry was shallower (or of same depth)
    if (hashTable && (entry.depth <= depth) && (count < InfInt(ALLSET)))
    {
        HashEntryPerft128b newEntry;
        newEntry.perftVal = count.toUnsignedLongLong();
        newEntry.hashKey.highPart = posHash128b.highPart;
        newEntry.hashKey.lowPart = (posHash128b.lowPart & hashBits);
        newEntry.depth = depth;

        // XOR hash part with data part for lockless hashing
        newEntry.hashKey.lowPart ^= newEntry.perftVal;
        newEntry.hashKey.highPart ^= newEntry.perftVal;

        hashTable[posHash128b.lowPart & indexBits] = newEntry;
    }
#endif
    // update disk hash table too!
#if ENABLE_DISK_HASH == 7
    if ((depth == diskHashDepth) && (count < InfInt(ALLSET)))
    {
        diskTTStore(&diskEntry, diskEntryIndex, posHash128b, depth, count.toUnsignedLongLong());
    }    
#endif

    return count;
}

// called only for bigger perfts - shows move count distribution for each move
void dividedPerft(HexaBitBoardPosition *pos, uint32 depth)
{
#if USE_COMPLETE_TT_AT_LAST_CPU_LEVEL == 1
    // (need to clear this as it would otherwise contain perfts of other depths)
    // no need to clear this if we include depth when computing position hashes
    //memset(completeTT, 0, GET_TT_SIZE_FROM_BITS(COMPLETE_TT_BITS) * sizeof(CompleteHashEntry));

    // no need to clear this if we don't reuse old entries
    //memset(chainMemory, 0, COMPLETE_HASH_CHAIN_ALLOC_SIZE * sizeof(CompleteHashEntry));
    //chainIndex = 0;
#endif
    cudaError_t cudaStatus;

    for (int i = 0; i < numGPUs; i++)
    {
        cudaSetDevice(i);
        cudaStatus = cudaMalloc(&gpuBoard[i], sizeof(HexaBitBoardPosition) * MAX_MOVES);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for gpuBoard, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(0);
        }

        cudaStatus = cudaMalloc(&gpu_perft[i], sizeof(uint64) * MAX_MOVES);
        if (cudaStatus != cudaSuccess) 
        {
            printf("cudaMalloc failed for gpu_perft, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(0);
        }

        cudaStatus = cudaMalloc(&gpuHashes[i], sizeof(HashKey128b)*MAX_MOVES);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaMalloc failed for gpuHashes, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));
            exit(0);
        }

    }
    cudaSetDevice(0);

    printf("\n");
    InfInt perft;
    START_TIMER
    perft = perft_bb_cpu_launcher(pos, depth, "..");
    STOP_TIMER

    //printf("Perft(%02d):%20llu, time: %8g s\n", depth, perft, gTime);
    printf("Perft(%02d):%20s, time: %8g s, gpuTime: %8g s\n", depth, perft.toString().c_str(), gTime, gpuTime/numGPUs);
    gpuTime = 0;
    fflush(stdout);Time: 
    for (int i = 0; i < numGPUs; i++)
    {
        cudaSetDevice(i);
        cudaFree(gpuBoard[i]);
        cudaFree(gpu_perft[i]);
    }
    cudaSetDevice(0);

}
#endif


// either launch GPU routine directly (for no-hash perfts) or call recursive serial CPU routine for divided perfts
void perftLauncher(HexaBitBoardPosition *pos, uint32 depth, int launchDepth)
{
#if USE_TRANSPOSITION_TABLE == 1
    // split at the topmost level to get best hash table utilization
    if (depth > MIN_SPLIT_DEPTH)
        splitDepth = depth;

    if (depth >= DISK_HASH_MIN_DEPTH) {
        diskHashDepth = depth - DISK_HASH_LEVEL;
    }

    dividedPerft(pos, depth);
#else
    uint64 res;
    HexaBitBoardPosition *gpuBoard;
    uint64 *gpu_perft;
    void *serial_perft_stack;
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc(&gpuBoard, sizeof(HexaBitBoardPosition));
    if (cudaStatus != cudaSuccess) printf("cudaMalloc failed for gpuBoard, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));

    cudaStatus = cudaMalloc(&serial_perft_stack, GPU_SERIAL_PERFT_STACK_SIZE);
    if (cudaStatus != cudaSuccess) printf("cudaMalloc failed for serial_perft_stack, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));

    cudaStatus = cudaMalloc(&gpu_perft, sizeof(uint64));
    if (cudaStatus != cudaSuccess) printf("cudaMalloc failed for gpu_perft, Err id: %d, str: %s\n", cudaStatus, cudaGetErrorString(cudaStatus));

    cudaMemcpy(gpuBoard, pos, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
    cudaMemset(gpu_perft, 0, sizeof(uint64));

    EventTimer gputime;
    gputime.start();

    // gpu_perft is a single 64 bit integer which is updated using atomic adds by leaf nodes
    perft_bb_driver_gpu <<<1, 1 >>> (gpuBoard, gpu_perft, depth, serial_perft_stack, preAllocatedBufferHost[0], launchDepth);

    cudaError_t err = cudaMemcpy(&res, gpu_perft, sizeof(uint64), cudaMemcpyDeviceToHost);
    gputime.stop();

    if (err != S_OK) printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));
    printf("\nGPU Perft %d: %llu,   ", depth, res);
    fflush(stdout);
    printf("Time taken: %g seconds, nps: %llu\n", gputime.elapsed() / 1000.0, (uint64)(((double)res / gputime.elapsed())*1000.0));

    cudaFree(gpuBoard);
    cudaFree(gpu_perft);
    cudaFree(serial_perft_stack);
#endif
}

void initGPU(int gpu)
{
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(gpu);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! error: %s\n", cudaGetErrorString(cudaStatus));
        exit(0);
    }

    size_t free = 0, total = 0;
    cudaMemGetInfo(&free, &total);
    printf("\ngpu: %d, memory total: %llu, free: %llu", gpu, total, free);

    // allocate the buffer to be used by device code memory allocations
    cudaStatus = cudaMalloc(&preAllocatedBufferHost[gpu], PREALLOCATED_MEMORY_SIZE);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "error in malloc for preAllocatedBuffer, error desc: %s", cudaGetErrorString(cudaStatus));
        exit(0);
    }
    else
    {
        printf("\nAllocated preAllocatedBuffer of %llu bytes, address: %X\n", PREALLOCATED_MEMORY_SIZE, preAllocatedBufferHost[gpu]);
    }

    cudaMemset(&preAllocatedMemoryUsed, 0, sizeof(uint32));
}

uint32 estimateLaunchDepth(HexaBitBoardPosition *pos)
{
    // estimate branching factor near the root
    double perft1 = perft_bb(pos, 1);
    double perft2 = perft_bb(pos, 2);
    double perft3 = perft_bb(pos, 3);

    // this works well when the root position has very low branching factor (e.g, in case king is in check)
    float geoMean = sqrt((perft3 / perft2) * (perft2 / perft1));
    float arithMean = ((perft3 / perft2) + (perft2 / perft1)) / 2;

    float branchingFactor = (geoMean + arithMean) / 2;
    if (arithMean / geoMean > 2.0f)
    {
        printf("\nUnstable position, defaulting to launch depth = 5\n");
        return 5;
    }

    //printf("\nEstimated branching factor: %g\n", branchingFactor);

    float memLimit = PREALLOCATED_MEMORY_SIZE / 2;  // be conservative as the branching factor can increase later

    // estimated depth is log of memLimit in base 'branchingFactor'
    uint32 depth = log(memLimit) / log(branchingFactor);

    //printf("\nEstimated launch depth: %d\n", depth);

    return depth;
}

void removeNewLine(char *str)
{
    while (*str)
    {
        if (*str == '\n' || *str == '\r')
        {
            *str = 0;
            break;
        }
        str++;
    }
}

#if PERFT_RECORDS_MODE == 1
void processPerftRecords(int argc, char *argv[])
{
    int depth = 7;

    FILE *fpInp;    // input file
    FILE *fpOp;     // output file

    int i = 0;
    if (argc != 3)
    {
        printf("usage: perft14_verif <inFile> [gpu]\n");
        return;
    }
    int g = atoi(argv[2]);
    initGPU(g);
    setupHashTables128b(TransTables128b[g]);
    MoveGeneratorBitboard::init();

    char opFile[1024];
    sprintf(opFile, "%s.op", argv[1]);
    printf("filename of op: %s", opFile);

    fpInp = fopen(argv[1], "rb+");
    fpOp = fopen(opFile, "ab+");

    fseek(fpOp, 0, SEEK_SET);

    BoardPosition testBoard;

    cudaMalloc(&gpuBoard[g], sizeof(HexaBitBoardPosition) * MAX_MOVES);
    cudaMalloc(&gpu_perft[g], sizeof(uint64) * MAX_MOVES);
    cudaMalloc(&gpuHashes[g], sizeof(HashKey128b)*MAX_MOVES);

    clock_t start, end;
    start = clock();

    char line[1024];
    int j = 0;
    while (fgets(line, 1024, fpInp))
    {
#if 0
        if (_kbhit())
        {
            printf("\nPaused.. press any key to continue.\n");
            getch();
        }
#endif
        i++;
        if (fgets(opFile, 1024, fpOp))
        {
            // skip already processed records
            continue;
        }

        Utils::readFENString(line, &testBoard);
        HexaBitBoardPosition testBB;

        //Utils::dispBoard(&testBoard);
        printf("\n%s", line);

        Utils::board088ToHexBB(&testBB, &testBoard);

        HashKey128b posHash128b;
        posHash128b = MoveGeneratorBitboard::computeZobristKey128b(&testBB);

        cudaMemcpy(gpuBoard[g], &testBB, sizeof(HexaBitBoardPosition), cudaMemcpyHostToDevice);
        cudaMemset(gpu_perft[g], 0, sizeof(uint64));
        cudaMemcpy(gpuHashes[g], &posHash128b, sizeof(HashKey128b), cudaMemcpyHostToDevice);

        perft_bb_gpu_simple_hash <<<1, 1 >>> (1, gpuBoard[g], gpuHashes[g], gpu_perft[g], depth, preAllocatedBufferHost[g], TransTables128b[g], true);


        uint64 res;
        cudaError_t err = cudaMemcpy(&res, gpu_perft[g], sizeof(uint64), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            printf("cudaMemcpyDeviceToHost returned %s\n", cudaGetErrorString(err));
            exit(0);
        }

        if (res == ALLSET)
        {
            res = perft_bb_last_level_launcher(&testBB, depth);
        }
        printf("GPU Perft %d: %llu", depth, res);
        // write to output file
        removeNewLine(line);

        // parse the occurence count (last number in the line)
        char *ptr = line;
        while (*ptr) ptr++;
        while (*ptr != ' ') ptr--;
        int occCount = atoi(ptr);

        fprintf(fpOp, "%s %llu %llu\n", line, res, res * occCount);
        fflush(fpOp);

        end = clock();
        double t = ((double)end - start) / CLOCKS_PER_SEC;
        j++;
        printf("\nRecords done: %d, Total: %g seconds, Avg: %g seconds\n", i, t, t / j);
        fflush(stdout);
    }


    cudaFree(gpuBoard);
    cudaFree(gpu_perft);


    fclose(fpInp);
    fclose(fpOp);

    cudaDeviceReset();

    printf("Retry launches: %d\n", numRetryLaunches);
}
#endif


void checkAndCreateDiskHash()
{
#if USE_TRANSPOSITION_TABLE == 1
#if ENABLE_DISK_HASH == 1
    // check if disk hash table is present, if not create one

    // the first 2^20 entries is the table part containing indexes to the chained part
    // the chained part immediately follow.
    // Every entry is a DiskHashEntry structure which is of 32 bytes
    //
    // size of chained part (and so the next pointer to free space) isn't 
    // stored anywhere - it's computed from file size

    int fd = open("perfthash.dat", O_WRONLY | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    if (fd != -1)
    {
        DiskHashEntry emptyEntry = {};
        for(int i = 0; i < GET_TT_SIZE_FROM_BITS(DISK_TT_BITS); i++)
            write(fd, &emptyEntry, sizeof(DiskHashEntry));
        close(fd);
    }
    
#endif
#endif
}

