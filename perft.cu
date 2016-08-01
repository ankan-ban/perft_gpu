#include "device_launch_parameters.h"
#include "perft_bb.h"
#include <math.h>
#include <stdlib.h>
#include <thread>
#include <mutex>

// can't make this bigger than 6/7, as the _simple kernel (breadth first search) gets called directly
// breadth first search uses lot of memory and can can't hold bigger tree 
#define GPU_LAUNCH_DEPTH 7

// print divided perft values (subtotals) after reaching this depth
#define DIVIDED_PERFT_DEPTH 10

// try launching multiple GPU kenerls in  parallel (on multiple GPUs)
#define ENABLE_MULTIPLE_PARALLEL_LAUNCHES 0

// use multiple CPU threads to split the tree at greater depth 
// (disables simple round  robin scheduling - i.e, ENABLE_MULTIPLE_PARALLEL_LAUNCHES when enabled)
#define PARALLEL_THREAD_GPU_SPLIT 1
// depth at which work is split among multiple GPUs
#define MIN_SPLIT_DEPTH 9

// launch one level of work serially on GPU
#define ENABLE_GPU_SERIAL_LEVEL 0

// size of transposition tables for each depth
// depth1 transposition table has special purpose -> to find duplicates for 'deep' levels during BFS
// 20 bits ->  1 million entries  (16 MB)
// 25 bits -> 32  million entries (512 MB)
// 26 bits -> 64  million ...     (1 GB)
// 27 bits -> 128 million ...     (2 GB)
// 28 bits -> 256 million ...     (4 GB)
//           depth->        0      1      2      3       4       5       6       7       8       9      10      11      12      13      14      15         

const bool  shallow[] = {true,  true,  true,   true,   true,  false,  false,  false,  false,  false,  false,  false,  false,  false,  false,  false};

#if 0
// settings for Titan X (12 GB card) + 16 GB sysmem
const uint32 ttBits[] = {0,       22,    25,     27,     27,      26,     26,     27,     26,      0,      0,      0,      0,      0,      0,      0};
const bool   sysmem[] = {true, false, false,  false,   false,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
const int  sharedHashBits = 26;
#elif 0
// settings for 8 GB card (GTX 1080) + just 4 GB sysmem
const uint32 ttBits[] = { 0,       22,    25,     26,     27,      26,      26,     26,      0,      0,      0,      0,      0,      0,      0,      0 };
const bool   sysmem[] = { true, false, false,  false,   false,   false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true };
const int  sharedHashBits = 26;

#elif 0
// settings for home PC (GTX 970: 4 GB card + 8 GB sysmem)
const uint32 ttBits[] = {0,       20,    25,     26,     26,     25,     26,     25,     25,      0,      0,      0,      0,      0,      0,      0};
const bool   sysmem[] = {true, false, false,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
const int  sharedHashBits = 25;
#else
// settings for laptop (2 GB card + 16 GB sysmem)
const uint32 ttBits[] = {0,       22,     25,     27,     26,     25,     25,      0,      0,      0,      0,      0,      0,      0,      0,      0};
const bool   sysmem[] = {true, false,  false,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true,   true};
const int  sharedHashBits = 27;
#endif
const bool sharedsysmem = true;

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
                        free(TransTables128b[g].cpuTable[i]);
                    else
                        cudaFree(TransTables128b[g].hashTable[i]);
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

uint64 perft_bb_cpu_launcher(HexaBitBoardPosition *pos, uint32 depth, char *dispPrefix);

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
volatile uint64 *perftForThread[MAX_GPUs];

std::mutex criticalSection;

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
            uint64 perftVal = perft_bb_cpu_launcher((HexaBitBoardPosition *)posForThread[gpuId], depth, (char*)dispStringForThread[gpuId]);
#if 0
            // Ankan - for testing
            criticalSection.lock();
            Utils::displayBoard(&pos);
            printf("perft: %llu, gpuId: %d\n", perftVal, gpuId);
            criticalSection.unlock();
#endif
            if (depth >= DIVIDED_PERFT_DEPTH)
            {
                criticalSection.lock();
                printf("%s   %20llu\n", dispStringForThread[gpuId], perftVal);
                fflush(stdout);
                criticalSection.unlock();
            }

            *(perftForThread[gpuId]) = perftVal;
            threadStatus[gpuId] = THREAD_IDLE;
        }
    }

    threadStatus[gpuId] = THREAD_TERMINATED;
}

// launch work on multiple threads (each associated with a single GPU), 
// wait for enough parallel work is done, and only then wait for the threads to finish
uint64 perft_multi_threaded_gpu_launcher(HexaBitBoardPosition *pos, uint32 depth, char *dispPrefix)
{
    CMove genMoves[MAX_MOVES];
    HexaBitBoardPosition childPos;
    HexaBitBoardPosition childBoards[MAX_MOVES];
    char childStrings[MAX_MOVES][128];
    uint64 perftResults[MAX_MOVES];

    int nMoves = generateMoves(pos, pos->chance, genMoves);
    sortMoves(genMoves, nMoves);

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

    // wait for all threads to terminate
    for (int t = 0; t < numGPUs; t++)
    {
        while (threadStatus[t] != THREAD_IDLE);
        threadStatus[t] = THREAD_TERMINATE_REQUEST;
        while (threadStatus[t] != THREAD_TERMINATED);
        threads[t].join();
    }
    

    uint64 count = 0;
    for (int i = 0; i < nMoves; i++)
    {
        count += perftResults[i];
    }
    return count;
}


int splitDepth = MIN_SPLIT_DEPTH;
int maxMemoryUsage = 0;


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

    int nNewBoards = 0;
    uint64 count = 0;
    for (int i = 0; i < nMoves; i++)
    {
        childBoards[nNewBoards] = *pos;
        HashKey128b newHash = makeMoveAndUpdateHash(&childBoards[nNewBoards], hash, moves[i], color);

        // check in hash table
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

        hashes[nNewBoards] = newHash;
        nNewBoards++;
    }

    // copy host->device in one go
    cudaMemcpy(gpuBoard[activeGpu], childBoards, sizeof(HexaBitBoardPosition)*nNewBoards, cudaMemcpyHostToDevice);
    cudaMemset(gpu_perft[activeGpu], 0, sizeof(uint64)*nNewBoards);
    cudaMemcpy(gpuHashes[activeGpu], hashes, sizeof(HashKey128b)*nNewBoards, cudaMemcpyHostToDevice);

    // hope that these will get scheduled on GPU in tightly packed manner without much overhead
    for (int i = 0; i < nNewBoards; i++)
    {
        perft_bb_gpu_simple_hash <<<1, 1 >>> (1, &gpuBoard[activeGpu][i], &gpuHashes[activeGpu][i], &gpu_perft[activeGpu][i], depth - 1, preAllocatedBufferHost[activeGpu],
                                              TransTables128b[activeGpu], true);
        //cudaDeviceSynchronize();
    }

    // copy device-> host in one go
    cudaError_t err = cudaMemcpy(perfts, gpu_perft[activeGpu], sizeof(uint64) * nNewBoards, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        printf("\nGot error: %s\n", cudaGetErrorString(err));
    }

    // update memory usage estimation
    int currentMemUsage = 0;
    cudaError_t s = cudaMemcpyFromSymbol(&currentMemUsage, maxMemoryUsed, sizeof(int), 0, cudaMemcpyDeviceToHost);
    if (currentMemUsage > maxMemoryUsage)
    {
        maxMemoryUsage = currentMemUsage;
    }


    // collect perft results and update hash table
    for (int i = 0; i < nNewBoards; i++)
    {
        count += perfts[i];

        HashKey128b posHash128b = hashes[i];

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

    return count;
}

uint64 perft_bb_cpu_launcher(HexaBitBoardPosition *pos, uint32 depth, char *dispPrefix)
{
    HexaBitBoardPosition newPositions[MAX_MOVES];
    CMove genMoves[MAX_MOVES];
    char  dispString[128];

    HashKey128b posHash128b;
    posHash128b = MoveGeneratorBitboard::computeZobristKey128b(pos);

    // check hash table
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

    uint32 nMoves = 0;
    uint64 count = 0;

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
        sortMoves(genMoves, nMoves);

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
            uint64 childPerft = perft_bb_cpu_launcher(&newPositions[i], depth - 1, dispString);

            if (depth > DIVIDED_PERFT_DEPTH)
            {
                criticalSection.lock();
                printf("%s   %20llu\n", dispString, childPerft);
                fflush(stdout);
                criticalSection.unlock();
            }
            count += childPerft;
        }

    }

    // store in hash table
    // replace only if old entry was shallower (or of same depth)
    if (hashTable && entry.depth <= depth)
    {
        HashEntryPerft128b newEntry;
        newEntry.perftVal = count;
        newEntry.hashKey.highPart = posHash128b.highPart;
        newEntry.hashKey.lowPart = (posHash128b.lowPart & hashBits);
        newEntry.depth = depth;

        // XOR hash part with data part for lockless hashing
        newEntry.hashKey.lowPart ^= newEntry.perftVal;
        newEntry.hashKey.highPart ^= newEntry.perftVal;

        hashTable[posHash128b.lowPart & indexBits] = newEntry;
    }

    return count;
}

// called only for bigger perfts - shows move count distribution for each move
void dividedPerft(HexaBitBoardPosition *pos, uint32 depth)
{
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
    uint64 perft;
    START_TIMER
    perft = perft_bb_cpu_launcher(pos, depth, "..");
    STOP_TIMER

    printf("Perft(%02d):%20llu, time: %8g s\n", depth, perft, gTime);
    fflush(stdout);

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

    uint64 free = 0, total = 0;
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

int main(int argc, char *argv[])
{
    BoardPosition testBoard;

    int totalGPUs;
    cudaGetDeviceCount(&totalGPUs);

    printf("No of GPUs detected: %d", totalGPUs);

    if (argc >= 4)
    {
        numGPUs = atoi(argv[3]);
        if (numGPUs < 1) 
            numGPUs = 1;
        if (numGPUs > totalGPUs) 
            numGPUs = totalGPUs;
        printf("\nUsing %d GPUs\n", numGPUs);
    }
    else
    {
        numGPUs = totalGPUs;
    }

    for (int g = 0; g < numGPUs; g++)
    {
        initGPU(g);
#if USE_TRANSPOSITION_TABLE == 1
        setupHashTables128b(TransTables128b[g]);
#endif
        MoveGeneratorBitboard::init();
    }
    // set default device to device 0
    cudaSetDevice(0);

    // some test board positions from http://chessprogramming.wikispaces.com/Perft+Results
    //Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard); // start.. 20 positions
    Utils::readFENString("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", &testBoard); // position 2 (caught max bugs for me)
    //Utils::readFENString("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", &testBoard); // position 3
    //Utils::readFENString("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", &testBoard); // position 4
    //Utils::readFENString("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &testBoard); // mirror of position 4
    //Utils::readFENString("rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6", &testBoard);   // position 5
    //Utils::readFENString("3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk w - - 0 1", &testBoard); // - 218 positions.. correct!
    //Utils::readFENString("r1b1kbnr/pppp1ppp/2n1p3/6q1/6Q1/2N1P3/PPPP1PPP/R1B1KBNR w KQkq - 4 4", &testBoard); // temp test

    int minDepth = 3;
    int maxDepth = 3;
    char fen[1024];
    if (argc >= 3)
    {
        strcpy(fen, argv[1]);
        maxDepth = atoi(argv[2]);
    }
    else
    {
        printf("\nUsage perft_gpu <fen> <depth> [<launchdepth>]\n");
        printf("\nAs no paramaters were provided... running default test\n");
    }

    if (strlen(fen) > 5)
    {
        Utils::readFENString(fen, &testBoard);
    }
    else
    {
        Utils::readFENString("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", &testBoard); // start.. 20 positions
    }
    Utils::dispBoard(&testBoard);

    

    HexaBitBoardPosition testBB;
    Utils::board088ToHexBB(&testBB, &testBoard);
    Utils::boardHexBBTo088(&testBoard, &testBB);

    // launchDepth is the depth at which the driver kernel launches the work kernels
    // we decide launch depth based by estimating memory requirment of the work kernel that would be launched.

    // TODO: need more accurate method to estimate launch depth
    // branching factor near the root is not accurate. E.g, for start pos, at root branching factor = 20
    // and we estimate launch depth = 6.. which would seem quite conservative (20^6 = 64M)
    // at depth 10, the avg branching factor is nearly 30 and 30^6 = 729M which is > 10X initial estimate :-/
    
    // At launch depth 6, some launches for perft 9 start using up > 350 MB memory
    // 384 MB is not sufficient for computing perft 10 (some of the launches consume more than that)
    // and 1 GB is not sufficient for computing perft 11!
    
    uint32 launchDepth = estimateLaunchDepth(&testBB);
    launchDepth = min(launchDepth, 11); // don't go too high

    // for best performance without GPU hash (also set PREALLOCATED_MEMORY_SIZE to 3 x 768MB)
    // launchDepth = 6;    // ankan - test!

    if (argc >= 5)
    {
        launchDepth = atoi(argv[4]);
    }

    if (maxDepth < launchDepth)
    {
        launchDepth = maxDepth;
    }

    for (int depth = minDepth; depth <= maxDepth; depth++)
    {
        perftLauncher(&testBB, depth, launchDepth);
    }


    freeHashTables();

    for (int g = 0; g < numGPUs; g++)
    {
        cudaFree(preAllocatedBufferHost[g]);
        cudaDeviceReset();
    }
    
    printf("\nMax memory usage: %d bytes\n", maxMemoryUsage);

    return 0;
}
