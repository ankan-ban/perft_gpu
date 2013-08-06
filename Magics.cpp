// magic table initialization routines

// random generators and basic idea of finding magics taken from:
// http://chessprogramming.wikispaces.com/Looking+for+Magics 


// we need move generator to generate moves using kogge-stone
// but normal Cpp code doesn't compile when __device__ variables are used, so just skip them
#define SKIP_CUDA_CODE
#include "MoveGeneratorBitboard.h"
#undef SKIP_CUDA_CODE

uint64 random_uint64() 
{
      uint64 u1, u2, u3, u4;
      u1 = (uint64)(rand()) & 0xFFFF; u2 = (uint64)(rand()) & 0xFFFF;
      u3 = (uint64)(rand()) & 0xFFFF; u4 = (uint64)(rand()) & 0xFFFF;
      return u1 | (u2 << 16) | (u3 << 32) | (u4 << 48);
}

uint64 random_uint64_sparse() 
{
    return random_uint64() & random_uint64() & random_uint64();
} 

// get i'th combo mask 
uint64 getOccCombo(uint64 mask, uint64 i)
{
    uint64 op = 0;
    while(i)
    {
        int bit = i % 2;
        uint64 opBit = MoveGeneratorBitboard::getOne(mask);
        mask &= ~opBit;
        op |= opBit * bit;
        i = i >> 1;
    }

    return op;
}

uint64 findMagicCommon(uint64 occCombos[], uint64 attacks[], uint64 attackTable[], int numCombos, int bits, uint64 preCalculatedMagic = 0, uint64 *uniqueAttackTable = NULL, uint8 *byteIndices = NULL, int *numUniqueAttacks = NULL)
{
    uint64 magic = 0;
    while(1)
    {
        if (preCalculatedMagic)
        {
            magic = preCalculatedMagic;
        }
        else
        {
            for (int i=0; i < (1 << bits); i++)
            {
                attackTable[i] = 0; // unused entry
            }

            magic = random_uint64_sparse();
            //magic = random_uint64();
        }

        // try all possible occupancy combos and check for collisions
        int i = 0;
        for (i = 0; i < numCombos; i++)
        {
            uint64 index = (magic * occCombos[i]) >> (64 - bits);
            if (attackTable[index] == 0)
            {
                uint64 attackSet = attacks[i];
                attackTable[index] = attackSet;

                // fill in the byte lookup table also
                if (numUniqueAttacks)
                {
                    // search if this attack set is already present in uniqueAttackTable
                    int j = 0;
                    for (j = 0; j < *numUniqueAttacks; j++)
                    {
                        if (uniqueAttackTable[j] == attackSet)
                        {
                            byteIndices[index] = j;
                            break;
                        }
                    }

                    // add new unique attack entry if not found
                    if (j == *numUniqueAttacks)
                    {
                        uniqueAttackTable[*numUniqueAttacks] = attackSet;
                        byteIndices[index] = *numUniqueAttacks;
                        (*numUniqueAttacks)++;
                    }
                }
            }
            else
            {
                // mismatching entry already found
                if (attackTable[index] != attacks[i])
                    break;
            }
        }

        if (i == numCombos)
            break;
        else
            assert(preCalculatedMagic == 0);
    }
    return magic;
}

uint64 findRookMagicForSquare(int square, uint64 magicAttackTable[], uint64 preCalculatedMagic, uint64 *uniqueAttackTable, uint8 *byteIndices, int *numUniqueAttacks)
{
    uint64 mask = RookAttacksMasked[square];
    uint64 thisSquare = BIT(square);

    int numBits   =  popCount(mask);
    int numCombos = (1 << numBits);
    
    uint64 occCombos[4096];     // the occupancy bits for each combination (actually permutation)
    uint64 attacks[4096];       // attacks for every combo (calculated using kogge stone)

    for (int i=0; i < numCombos; i++)
    {
        occCombos[i] = getOccCombo(mask, i);
        attacks[i]   = MoveGeneratorBitboard::rookAttacksKoggeStone(thisSquare, ~occCombos[i]);
    }

    return findMagicCommon(occCombos, attacks, magicAttackTable, numCombos, ROOK_MAGIC_BITS, preCalculatedMagic, uniqueAttackTable, byteIndices, numUniqueAttacks);

}

uint64 findBishopMagicForSquare(int square, uint64 magicAttackTable[], uint64 preCalculatedMagic, uint64 *uniqueAttackTable, uint8 *byteIndices, int *numUniqueAttacks)
{
    uint64 mask = BishopAttacksMasked[square];
    uint64 thisSquare = BIT(square);

    int numBits   =  popCount(mask);
    int numCombos = (1 << numBits);
    
    uint64 occCombos[4096];     // the occupancy bits for each combination (actually permutation)
    uint64 attacks[4096];       // attacks for every combo (calculated using kogge stone)

    for (int i=0; i < numCombos; i++)
    {
        occCombos[i] = getOccCombo(mask, i);
        attacks[i]   = MoveGeneratorBitboard::bishopAttacksKoggeStone(thisSquare, ~occCombos[i]);
    }

    return findMagicCommon(occCombos, attacks, magicAttackTable, numCombos, BISHOP_MAGIC_BITS, preCalculatedMagic, uniqueAttackTable, byteIndices, numUniqueAttacks);
}

// only for testing
#if 0
uint64 rookMagicAttackTables[64][1 << ROOK_MAGIC_BITS];
uint64 bishopMagicAttackTables[64][1 << BISHOP_MAGIC_BITS];

void findBishopMagics()
{
    printf("\n\nBishop Magics: ...");
    for (int square = A1; square <= H8; square++)
    {
        uint64 magic = findBishopMagicForSquare(square, bishopMagicAttackTables[square]);
        printf("\nSquare: %c%c, Magic: %X%X", 'A' + (square%8), '1' + (square / 8), HI(magic), LO(magic));
    }

}

void findRookMagics()
{

    printf("\n\nRook Magics: ...");
    //int square = A8;
    for (int square = A1; square <= H8; square++)
    {
        uint64 magic = findRookMagicForSquare(square, rookMagicAttackTables[square]);
        printf("\nSquare: %c%c, Magic: %X%X", 'A' + (square%8), '1' + (square / 8), HI(magic), LO(magic));
    }

}


void findMagics()
{
    srand (time(NULL));

    findBishopMagics();
    findRookMagics();
}
#endif