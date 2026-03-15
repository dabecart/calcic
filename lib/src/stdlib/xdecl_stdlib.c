/***************************************************************************************************
 * xdecl_stdlib.c
 * 
 * Contains all declarations of global variables used by the stdlib functions.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#define COMPILING_STDLIB
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <math.h>
#include <calcilib.h>

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Pseudo-random sequence generation.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unsigned long NEXT_RAND = 1;

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Memory management.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
HeapChunkHeader *firstHeapChunk = NULL;
BlockHeader *freeListHead = NULL;
size_t heapSize = 0;
HeapChunkHeader *chunkDeallocateList[DEALLOCATE_LIST_LEN] = {NULL};
int chunkDeallocateLen = 0;

void _insertInFreeList(BlockHeader* block) {
    // This block needs to be inserted along the other free blocks from the same heap.
    if(block->heap->lastFreed == NULL) {
        // No other block from the heap was freed, add this block to the end of the free list.
        block->nextFree = NULL;
        block->prevFree = freeListHead;
        if(freeListHead != NULL) {
            freeListHead->nextFree = block;
        }
        freeListHead = block;
    }else {
        // 'prevInList' will be the previous block to the current one in the free list.
        BlockHeader *prevInList = block->heap->lastFreed;
        BlockHeader *nextInList = prevInList->nextFree;
        while((prevInList != NULL) && (prevInList->heap == block->heap) && (block < prevInList)) {
            nextInList = prevInList;
            prevInList = prevInList->prevFree;
        }

        // Insert the block to the right of the last freed block in the same heap.
        block->prevFree = prevInList;
        block->nextFree = nextInList;

        if(nextInList == NULL) {
            // The 'prevInList' was the head of the free list. Set the head now to be the current 
            // block.
            freeListHead = block;
        }else {
            nextInList->prevFree = block;
        }

        if(prevInList != NULL) {
            prevInList->nextFree = block;
        }
    }
    
    // Store the rightmost freed direction of the heap.
    if(block > block->heap->lastFreed) {
        block->heap->lastFreed = block;
    }
}

void _removeFromFreeList(BlockHeader* block) {
    // If the block being removed is the 'lastFreed' block of the heap, set it to the previous one 
    // in the same heap or NULL if there isn't one.
    if(block == block->heap->lastFreed) {
        if(block->prevFree != NULL && block->prevFree->heap == block->heap) {
            block->heap->lastFreed = block->prevFree;
        }else {
            block->heap->lastFreed = NULL;
        }
    }

    if(block->nextFree == NULL) {
        // This is the head of the free list.
        freeListHead = block->prevFree;
    }else {
        block->nextFree->prevFree = block->prevFree;
    }

    if(block->prevFree != NULL) {
        block->prevFree->nextFree = block->nextFree;
    }
}

// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
// Numeric conversion helpers.
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
unsigned long _strToInteger(const char *nptr, char **endptr, int base, 
    int* negative, int *overflow) 
{
    unsigned long number = 0;
    
    // Base can be 0, or in the range of [2,36].
    if((nptr == NULL) || !((base == 0) || ((base >= 2) && (base <= 36)))) {
        goto exit_strToInteger;
    }
    
    const char *strInput = nptr;
    *negative = 0;
    *overflow = 0;
    
    // Trim the whitespaces.
    while(isspace(*strInput)) {
        strInput++;
    }
    
    // Parse the optional sign.
    if(*strInput == '+') {
        strInput++;
    }else if(*strInput == '-') {
        *negative = 1;
        strInput++;
    }

    // Remove prefixes from different bases.
    if(base == 0) {
        // Parse the number as if it was a C integer literal.
        if(*strInput == '0') {
            strInput++;

            char afterZero = tolower(*strInput);
            if(afterZero == 'x') {
                base = 16;
                strInput++;
            }else if(afterZero == 'b') {
                base = 2;
                strInput++;
            }else {
                base = 8;
            }
        }
    }else if(base == 16) {
        // Remove the 0x if present.
        if(strInput[0] == '0' && tolower(strInput[1]) == 'x') {
            strInput += 2;
        }
    }else if(base == 2) {
        // Remove the 0b if present.
        if(strInput[0] == '0' && tolower(strInput[1]) == 'b') {
            strInput += 2;
        }
    }

    const char *beforeNumberParsing = strInput;
    
    // Remove leading zeros (they would interfere with the overflow detection).
    while(*strInput == '0') {
        strInput++;
    }

    // Parse the number.
    while(isalnum(*strInput)) {
        int representingDigit = *strInput;
        if(isalpha(representingDigit)) {
            representingDigit = toupper(representingDigit) - 'A' + 10;
        }else {
            representingDigit -= '0';
        }

        if(representingDigit >= base) {
            // The character does not belong to 'base'.
            break;
        }

        // Do not operate on 'number' if there's been an overflow already.
        if(!(*overflow)) {
            unsigned long newNumber = number*base + representingDigit;
            if(newNumber <= number) {
                // An overflow has occurred.
                *overflow = 1;
                // Continue parsing the number nonetheless.
            }
    
            number = newNumber;
        }

        // Continue with the next digit...
        strInput++;
    }

    // There must be at least a digit for this to be considered a number and advance nptr.
    if(strInput != beforeNumberParsing) {
        // We parsed the number successfully, advance nptr.
        nptr = strInput;
    }

exit_strToInteger:
    if(endptr != NULL) {
        *endptr = (char*) nptr;
    }
    return number;
}

static int strncmp_case(const char *s1, const char *s2, size_t n) {
    const unsigned char *p1 = (const unsigned char *)s1;
    const unsigned char *p2 = (const unsigned char *)s2;
    int result;

    if (p1 == p2) return 0;

    size_t iteration = 0;
    while (iteration < n && (result = tolower(*p1) - tolower(*p2)) == 0) {
        if(*p1++ == '\0') break;
        p2++;
        iteration++;
    }

    return result;
}

double _strToDecimal(const char *nptr, char **endptr,
    const int min10Exp, const int max10Exp, const double minValue, const double maxValue,
    int *negative, int *underflow, int *overflow)
{
    // TODO: Hex double.
    double number = 0;
    
    if(nptr == NULL) {
        goto exit_strtod;
    }
    
    const char *strInput = nptr;
    *negative = 0;
    *underflow = 0;
    *overflow = 0;
    
    // Trim the whitespaces.
    while(isspace(*strInput)) {
        strInput++;
    }
    
    // Parse the optional sign.
    if(*strInput == '+') {
        strInput++;
    }else if(*strInput == '-') {
        *negative = 1;
        strInput++;
    }

    const char *beforeNumberParsing = strInput;

    // Special values.
    if(strncmp_case(strInput, INFINITY_STR, strlen(INFINITY_STR)) == 0) {
        number = INFINITY;
        strInput += strlen(INFINITY_STR);
        goto success_strtod;

    }else if(strncmp_case(strInput, INF_STR, strlen(INF_STR)) == 0) {
        number = INFINITY;
        strInput += strlen(INF_STR);
        goto success_strtod;

    }else if(strncmp_case(strInput, NAN_STR, strlen(NAN_STR)) == 0) {
        number = NAN;
        strInput += strlen(NAN_STR);

        // Parse the NaN payload. We don't use it at the moment.
        if(*strInput == '(') {
            strInput++;

            while(isalnum(*strInput) || (*strInput == '_')) {
                strInput++;
            }

            // Expect a closing parenthesis.
            if(*strInput == ')'){
                strInput++;
            }else {
                // Error.
                number = 0;
                goto exit_strtod;
            }
        }
        goto success_strtod;
    }

    // Remove leading zeros.
    while(*strInput == '0') {
        strInput++;
    }

    double mantissa = 0;
    int exponent = 0;
    
    // Parse the mantissa.
    int decimalSeparator = 0;
    int foundSeparator = 0;
    while(isdigit(*strInput) | (decimalSeparator = (*strInput == '.'))) {
        if(decimalSeparator && exponent < 0) {
            // Got two decimal separators in the mantissa. We have fully parsed the number.
            break;
        }

        if(foundSeparator) {
            exponent--;
        }else {
            foundSeparator |= decimalSeparator;
        }

        if(!decimalSeparator) {
            int representingDigit = *strInput - '0';
            mantissa = mantissa * 10.0 + representingDigit;
        }

        // Continue with the next digit...
        strInput++;
    }

    if(strInput == beforeNumberParsing) {
        // We have no digits, this isn't a valid double representation.
        goto exit_strtod;
    }

    // Parse the optional exponent. Once parsed, sum it to the exponent from the previous step.
    if(tolower(*strInput) == 'e') {
        strInput++;
        
        // Parse the optional sign of the exponent.
        int negativeExponent = 0;
        if(*strInput == '+') {
            strInput++;
        }else if(*strInput == '-') {
            negativeExponent = 1;
            strInput++;
        }

        beforeNumberParsing = strInput;
        
        int strExponent = 0;
        while(isdigit(*strInput)) {
            int representingDigit = *strInput - '0';
            strExponent = strExponent * 10 + representingDigit;

            // Continue with the next digit...
            strInput++;
        }

        // If there are no digits in the exponent, the number has to be cut out before the e; in 
        // other words, the e doesn't belong to the number.
        if(strInput != beforeNumberParsing) {
            if(negativeExponent) {
                strExponent = -strExponent;
            }
    
            // Add to the previous exponent.
            exponent += strExponent;
        }
    }

    // Check if there's an overflow by comparing the exponent of the result with the input ranges.
    int log10_result = exponent + ilog10(mantissa);

    // We'll only be able to represent numbers in the range of [10^min10Exp, 10^(max10Exp-1)]. 
    // The mantissa will always be greater or equal to 1, so we'll supose we can reach 10^min10Exp,
    // but not 10^max10Exp, as we'll normally have a mantissa greater than 1.
    // This decision makes the program faster and shorter, sacrificing a bit of precision.
    if(log10_result < min10Exp) {
        number = minValue;
        *underflow = 1;
    }else if(log10_result >= max10Exp) {
        number = maxValue;
        *overflow = 1;
    }else {
        number = mantissa * pow10(exponent);
    }

success_strtod:
    // We got the mantissa and exponent, advance nptr.
    nptr = strInput;

exit_strtod:
    if(endptr != NULL) {
        *endptr = (char*) nptr;
    }
    return number;
}