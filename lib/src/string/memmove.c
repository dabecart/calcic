/***************************************************************************************************
 * memmove.c
 * 
 * This function is part of the <string.h> standard library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

 #include <string.h>

void *memmove(void *s1, const void *s2, size_t n) {
    char *sc1       = (char*) s1;
    const char *sc2 = (const char*) s2;

    // As an example, let n = 5:
    // [0] [1] [2] [3] [4] [5] [6] [7]
    //              s1----------------
    //  s2---------------
    // The two regions overlap, we need to start moving from the right to the left: [4] -> [7], 
    // [3] -> [6]...
    if((s2 < s1) && (sc1 < (sc2 + n))) {
        // This is so that we can start with index = n and stop when we reach index = 0. The reason
        // to do this is that size_t is an unsigned long.
        sc1--;
        sc2--;
        for(size_t index = n; index > 0; index--) {
            sc1[index] = sc2[index];
        }
    }
    
    // [0] [1] [2] [3] [4] [5] [6] [7]
    //  s1---------------
    //              s2----------------
    // The two regions overlap and we need to start moving from the left to the right: [3] -> [0], 
    // [4] -> [1]... If the regions don't overlap, we can also move data from left to right.
    else {
        memcpy(s1, s2, n);
    }

    return s1;
}
