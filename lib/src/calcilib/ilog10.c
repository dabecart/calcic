/***************************************************************************************************
 * ilog10.c
 * 
 * This function is part of the <calcilib.h> library.
 * 
 * This library is part of the calcic compiler, written by @dabecart. 2026.
***************************************************************************************************/

#include <calcilib.h>
#include <string.h>

int ilog10(double x) {
    if(x <= 0) {
        return -1;
    }

    // Convert double to ulong.
    unsigned long u;
    memcpy(&u, &x, sizeof(x));
    
    // Extract binary exponent: floor(log2(x)).
    // Bits 52 to 62. Exponent is encoded using offset-binary representation. For a exponent of 0, 
    // the value is 1023.
    int e_bin = (int)((u >> 52) & 0x7FF) - 1023;
    
    // log10(x) = log2(x) * log10(2), where log10(2) = 0.30103 ~ 1233/4096
    int e_dec = (e_bin * 1233) >> 12;

    // e_dec is an approximation, we have -1,+1 of error.
    double powe = pow10(e_dec);
    double powe_1 = powe * 10.0;

    if(x >= powe_1) {
        e_dec++;
    }else if(x < powe) {
        e_dec--;
    }
    
    return e_dec;
}